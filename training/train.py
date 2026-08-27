"""
Unified training script for all baseline and custom models.

Usage examples:
  # Reproduce FocalConvNet with official split (paper numbers)
  python training/train.py --model focalconvnet --dataset kvasir \
      --data-root /pvc/kvasir-capsule --split-mode official \
      --focalconvnet-dir /workspace/FocalConvNet \
      --output-dir /pvc/results --seed 42

  # EfficientNet-B4 with honest video-level split, custom epochs
  python training/train.py --model efficientnet_b4 --dataset kvasir \
      --data-root /pvc/kvasir-capsule --split-mode video \
      --output-dir /pvc/results --seed 42 --epochs 100

All hyperparameters default to the paper's settings (see models/registry.py).
CLI overrides take precedence over defaults.
"""
import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Make repo root importable regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import KvasirDataset
from models   import build_model, MODEL_HPARAMS
from utils    import compute_metrics, ResultsLogger
from training.imbalance import build_criterion, build_sampler, seed_worker
from utils.metrics import print_report


# ── logit adjustment for long-tail training ───────────────────────────────────

class _LogitAdjust(nn.Module):
    """Menon et al. 2021: add tau*log(pi_c) to logits before CE and argmax."""
    def __init__(self, class_counts, tau=1.0):
        super().__init__()
        n = sum(class_counts)
        lp = torch.tensor([math.log(c / n) for c in class_counts], dtype=torch.float32)
        self.register_buffer("log_prior", lp)
        self.tau = tau

    def forward(self, logits):
        return logits + self.tau * self.log_prior


# ── argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="VCE baseline training")

    # Required
    p.add_argument("--model",      required=True,
                   choices=list(MODEL_HPARAMS.keys()),
                   help="Model architecture")
    p.add_argument("--dataset",    required=True, choices=["kvasir", "galar", "hyperkvasir", "gastrovision"])
    p.add_argument("--data-root",  required=True, help="Path to dataset root on PVC")
    p.add_argument("--hk-split-csv", default="/pvc/hyperkvasir/2_fold_split.csv",
                   help="hyperkvasir: official 2_fold_split.csv (file-name;class-name;split-index)")
    p.add_argument("--hk-val-frac", default=0.1, type=float,
                   help="hyperkvasir: stratified val fraction carved from the training fold")
    p.add_argument("--galar-index", default="/pvc/results/galar_tar_index.pkl",
                   help="GALAR tar offset index (from datasets.galar --build-index)")
    p.add_argument("--galar-task", default="section", choices=["section", "pathology"],
                   help="GALAR task: 'section' (5-class anatomy) or 'pathology' "
                        "(multiclass rare-disease, video-disjoint split from build_galar_pathology.py)")
    p.add_argument("--galar-splits-dir", default="/pvc/results/galar_pathology_splits",
                   help="pathology task: dir with split_k/{train,val,test}.csv + classes.txt")
    p.add_argument("--galar-max-per-class", default=25000, type=int,
                   help="GALAR train: cap frames/class (keeps all rare-class frames)")
    p.add_argument("--galar-val-cap", default=8000, type=int,
                   help="GALAR section val: cap frames/class for faster val/OT tuning")
    p.add_argument("--galar-eval-frac", default=0.1, type=float,
                   help="pathology val/test: uniform subsample fraction (preserves skew)")
    p.add_argument("--output-dir", required=True, help="Path to save results on PVC")

    # Split
    p.add_argument("--split-mode", default="video", choices=["video", "official", "official_clean"],
                   help="video: honest video-level split; official: reproduces paper numbers "
                        "(has patient leakage); official_clean: leakage-free 2-fold "
                        "(run datasets/make_clean_split.py first)")
    p.add_argument("--split-id",   default=0, type=int,
                   help="Official split index (0 or 1)")

    # Reproducibility
    p.add_argument("--seed", default=42, type=int)

    # FocalConvNet needs its repo cloned separately
    p.add_argument("--focalconvnet-dir", default=None,
                   help="Path to cloned FocalConvNet repo (required for that model)")

    # Optional VCE-pretrained backbone checkpoint (from training/pretrain.py)
    p.add_argument("--pretrained-backbone", default=None,
                   help="Path to checkpoint_best.pt from pretrain.py — loads backbone "
                        "weights before fine-tuning. Works for FLAT architectures whose "
                        "own top-level keys match the pretrained backbone's (e.g. "
                        "dinov2_vitl/vits, densenet161). Do NOT use this for "
                        "densenet201_swint, its DenseNet201 is nested under "
                        "model.densenet.*, so a plain load_state_dict on the full model "
                        "silently fails to match those keys — use "
                        "--pretrained-densenet-backbone instead.")
    p.add_argument("--pretrained-densenet-backbone", default=None,
                   help="Path to a pretrain.py checkpoint_best.pt (--backbone densenet201) "
                        "loaded into ONLY the model's '.densenet' submodule (e.g. the CNN "
                        "stream of densenet201_swint), leaving the rest of the model "
                        "(e.g. Swin-T) at its default ImageNet init. Errors if the chosen "
                        "--model has no '.densenet' submodule.")

    # Linear probe: freeze backbone, train only the classifier head
    p.add_argument("--linear-probe", action="store_true",
                   help="Freeze backbone weights; train only the final linear classifier. "
                        "Used for the pretraining pilot evaluation.")

    # Hyperparameters — all default to paper values, can be overridden
    p.add_argument("--epochs",       default=None, type=int)
    p.add_argument("--batch-size",   default=None, type=int)
    p.add_argument("--lr",           default=None, type=float)
    p.add_argument("--weight-decay", default=None, type=float)
    p.add_argument("--workers",      default=8,    type=int)
    p.add_argument("--amp", action="store_true",
                   help="mixed-precision training (A100 tensor-core throughput + half memory)")
    p.add_argument("--prefetch", default=4, type=int, help="DataLoader prefetch_factor (workers>0)")
    p.add_argument("--patience",     default=None, type=int,
                   help="Stop if val F1-macro has not improved for this many epochs. "
                        "Default: no patience stopping (run all epochs).")

    # Experiment name — auto-generated if not provided
    p.add_argument("--experiment", default=None,
                   help="Run name used for output directory and summary.csv")

    # Skip training and evaluate an existing checkpoint
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training; load existing best checkpoint and run test eval")
    p.add_argument("--checkpoint", default=None,
                   help="Explicit checkpoint path for --eval-only (defaults to the "
                        "standard best-checkpoint path for this experiment/seed)")
    p.add_argument("--mined-csv", default=None,
                   help="Path to rare_mined.csv from mine_rare_classes.py. "
                        "Appended to the training split only.")
    p.add_argument("--synth-csv", default=None,
                   help="Path to a copy-paste synthetic manifest "
                        "(generate_copypaste.py). Appended to the training split "
                        "only; kept separate from --mined-csv.")
    p.add_argument("--copypaste-onthefly", action="store_true",
                   help="Generate copy-paste composites live each epoch (train "
                        "split only) instead of a fixed pre-generated set. "
                        "Reproducible via per-worker seeded RNG.")
    p.add_argument("--synth-prob", default=0.3, type=float,
                   help="Per-Normal-frame probability of on-the-fly synthesis "
                        "(only with --copypaste-onthefly).")
    p.add_argument("--loss", default="weighted_ce",
                   choices=["ce", "weighted_ce", "focal", "la_loss", "ldam"],
                   help="Loss: ce | weighted_ce (inverse-freq) | focal (Lin 2017) | "
                        "la_loss (logit adjustment, Menon 2021) | ldam (Cao 2019, +DRW)")
    p.add_argument("--sampler", default="random", choices=["random", "class_balanced"],
                   help="random shuffle, or class-balanced WeightedRandomSampler")
    p.add_argument("--focal-gamma", default=2.0, type=float, help="focal loss gamma")
    p.add_argument("--drw-epoch", default=20, type=int,
                   help="epoch at which LDAM switches on class-balanced reweighting (DRW)")

    return p.parse_args()


def resolve_hparams(args):
    """Merge paper defaults with any CLI overrides."""
    defaults = MODEL_HPARAMS[args.model].copy()
    if args.epochs       is not None: defaults["epochs"]       = args.epochs
    if args.batch_size   is not None: defaults["batch_size"]   = args.batch_size
    if args.lr           is not None: defaults["lr"]           = args.lr
    if args.weight_decay is not None: defaults["weight_decay"] = args.weight_decay
    # Write resolved values back onto args so ResultsLogger can read them
    for k, v in defaults.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    return defaults


# ── seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    # Cross-GPU reproducibility (see train_consistency.set_seed): TF32 off so
    # Ampere/Ada match the Volta V100 FP32 baseline, deterministic algorithms +
    # fixed cuBLAS workspace for run-to-run stability. Still pin to one GPU
    # product since bit-identical across architectures is not achievable.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # Deterministic attention: swin's fused (flash/mem-efficient) SDPA has a
    # nondeterministic backward that strict determinism rejects; force the math
    # kernel, which is deterministic and reproducible per seed on one GPU type.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)      # strict (no warn_only)


# ── dataset factory ──────────────────────────────────────────────────────────

def build_datasets(args):
    common = dict(
        root=args.data_root,
        split_mode=args.split_mode,
        split_id=args.split_id,
        seed=args.seed,
    )
    if args.dataset == "kvasir":
        if getattr(args, "copypaste_onthefly", False):
            from datasets.kvasir_copypaste_live import CopyPasteLiveDataset
            train_ds = CopyPasteLiveDataset(split="train", synth_prob=args.synth_prob,
                                            **common)
        else:
            train_ds = KvasirDataset(split="train", mined_csv=args.mined_csv,
                                     synth_csv=args.synth_csv, **common)
        val_ds   = KvasirDataset(split="val",   **common)
        test_ds  = KvasirDataset(split="test",  **common)
    elif args.dataset == "hyperkvasir":
        # HyperKvasir labeled-image classification (23 classes, stills, official 2-fold).
        # Same model/loss/optim as GALAR pathology; OT-only (no temporal -> no tags).
        from datasets.hyperkvasir import HyperKvasirDataset
        hk = dict(root=args.data_root, split_id=args.split_id, seed=args.seed,
                  split_csv=args.hk_split_csv, val_frac=args.hk_val_frac)
        train_ds = HyperKvasirDataset(split="train", **hk)
        val_ds   = HyperKvasirDataset(split="val",   **hk)
        test_ds  = HyperKvasirDataset(split="test",  **hk)
    elif args.dataset == "gastrovision":
        # GastroVision (27-class GI stills, one folder per class, no split CSV). Stratified
        # 2-fold + val carve; same model/loss/optim as GALAR. OT-only (no temporal).
        from datasets.folder_stills import FolderStillsDataset
        gs = dict(root=args.data_root, split_id=args.split_id, seed=args.seed,
                  val_frac=args.hk_val_frac)
        train_ds = FolderStillsDataset(split="train", **gs)
        val_ds   = FolderStillsDataset(split="val",   **gs)
        test_ds  = FolderStillsDataset(split="test",  **gs)
    elif args.dataset == "galar" and args.galar_task == "pathology":
        # GALAR `pathology` task: single-label multiclass rare-disease, video-disjoint
        # 2-fold split (split_id = fold) built by analysis/build_galar_pathology.py.
        # Natural skew PRESERVED: train caps only the giant classes; val/test are
        # uniform subsamples (skew intact) so rarity + OT prior stay honest.
        from datasets.galar import GalarLabelDataset, load_classes
        d = args.galar_splits_dir
        idx = args.galar_index
        classes = load_classes(os.path.join(d, "classes.txt"))
        sd = os.path.join(d, f"split_{args.split_id}")
        train_ds = GalarLabelDataset(os.path.join(sd, "train.csv"), idx, classes, split="train",
                                     cap_per_class=args.galar_max_per_class, seed=args.seed)
        val_ds   = GalarLabelDataset(os.path.join(sd, "val.csv"), idx, classes, split="val",
                                     frac=args.galar_eval_frac, seed=args.seed)
        test_ds  = GalarLabelDataset(os.path.join(sd, "test.csv"), idx, classes, split="test",
                                     frac=args.galar_eval_frac, seed=args.seed)
    elif args.dataset == "galar":
        # GALAR `section` task: official 5-fold splits (split_id = fold) + shared
        # held-out test.csv; frames read from tar shards via the offset index.
        from datasets.galar import GalarSectionDataset
        sp  = os.path.join(args.data_root, "splits_publication", "section")
        idx = args.galar_index
        train_ds = GalarSectionDataset(os.path.join(sp, f"split_{args.split_id}", "train.csv"),
                                       idx, split="train",
                                       max_per_class=args.galar_max_per_class, seed=args.seed)
        val_ds   = GalarSectionDataset(os.path.join(sp, f"split_{args.split_id}", "val.csv"),
                                       idx, split="val",
                                       max_per_class=args.galar_val_cap, seed=args.seed)
        test_ds  = GalarSectionDataset(os.path.join(sp, "test.csv"), idx, split="test")
    else:
        raise NotImplementedError(f"unknown dataset {args.dataset}")
    return train_ds, val_ds, test_ds


# ── optimizer / scheduler ────────────────────────────────────────────────────

def build_optimizer(model, hp):
    if hp["optimizer"] == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=hp["lr"],
            momentum=hp.get("momentum", 0.9),
            weight_decay=hp.get("weight_decay", 0.0),
        )
    elif hp["optimizer"] == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=hp["lr"],
            weight_decay=hp.get("weight_decay", 0.01),
        )
    raise ValueError(f"Unknown optimizer: {hp['optimizer']}")


def build_scheduler(optimizer, hp, steps_per_epoch):
    if hp["scheduler"] == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",           # monitor val F1
            patience=hp.get("plateau_patience", 10),
            factor=hp.get("plateau_factor", 0.1),
            verbose=True,
        ), "plateau"
    elif hp["scheduler"] == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=hp["epochs"]
        ), "cosine"
    raise ValueError(f"Unknown scheduler: {hp['scheduler']}")


# ── one epoch ────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, logit_adjust=None,
              scaler=None):
    model.train() if train else model.eval()
    total_loss, all_labels, all_preds, all_probs = 0.0, [], [], []

    _unwrapped     = model.module if isinstance(model, nn.DataParallel) else model
    _needs_labels  = getattr(_unwrapped, "requires_labels_for_forward", False)
    # AMP only on the train pass (tensor-core throughput + half memory on A100);
    # eval stays fp32 so val/test logits match the fp32 logit-collection used by OT.
    use_amp = scaler is not None and train

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, leave=False):
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                if train and _needs_labels:
                    logits = model(imgs, labels)
                else:
                    logits = model(imgs)
                adj = logit_adjust(logits) if logit_adjust is not None else logits
                loss = criterion(adj, labels)

            if train:
                optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * len(labels)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(adj.argmax(1).cpu().numpy())
            all_probs.extend(torch.softmax(adj.detach(), dim=1).cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_labels, all_preds, all_probs


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    hp   = resolve_hparams(args)
    set_seed(args.seed)

    exp_name = args.experiment or (
        f"{args.model}_{args.dataset}_{args.split_mode}"
        + ("_linprobe" if args.linear_probe else "")
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Experiment: {exp_name}  |  Seed: {args.seed}")
    print(f"Hyperparameters: {hp}")

    # ── data ─────────────────────────────────────────────────────────────────
    train_ds, val_ds, test_ds = build_datasets(args)
    args.train_size = len(train_ds)
    args.val_size   = len(val_ds)
    args.test_size  = len(test_ds)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    _wif = seed_worker
    if getattr(args, "copypaste_onthefly", False):
        from datasets.kvasir_copypaste_live import copypaste_worker_init_fn
        _wif = copypaste_worker_init_fn
    _gen = torch.Generator(); _gen.manual_seed(args.seed)
    sampler = build_sampler(args.sampler, train_ds.sample_labels(), args.seed)
    # keep workers warm + prefetch so the A100 isn't starved on tar-seek/PNG-decode
    _lk = dict(num_workers=args.workers, pin_memory=True)
    # keep Kvasir's loader byte-identical (persistent workers can shift the per-epoch
    # augmentation RNG); enable the throughput extras only for other datasets
    if args.workers > 0 and args.dataset != "kvasir":
        _lk.update(persistent_workers=True, prefetch_factor=args.prefetch)
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"],
                              sampler=sampler, shuffle=(sampler is None),
                              drop_last=True, worker_init_fn=_wif, generator=_gen, **_lk)
    val_loader   = DataLoader(val_ds,   batch_size=hp["batch_size"], shuffle=False, **_lk)
    test_loader  = DataLoader(test_ds,  batch_size=hp["batch_size"], shuffle=False, **_lk)

    # ── model ─────────────────────────────────────────────────────────────────
    if hasattr(train_ds, "df") and "finding_class" in train_ds.df.columns:
        counts_dict  = train_ds.df["finding_class"].value_counts().to_dict()
        class_counts = [counts_dict.get(c, 0) for c in train_ds.classes]
    else:
        _l = np.asarray(train_ds.sample_labels())
        class_counts = [int((_l == i).sum()) for i in range(len(train_ds.classes))]
    print(f"Class counts (train): { {c: n for c, n in zip(train_ds.classes, class_counts)} }")

    model = build_model(
        args.model,
        num_classes=train_ds.num_classes,
        focalconvnet_dir=args.focalconvnet_dir,
        class_counts=class_counts,
    ).to(device)

    # ── frozen FFT branch: build centroids from training data ────────────────
    if hasattr(model, "fft_probe") and hasattr(model.fft_probe, "set_centroids"):
        print("Building frozen FFT centroids from training set...")
        _centroid_loader = DataLoader(
            train_ds, batch_size=hp["batch_size"],
            shuffle=False, num_workers=args.workers, pin_memory=True,
        )
        model.fft_probe.set_centroids(_centroid_loader, device)
        print("  Centroids ready.")

    # ── linear probe: freeze everything except classifier head ────────────────
    if args.linear_probe:
        for param in model.parameters():
            param.requires_grad_(False)
        for name, param in model.named_parameters():
            if any(k in name for k in ("head", "classifier", "fc")):
                param.requires_grad_(True)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Linear probe mode — {trainable:,} trainable params (head only)")

    # ── load VCE-pretrained backbone if provided ──────────────────────────────
    if args.pretrained_backbone:
        ckpt = torch.load(args.pretrained_backbone, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt["backbone_state"], strict=False)
        print(f"Loaded pretrained backbone from {args.pretrained_backbone}")
        if missing:
            print(f"  Missing keys (expected — classifier head): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    # ── load VCE-pretrained DenseNet into ONLY the model's .densenet submodule,
    #    e.g. the CNN stream of densenet201_swint, leaving the rest of the
    #    model (Swin-T) at its default ImageNet init ─────────────────────────
    if args.pretrained_densenet_backbone:
        if not hasattr(model, "densenet"):
            raise ValueError(
                f"--pretrained-densenet-backbone requires a model with a "
                f"'.densenet' submodule (e.g. densenet201_swint); got --model {args.model}"
            )
        ckpt = torch.load(args.pretrained_densenet_backbone, map_location=device)
        missing, unexpected = model.densenet.load_state_dict(ckpt["backbone_state"], strict=False)
        print(f"Loaded pretrained DenseNet backbone into model.densenet from "
              f"{args.pretrained_densenet_backbone}")
        if missing:
            print(f"  Missing keys (expected — no classifier head in this "
                  f"submodule): {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys: {unexpected}")

    # ── multi-GPU ─────────────────────────────────────────────────────────────
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)
    unwrapped = model.module if isinstance(model, nn.DataParallel) else model

    # ── loss ─────────────────────────────────────────────────────────────────
    if hasattr(train_ds, "df") and "finding_class" in train_ds.df.columns:  # Kvasir path (unchanged)
        counts_map    = train_ds.df["finding_class"].value_counts().to_dict()
        class_counts  = [counts_map.get(c, 1) for c in train_ds.classes]
        class_weights = train_ds.class_weights().to(device)
    else:                                             # dataset-agnostic (GALAR etc.)
        _labs = np.asarray(train_ds.sample_labels())
        class_counts  = [int((_labs == i).sum()) for i in range(len(train_ds.classes))]
        _w = torch.tensor([1.0 / max(c, 1) for c in class_counts], dtype=torch.float32)
        class_weights = (_w / _w.sum() * len(class_counts)).to(device)
    # shared recipe code with train_consistency.py (identical loss for fair compare)
    criterion, logit_adjust = build_criterion(
        args.loss, class_counts, class_weights, args.focal_gamma, device)
    print(f"loss={args.loss} sampler={args.sampler} "
          f"{'(DRW@'+str(args.drw_epoch)+')' if args.loss=='ldam' else ''}")

    # ── optimizer / scheduler ─────────────────────────────────────────────────
    optimizer          = build_optimizer(model, hp)
    scheduler, sched_type = build_scheduler(optimizer, hp, len(train_loader))
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == "cuda") else None
    if scaler is not None:
        print("AMP: mixed-precision training enabled")

    # ── logging ───────────────────────────────────────────────────────────────
    if args.dataset == "kvasir":
        summary_name = "summary.csv"
    elif args.dataset == "galar" and args.galar_task == "pathology":
        summary_name = "summary_galar_pathology.csv"
    else:
        summary_name = f"summary_{args.dataset}.csv"
    logger = ResultsLogger(args.output_dir, exp_name, args.seed, args, summary_name=summary_name)
    tb_dir = os.path.join(args.output_dir, "tensorboard", f"{exp_name}_seed{args.seed}")
    writer = SummaryWriter(tb_dir)

    ckpt_dir  = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = args.checkpoint or os.path.join(
        ckpt_dir, f"{exp_name}_seed{args.seed}_best.pt"
    )

    # ── eval-only: skip training, load checkpoint, run test ──────────────────
    if args.eval_only:
        print(f"Eval-only mode — loading {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device)
        unwrapped.load_state_dict(ckpt["model_state"])
        best_epoch = ckpt.get("epoch", 0)
        _, t_labels, t_preds, t_probs = run_epoch(
            model, test_loader, criterion, optimizer, device, train=False,
            logit_adjust=logit_adjust,
        )
        test_metrics = compute_metrics(t_labels, t_preds, train_ds.classes, probs=t_probs)
        print("\n=== Test Results ===")
        print_report(t_labels, t_preds, train_ds.classes)
        print(f"F1 (weighted): {test_metrics['f1_weighted']:.4f}")
        print(f"F1 (macro):    {test_metrics['f1_macro']:.4f}")
        print(f"MCC:           {test_metrics['mcc']:.4f}")
        logger.log_test(test_metrics, best_epoch)
        logger.close()
        writer.close()
        return

    # ── training loop ─────────────────────────────────────────────────────────
    best_val_f1  = 0.0
    best_epoch   = 0
    early_stop_lr = hp.get("early_stop_lr", 1e-5)

    for epoch in range(1, hp["epochs"] + 1):
        if args.loss == "ldam" and epoch == args.drw_epoch:
            criterion.reweight(class_weights)
            print(f"[DRW] epoch {epoch}: LDAM switched to class-balanced weights")
        train_loss, _, _, _              = run_epoch(model, train_loader, criterion, optimizer, device, train=True,  logit_adjust=logit_adjust)
        val_loss, v_labels, v_preds, v_probs = run_epoch(model, val_loader, criterion, optimizer, device, train=False, logit_adjust=logit_adjust)

        val_metrics = compute_metrics(v_labels, v_preds, train_ds.classes, probs=v_probs)
        # Use macro-F1 as primary metric — weighted-F1 is dominated by Normal class
        val_f1      = val_metrics["f1_macro"]

        # Tensorboard
        writer.add_scalar("Loss/train",      train_loss,                  epoch)
        writer.add_scalar("Loss/val",        val_loss,                    epoch)
        writer.add_scalar("MCC/val",         val_metrics["mcc"],          epoch)
        writer.add_scalar("F1/val_macro",    val_metrics["f1_macro"],     epoch)
        writer.add_scalar("F1/val_weighted", val_metrics["f1_weighted"],  epoch)
        for cls_name in train_ds.classes:
            safe = cls_name.replace(" ", "_").replace("-", "_").lower()
            writer.add_scalar(f"Recall_per_class/{safe}", val_metrics[f"recall_{safe}"], epoch)
            writer.add_scalar(f"F1_per_class/{safe}",     val_metrics[f"f1_{safe}"],     epoch)

        logger.log_epoch(epoch, train_loss, val_loss, val_metrics)

        print(
            f"Epoch {epoch:4d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_mcc={val_metrics['mcc']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f} | "
            f"val_recall_macro={val_metrics['recall_macro']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # Per-class recall every 10 epochs and at new best
        if epoch % 10 == 0 or val_f1 > best_val_f1:
            tag = "(new best)" if val_f1 > best_val_f1 else ""
            print(f"  Per-class recall/F1 {tag}:")
            for cls_name in train_ds.classes:
                safe = cls_name.replace(" ", "_").replace("-", "_").lower()
                print(f"    {cls_name:<30s} recall={val_metrics[f'recall_{safe}']:.3f}  f1={val_metrics[f'f1_{safe}']:.3f}")

        # Save best checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            torch.save({
                "epoch":       epoch,
                "model_state": unwrapped.state_dict(),
                "val_f1":      val_f1,
                "args":        vars(args),
            }, best_ckpt)

        # Scheduler step
        if sched_type == "plateau":
            scheduler.step(val_f1)
            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr <= early_stop_lr:
                print(f"Early stop: lr={current_lr:.2e} ≤ {early_stop_lr:.2e}")
                break
        else:
            scheduler.step()

        # Patience-based early stopping (works with any scheduler)
        if args.patience and (epoch - best_epoch) >= args.patience:
            print(f"Early stop: no improvement for {args.patience} epochs (best epoch {best_epoch}, val_f1={best_val_f1:.4f})")
            break

    # ── test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (epoch {best_epoch}, val_f1_macro={best_val_f1:.4f})")
    ckpt = torch.load(best_ckpt, map_location=device)
    unwrapped.load_state_dict(ckpt["model_state"])

    _, t_labels, t_preds, t_probs = run_epoch(model, test_loader, criterion, optimizer, device, train=False, logit_adjust=logit_adjust)
    test_metrics = compute_metrics(t_labels, t_preds, train_ds.classes, probs=t_probs)

    print("\n=== Test Results ===")
    print(f"MCC:              {test_metrics['mcc']:.4f}")
    print(f"F1  (macro):      {test_metrics['f1_macro']:.4f}")
    print(f"Recall (macro):   {test_metrics['recall_macro']:.4f}")
    print(f"F1  (weighted):   {test_metrics['f1_weighted']:.4f}  [not primary]")
    print("\nPer-class recall/F1:")
    for cls_name in train_ds.classes:
        safe = cls_name.replace(" ", "_").replace("-", "_").lower()
        print(f"  {cls_name:<30s} recall={test_metrics[f'recall_{safe}']:.3f}  f1={test_metrics[f'f1_{safe}']:.3f}")
    print()
    print_report(t_labels, t_preds, train_ds.classes)

    logger.log_test(test_metrics, best_epoch)
    logger.close()
    writer.close()
    print(f"\nResults saved to {logger.run_dir}")


if __name__ == "__main__":
    main()
