"""
Minimal val+test logit collector for a trained checkpoint.

Forwards the official-split val and test sets once and saves raw logits + labels
to {output_dir}/logits/{exp}_f{fold}_seed{seed}.npz (keys vL,vY,tL,tY), the same
format metric_optimal_decode.py uses. This lets decode_sweep.py build heterogeneous
(multi-architecture / multi-method) ensembles and decode them with zero re-forward.

Usage:
  python analysis/collect_logits.py --model densenet161 \\
      --checkpoint <ckpt> --data-root /pvc/kvasir-capsule --output-dir /pvc/results \\
      --split-id 0 --seed 42 --exp densenet161
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.kvasir import KvasirDataset


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    L, Y = [], []
    for imgs, y in loader:
        out = model(imgs.to(device))
        # .copy() so we keep NO reference to the worker's shared-memory batch tensor
        # (np.asarray(y) would view it); then drop refs so /dev/shm frees each batch.
        L.append(out.float().cpu().numpy().copy())
        Y.append(y.cpu().numpy().copy())
        del imgs, y, out
    return np.concatenate(L), np.concatenate(Y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split-mode", default="official")
    p.add_argument("--split-id", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--exp", required=True, help="Prefix for the npz (arch/method id).")
    p.add_argument("--focalconvnet-dir", default=None)
    p.add_argument("--dataset", default="kvasir", choices=["kvasir", "galar", "hyperkvasir", "gastrovision"])
    p.add_argument("--hk-split-csv", default="/pvc/hyperkvasir/2_fold_split.csv")
    p.add_argument("--hk-val-frac", default=0.1, type=float)
    p.add_argument("--galar-task", default="section", choices=["section", "pathology"])
    p.add_argument("--galar-splits-dir", default="/pvc/results/galar_pathology_splits",
                   help="pathology: dir with split_k/{val,test}.csv + classes.txt")
    p.add_argument("--galar-index", default="/pvc/results/galar_tar_index.pkl")
    p.add_argument("--galar-val-cap", default=4000, type=int,
                   help="section: cap val frames/class for faster OT tuning")
    p.add_argument("--galar-test-cap", default=0, type=int,
                   help="section: cap test frames/class (0 = full held-out test)")
    p.add_argument("--galar-eval-frac", default=0.1, type=float,
                   help="pathology: uniform val/test subsample fraction (preserves skew)")
    a = p.parse_args()

    from models import build_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.dataset == "hyperkvasir":
        from datasets.hyperkvasir import HyperKvasirDataset
        hk = dict(root=a.data_root, split_id=a.split_id, seed=a.seed,
                  split_csv=a.hk_split_csv, val_frac=a.hk_val_frac)
        val_ds  = HyperKvasirDataset(split="val",  **hk)
        test_ds = HyperKvasirDataset(split="test", **hk)
    elif a.dataset == "gastrovision":
        from datasets.folder_stills import FolderStillsDataset
        gs = dict(root=a.data_root, split_id=a.split_id, seed=a.seed, val_frac=a.hk_val_frac)
        val_ds  = FolderStillsDataset(split="val",  **gs)
        test_ds = FolderStillsDataset(split="test", **gs)
    elif a.dataset == "kvasir":
        common = dict(root=a.data_root, split_mode=a.split_mode, split_id=a.split_id, seed=a.seed)
        val_ds  = KvasirDataset(split="val",  **common)
        test_ds = KvasirDataset(split="test", **common)
    elif a.galar_task == "pathology":  # single-label multiclass, uniform-subsampled (skew intact)
        from datasets.galar import GalarLabelDataset, load_classes
        classes = load_classes(os.path.join(a.galar_splits_dir, "classes.txt"))
        sd = os.path.join(a.galar_splits_dir, f"split_{a.split_id}")
        val_ds  = GalarLabelDataset(os.path.join(sd, "val.csv"), a.galar_index, classes,
                                    split="val", frac=a.galar_eval_frac, seed=a.seed)
        test_ds = GalarLabelDataset(os.path.join(sd, "test.csv"), a.galar_index, classes,
                                    split="test", frac=a.galar_eval_frac, seed=a.seed)
    else:  # galar section: val (capped) + full held-out test.csv, read from tar index
        from datasets.galar import GalarSectionDataset
        sp = os.path.join(a.data_root, "splits_publication", "section")
        val_ds  = GalarSectionDataset(os.path.join(sp, f"split_{a.split_id}", "val.csv"),
                                      a.galar_index, split="val",
                                      max_per_class=a.galar_val_cap, seed=a.seed)
        test_ds = GalarSectionDataset(os.path.join(sp, "test.csv"), a.galar_index, split="test",
                                      max_per_class=(a.galar_test_cap or None), seed=a.seed)
    K = val_ds.num_classes
    # pin_memory off (forward-only; avoids extra pinned buildup), modest workers
    vl = DataLoader(val_ds,  batch_size=a.batch_size, num_workers=a.workers, pin_memory=False)
    tl = DataLoader(test_ds, batch_size=a.batch_size, num_workers=a.workers, pin_memory=False)

    model = build_model(a.model, num_classes=K, focalconvnet_dir=a.focalconvnet_dir).to(dev)
    ckpt = torch.load(a.checkpoint, map_location=dev)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    print(f"Loaded {a.checkpoint}")

    vL, vY = collect(model, vl, dev)
    tL, tY = collect(model, tl, dev)
    ldir = os.path.join(a.output_dir, "logits"); os.makedirs(ldir, exist_ok=True)
    out = os.path.join(ldir, f"{a.exp}_f{a.split_id}_seed{a.seed}.npz")
    save = dict(vL=vL, vY=vY, tL=tL, tY=tY)
    # eval order == dataset sample order (shuffle=False), so (video,frame) tags align
    # row-for-row with the logits -> enables within-video temporal decoding, no re-forward.
    if hasattr(val_ds, "groups") and hasattr(test_ds, "groups"):
        save.update(vgroup=np.array(val_ds.groups),  vframe=np.array(val_ds.frames,  dtype=np.int32),
                    tgroup=np.array(test_ds.groups), tframe=np.array(test_ds.frames, dtype=np.int32))
        print(f"tagged with (video,frame): val {len(val_ds.groups)} test {len(test_ds.groups)}")
    np.savez(out, **save)
    print(f"saved -> {out}  (val {vL.shape}, test {tL.shape})")


if __name__ == "__main__":
    main()
