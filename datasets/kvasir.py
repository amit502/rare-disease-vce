import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .split import stratified_video_split, official_split

# Exact class names as they appear in finding_class column and directory names
CLASSES = [
    "Angiectasia",
    "Blood - fresh",
    "Erosion",
    "Erythema",
    "Foreign Body",
    "Ileocecal valve",
    "Lymphangiectasia",
    "Normal clean mucosa",
    "Pylorus",
    "Reduced Mucosal View",
    "Ulcer",
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

# Rare pathology tail: the classes bounded by patient scarcity (as few as 2-3
# patients). Pseudo-labelled frames from the 74 unlabelled-only videos are added
# ONLY for these classes; Normal / Pylorus / Ileocecal valve / Reduced Mucosal
# View already have ample real data and gain nothing from augmentation.
RARE_CLASSES = [
    "Angiectasia",
    "Blood - fresh",
    "Erosion",
    "Erythema",
    "Foreign Body",
    "Lymphangiectasia",
    "Ulcer",
]

# Matches official Kvasir-Capsule training augmentation (from their training scripts)
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def _build_path_index(labelled_images_dir):
    """
    Walk labelled_images/ and map every image filename to its full path.
    This handles any extraction nesting without assuming a fixed structure.
    """
    index = {}
    exts  = {".jpg", ".jpeg", ".png"}
    for p in Path(labelled_images_dir).rglob("*"):
        if p.suffix.lower() in exts:
            index[p.name] = p
    return index


class KvasirDataset(Dataset):
    """
    Kvasir-Capsule labeled image dataset.

    split_mode="official": uses official frame-level splits from the dataset repo
                           (reproduces published numbers, but has data leakage)
    split_mode="video":    video-level stratified split (honest, no leakage)

    mined_csv: path to rare_mined.csv from mine_rare_classes.py. When provided
               and split=="train", mined frames (from the 74 unlabelled videos)
               are appended to the training set. Columns: path, finding_class.
    synth_csv: path to a copy-paste synthetic manifest (generate_copypaste.py).
               When provided and split=="train", synthetic composites are
               appended, kept as a SEPARATE source from mined_csv. Same columns.
    pseudo_csv: path to a per-fold pseudo-label manifest (build_pseudo_pool.py),
               an 11-class classification of the 74-video pathology pool. When
               provided and split=="train", only the RARE_CLASSES rows are
               appended as extra pseudo-labelled tail patients (the abundant
               classes are dropped), gated by pseudo_min_conf and capped per
               source video (pseudo_per_video_cap) to preserve patient diversity.
               The 74 videos sit outside every fold, so there is no test leakage.
               Columns: path, finding_class, video_id, confidence.
    """

    def __init__(self, root, split="train", split_mode="video",
                 split_id=0, seed=42, transform=None, mined_csv=None,
                 synth_csv=None, pseudo_csv=None,
                 pseudo_min_conf=0.9, pseudo_per_video_cap=40):
        root = Path(root)
        df   = pd.read_csv(root / "metadata.csv", sep=";")
        df   = df[df["finding_class"].isin(CLASS2IDX)].drop_duplicates("filename").reset_index(drop=True)

        if split_mode in ("official", "official_clean"):
            subdir = "official_splits_clean" if split_mode == "official_clean" else "official_splits"
            split_dir = root / subdir
            train_f, val_f, test_f = official_split(
                df, split_dir, split_id, seed=seed,
                allow_download=(split_mode == "official"),
            )
            split_map = {"train": train_f, "val": val_f, "test": test_f}
            mask = df["filename"].isin(split_map[split])
            self.df = df[mask].reset_index(drop=True)
        else:
            train_v, val_v, test_v = stratified_video_split(
                df, video_col="video_id", class_col="finding_class", seed=seed
            )
            split_map = {"train": train_v, "val": val_v, "test": test_v}
            mask = df["video_id"].isin(split_map[split])
            self.df = df[mask].reset_index(drop=True)

        self.root        = root
        self.classes     = CLASSES
        self.num_classes = len(CLASSES)
        self.transform   = transform or (
            TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM
        )
        # Build a filename→path index by scanning labelled_images/ once.
        # Handles any extraction nesting (flat, class-subdir, nested subdir).
        self._path_index = _build_path_index(root / "labelled_images")

        # Extra training frames appended to the labelled train split. Two
        # SEPARATE sources, kept distinct: mined frames (from the 74 unlabelled
        # videos) and synthetic copy-paste composites. Only for split=="train".
        def _load_extra(csv_path, tag):
            rows = []
            if csv_path is not None and split == "train":
                edf = pd.read_csv(csv_path)
                edf = edf[edf["finding_class"].isin(CLASS2IDX)]
                rows = [(r["path"], CLASS2IDX[r["finding_class"]]) for _, r in edf.iterrows()]
                print(f"[KvasirDataset] appended {len(rows)} {tag} frames from {csv_path}")
            return rows

        def _load_pseudo(csv_path):
            # Rare-class only, confidence-gated, per-video-capped pseudo labels.
            rows = []
            if csv_path is None or split != "train":
                return rows
            edf = pd.read_csv(csv_path)
            edf = edf[edf["finding_class"].isin(RARE_CLASSES)]
            if "confidence" in edf.columns:
                edf = edf[edf["confidence"] >= pseudo_min_conf]
            edf = edf.sort_values("confidence", ascending=False) \
                if "confidence" in edf.columns else edf
            per_video: dict = {}
            for _, r in edf.iterrows():
                key = (r["finding_class"], str(r.get("video_id", "")))
                if per_video.get(key, 0) >= pseudo_per_video_cap:
                    continue
                per_video[key] = per_video.get(key, 0) + 1
                rows.append((r["path"], CLASS2IDX[r["finding_class"]]))
            from collections import Counter
            dist = Counter(CLASSES[i] for _, i in rows)
            print(f"[KvasirDataset] appended {len(rows)} pseudo frames from "
                  f"{csv_path} (min_conf={pseudo_min_conf}, cap={pseudo_per_video_cap}/video) "
                  f"{dict(dist)}")
            return rows

        self._mined: list[tuple[str, int]] = _load_extra(mined_csv, "mined")
        self._synth: list[tuple[str, int]] = _load_extra(synth_csv, "synthetic")
        self._pseudo: list[tuple[str, int]] = _load_pseudo(pseudo_csv)
        self._extra = self._mined + self._synth + self._pseudo

    def __len__(self):
        return len(self.df) + len(self._extra)

    def __getitem__(self, idx):
        if idx < len(self.df):
            row      = self.df.iloc[idx]
            cls_name = row["finding_class"]
            fname    = row["filename"]
            img_path = self._path_index.get(fname)
            if img_path is None:
                raise FileNotFoundError(
                    f"{fname} not found under {self.root}/labelled_images. "
                    "Re-run the download job or check extraction logs."
                )
        else:
            abs_path, cls_idx = self._extra[idx - len(self.df)]
            img = Image.open(abs_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, cls_idx

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, CLASS2IDX[cls_name]

    def class_weights(self):
        """Inverse-frequency weights for weighted cross-entropy loss."""
        counts = self.df["finding_class"].value_counts().to_dict()
        for _, cls_idx in self._extra:
            cls_name = CLASSES[cls_idx]
            counts[cls_name] = counts.get(cls_name, 0) + 1
        weights = [1.0 / counts.get(c, 1) for c in CLASSES]
        total   = sum(weights)
        weights = [w / total * len(CLASSES) for w in weights]
        return torch.tensor(weights, dtype=torch.float32)

    def sample_labels(self):
        """Per-sample int class label aligned to __getitem__ order (df rows first,
        then appended _extra). Used to build the class-balanced sampler."""
        labels = [CLASS2IDX[c] for c in self.df["finding_class"]]
        labels.extend(cls_idx for _, cls_idx in self._extra)
        return labels
