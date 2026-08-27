"""
GALAR VCE dataset loader — reads frames straight from the tar shards on CephFS.

Per the staging lessons (see memory): GALAR frames live as `shards/Galar_Frames_*.tar`
(one tar per ~10-video range), NEVER as loose files (CephFS caps small-file I/O at
~4/s). So we build a persistent name -> (shard, offset, size) index once and seek
into the tars for random access at train time. Frame entries are
`<video>/frame_NNNNNN.PNG`; the repaired 41_to_50 shard has a leading `./` that we
strip so those videos are found.

Labels come from the official `splits_publication/<task>/split_k/{train,val}.csv`
and the shared `test.csv` (60 train/val videos, 20 held-out test; patient-disjoint,
StratifiedGroupKFold), matching the dataset paper. This module implements the
`section` task (5-class anatomical multiclass) — the single-label analog to
Kvasir-Capsule, so the same softmax/CE(+focal) training and OT decoding apply.

Build the index once (any pod that mounts the galar PVC):
  python -m datasets.galar --build-index --shards-dir /galar/galar/shards \\
      --out /pvc/results/galar_tar_index.pkl
"""
import argparse
import io
import os
import pickle
import tarfile

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# section = anatomical location (single-label, one-hot in the split csv)
SECTION_CLASSES = ["mouth", "esophagus", "stomach", "small intestine", "colon"]

_NORM = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(256), transforms.Resize(224),
    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90), transforms.ToTensor(), _NORM,
])
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(256), transforms.Resize(224),
    transforms.ToTensor(), _NORM,
])


def _norm_key(name):
    """Canonical member key: strip leading './' (41_to_50 shard) and any dir noise."""
    name = name.lstrip("./")
    return name


def _frame_num(path):
    """Frame index from '<video>/frame_NNNNNN.PNG' (for temporal ordering); -1 if absent."""
    try:
        return int(path.rsplit("frame_", 1)[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


def _scan_shard(shards_dir, shard_name):
    """name -> (offset_data, size) for one shard (reads tar headers only)."""
    frames = {}
    with tarfile.open(os.path.join(shards_dir, shard_name), "r:") as tf:
        for m in tf:
            if m.isfile() and m.name.lower().endswith((".png", ".jpg", ".jpeg")):
                frames[_norm_key(m.name)] = (m.offset_data, m.size)
    return frames


def build_partial(shards_dir, shard_name, out_path):
    """Index ONE shard -> partial pickle. Run one pod per shard (tiny: cpu<=1,
    <2GB) so 8 scan in parallel; a ~75GB shard is slow to header-scan on CephFS."""
    frames = _scan_shard(shards_dir, shard_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"shard": shard_name, "shards_dir": shards_dir, "frames": frames}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"partial {shard_name}: {len(frames)} frames -> {out_path}", flush=True)


def merge_partials(parts_dir, out_path):
    """Combine per-shard partials into the final name -> (shard_idx, offset, size)."""
    import glob
    shards_dir = None
    per_shard = {}
    for p in sorted(glob.glob(os.path.join(parts_dir, "*.pkl"))):
        with open(p, "rb") as f:
            d = pickle.load(f)
        shards_dir = d["shards_dir"]
        per_shard[d["shard"]] = d["frames"]
    shards = sorted(per_shard)
    index = {}
    for si, sh in enumerate(shards):
        for name, (off, size) in per_shard[sh].items():
            index[name] = (si, off, size)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"shards": shards, "shards_dir": shards_dir, "index": index}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"merged {len(shards)} shards: {len(index)} frames -> {out_path}", flush=True)


def build_tar_index(shards_dir, out_path):
    """Sequential single-pass build (fallback; prefer build_partial + merge)."""
    shards = sorted(p for p in os.listdir(shards_dir) if p.endswith(".tar"))
    index = {}
    for si, sh in enumerate(shards):
        for name, (off, size) in _scan_shard(shards_dir, sh).items():
            index[name] = (si, off, size)
        print(f"[{si}] {sh}: total {len(index)}", flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"shards": shards, "shards_dir": shards_dir, "index": index}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote index: {len(index)} frames -> {out_path}", flush=True)


class GalarSectionDataset(Dataset):
    """GALAR anatomical-section task, read from tar shards via the offset index.

    split_csv: path to a splits_publication/section/split_k/{train,val}.csv or the
               shared test.csv (columns: the 5 one-hot section cols + `path`).
    tar_index: path to the pickle from build_tar_index.
    """
    classes = SECTION_CLASSES

    def __init__(self, split_csv, tar_index, split="train", transform=None,
                 max_per_class=None, seed=0):
        with open(tar_index, "rb") as f:
            idx = pickle.load(f)
        self.shards_dir = idx["shards_dir"]
        self.shards = idx["shards"]
        self._index = idx["index"]
        self._fh = None  # per-worker tar file handles, opened lazily after fork

        df = pd.read_csv(split_csv)
        # single label = argmax of the one-hot section columns
        present = [c for c in SECTION_CLASSES if c in df.columns]
        assert len(present) == len(SECTION_CLASSES), f"missing section cols in {split_csv}"
        labels = df[SECTION_CLASSES].to_numpy().argmax(1)
        paths = df["path"].map(_norm_key).tolist()
        # keep only frames actually present in the index (guards partial shards)
        keep = [(p, int(l)) for p, l in zip(paths, labels) if p in self._index]
        dropped = len(paths) - len(keep)
        if dropped:
            print(f"[galar] {split}: dropped {dropped}/{len(paths)} frames not in index", flush=True)

        # subsample: cap each class at max_per_class (keeps ALL frames of rare
        # classes, downsamples the abundant ones). GALAR section is ~2.6M frames,
        # colon/small-intestine dominate; this makes runs tractable and eases the
        # imbalance without touching the rare mouth/esophagus classes.
        if max_per_class:
            import numpy as np
            rng = np.random.default_rng(seed)
            by = {}
            for p, l in keep:
                by.setdefault(l, []).append(p)
            keep = []
            for l, ps in by.items():
                if len(ps) > max_per_class:
                    ps = [ps[i] for i in rng.choice(len(ps), max_per_class, replace=False)]
                keep.extend((p, l) for p in ps)
            print(f"[galar] {split}: capped to <= {max_per_class}/class -> {len(keep)} frames", flush=True)

        # Resolve to compact (shard_idx, offset, size, label) and DROP the full
        # 3.5M-entry index. Otherwise every DataLoader worker fork-copies the huge
        # dict (refcount breaks copy-on-write) and the job OOMs with many workers.
        self.samples = [(self._index[p][0], self._index[p][1], self._index[p][2], l)
                        for p, l in keep]
        self._index = None
        self.transform = transform or (TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM)

    @property
    def num_classes(self):
        return len(SECTION_CLASSES)

    def sample_labels(self):
        return [s[3] for s in self.samples]

    def _handle(self, si):
        if self._fh is None:
            self._fh = {}
        fh = self._fh.get(si)
        if fh is None:
            fh = open(os.path.join(self.shards_dir, self.shards[si]), "rb")
            self._fh[si] = fh
        return fh

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        si, off, size, label = self.samples[i]
        fh = self._handle(si)
        fh.seek(off)
        buf = fh.read(size)
        img = Image.open(io.BytesIO(buf)).convert("RGB")
        return self.transform(img), label


class GalarLabelDataset(Dataset):
    """GALAR single-label multiclass task from a `label,path` CSV (built by
    analysis/build_galar_pathology.py), read from tar shards via the offset index.

    Two subsample modes, deliberately different so the natural class imbalance is
    PRESERVED (the whole point of the rare-class task):
      - cap_per_class (train): trim each class to <= cap. Set the cap HIGH so it only
        bites the one or two giant classes (normal, blood) for tractability while
        every rarer class stays fully intact. This keeps the skew (normal >> rare),
        so argmax still collapses rare classes and OT has something to recover. It is
        NOT balancing (rare classes are far below the cap).
      - frac (val/test): uniform random subsample across ALL frames, preserving the
        exact skew, so evaluation reflects the true rarity and the OT prior stays
        correct. Never per-class here (that would rebalance and destroy the prior).
    """

    def __init__(self, split_csv, tar_index, classes, split="train", transform=None,
                 cap_per_class=None, frac=None, seed=0):
        import numpy as np
        with open(tar_index, "rb") as f:
            idx = pickle.load(f)
        self.shards_dir = idx["shards_dir"]
        self.shards = idx["shards"]
        self._index = idx["index"]
        self._fh = None  # per-worker tar file handles, opened lazily after fork
        self.classes = list(classes)

        df = pd.read_csv(split_csv)
        labels = df["label"].astype(int).tolist()
        paths = df["path"].map(_norm_key).tolist()
        keep = [(p, l) for p, l in zip(paths, labels) if p in self._index]
        dropped = len(paths) - len(keep)
        if dropped:
            print(f"[galar] {split}: dropped {dropped}/{len(paths)} frames not in index", flush=True)

        rng = np.random.default_rng(seed)
        if cap_per_class:  # trim only the giant classes; keep rare fully (train)
            by = {}
            for p, l in keep:
                by.setdefault(l, []).append(p)
            keep = []
            for l, ps in by.items():
                if len(ps) > cap_per_class:
                    ps = [ps[i] for i in rng.choice(len(ps), cap_per_class, replace=False)]
                keep.extend((p, l) for p in ps)
            print(f"[galar] {split}: capped giant classes to <= {cap_per_class} -> {len(keep)} frames", flush=True)
        elif frac and frac < 1.0:  # uniform subsample, preserve skew (val/test)
            m = rng.random(len(keep)) < frac
            keep = [kv for kv, take in zip(keep, m) if take]
            print(f"[galar] {split}: uniform frac {frac} -> {len(keep)} frames (skew preserved)", flush=True)

        self.samples = [(self._index[p][0], self._index[p][1], self._index[p][2], l)
                        for p, l in keep]
        # (video_id, frame_idx) per sample, aligned with self.samples order, for
        # sequence-aware (temporal) decoding. path = "<video>/frame_NNNNNN.PNG".
        self.groups = [p.split("/", 1)[0] for p, _ in keep]
        self.frames = [_frame_num(p) for p, _ in keep]
        self._index = None
        self.transform = transform or (TRAIN_TRANSFORM if split == "train" else EVAL_TRANSFORM)

    @property
    def num_classes(self):
        return len(self.classes)

    def sample_labels(self):
        return [s[3] for s in self.samples]

    def _handle(self, si):
        if self._fh is None:
            self._fh = {}
        fh = self._fh.get(si)
        if fh is None:
            fh = open(os.path.join(self.shards_dir, self.shards[si]), "rb")
            self._fh[si] = fh
        return fh

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        si, off, size, label = self.samples[i]
        fh = self._handle(si)
        fh.seek(off)
        buf = fh.read(size)
        img = Image.open(io.BytesIO(buf)).convert("RGB")
        return self.transform(img), label


def load_classes(path):
    """Read classes.txt (one class name per line) written by the pathology builder."""
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-index", action="store_true", help="sequential single-pass")
    ap.add_argument("--build-partial", action="store_true", help="index one shard")
    ap.add_argument("--merge", action="store_true", help="merge partials -> final index")
    ap.add_argument("--shard", default=None, help="shard filename for --build-partial")
    ap.add_argument("--shards-dir", default="/galar/galar/shards")
    ap.add_argument("--parts-dir", default="/pvc/results/galar_index_parts")
    ap.add_argument("--out", default="/pvc/results/galar_tar_index.pkl")
    a = ap.parse_args()
    if a.build_partial:
        build_partial(a.shards_dir, a.shard, os.path.join(a.parts_dir, a.shard + ".pkl"))
    elif a.merge:
        merge_partials(a.parts_dir, a.out)
    elif a.build_index:
        build_tar_index(a.shards_dir, a.out)


if __name__ == "__main__":
    main()
