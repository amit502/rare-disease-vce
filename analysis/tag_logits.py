"""
Splice (video_id, frame_idx) tags into ALREADY-COLLECTED GALAR pathology logit npz,
so the sequence-aware (temporal) decoder can smooth within-video WITHOUT re-forwarding.

Safe because collection used frac=1.0 (no subsample, deterministic): the logit rows are
exactly the split-CSV rows whose path is in the tar index, in CSV order. We replicate
that same filter to get the tags and assert the row count matches the saved logits
(a hard alignment check) before writing them back.

Run as a CPU job (amit PVC):
  python analysis/tag_logits.py --splits-dir /pvc/results/galar_pathology_splits \\
      --index /pvc/results/galar_tar_index.pkl --logits-dir /pvc/results/logits \\
      --exp densenet201_swint_galar_pathology_focal --folds 0,1 --seeds 0,1,42
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Self-contained copies of datasets.galar helpers (importing the package pulls in
# datasets/__init__ -> kvasir -> iterstrat, which the decode image does not have).
# These MUST stay identical to datasets.galar so the row filter matches collection.
def _norm_key(name):
    return name.lstrip("./")


def _frame_num(path):
    try:
        return int(path.rsplit("frame_", 1)[1].split(".")[0])
    except (IndexError, ValueError):
        return -1


def tags_for(csv_path, index):
    df = pd.read_csv(csv_path)
    paths = df["path"].map(_norm_key).tolist()
    keep = [p for p in paths if p in index]          # same filter as GalarLabelDataset
    groups = np.array([p.split("/", 1)[0] for p in keep])
    frames = np.array([_frame_num(p) for p in keep], dtype=np.int32)
    return groups, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="/pvc/results/galar_pathology_splits")
    ap.add_argument("--index", default="/pvc/results/galar_tar_index.pkl")
    ap.add_argument("--logits-dir", default="/pvc/results/logits")
    ap.add_argument("--exp", default="densenet201_swint_galar_pathology_focal")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--seeds", default="0,1,42")
    a = ap.parse_args()

    print("loading tar index ...", flush=True)
    with open(a.index, "rb") as f:
        index = pickle.load(f)["index"]

    for fold in [int(x) for x in a.folds.split(",")]:
        sd = os.path.join(a.splits_dir, f"split_{fold}")
        vg, vf = tags_for(os.path.join(sd, "val.csv"), index)
        tg, tf = tags_for(os.path.join(sd, "test.csv"), index)
        print(f"fold {fold}: val tags {len(vg)}  test tags {len(tg)}", flush=True)
        for seed in [int(x) for x in a.seeds.split(",")]:
            fp = os.path.join(a.logits_dir, f"{a.exp}_f{fold}_seed{seed}.npz")
            z = dict(np.load(fp))
            assert len(vg) == z["vL"].shape[0], f"val misalign f{fold}s{seed}: {len(vg)} vs {z['vL'].shape[0]}"
            assert len(tg) == z["tL"].shape[0], f"test misalign f{fold}s{seed}: {len(tg)} vs {z['tL'].shape[0]}"
            z.update(vgroup=vg, vframe=vf, tgroup=tg, tframe=tf)
            np.savez(fp, **z)
            print(f"  tagged {os.path.basename(fp)}  (val {z['vL'].shape[0]}, test {z['tL'].shape[0]})", flush=True)
    print("DONE tag_logits", flush=True)


if __name__ == "__main__":
    main()
