"""
Splice (video_id, frame_number) tags into ALREADY-COLLECTED Kvasir OT logit npz, so
the sequence-aware (temporal) decoder can smooth within-video WITHOUT re-forwarding.

Kvasir-Capsule labelled frames are dense (median inter-frame gap = 1), so temporal
smoothing is meaningful. We rebuild KvasirDataset per (fold, seed) with the SAME params
collection used (shuffle=False -> self.df is the exact eval order), read video_id +
frame_number, and write them back into the npz after asserting the row count matches
the saved logits. Non-destructive: vL/vY/tL/tY are untouched, so the OT numbers stand.

Run as a CPU job (amit PVC, needs pandas + iterative-stratification):
  python analysis/tag_logits_kvasir.py --root /pvc/kvasir-capsule \\
      --logits-dir /pvc/results/logits --exp densenet201_swint_otdecode_official \\
      --split-mode official --folds 0,1 --seeds 0,1,42
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# We only need self.df (order + video_id/frame_number), NOT __getitem__, so skip the
# 47k-file labelled_images scan (slow on CephFS) by stubbing its path-index builder.
import datasets.kvasir as _kv
_kv._build_path_index = lambda d: {}
from datasets.kvasir import KvasirDataset


def df_tags(ds):
    g = ds.df["video_id"].astype(str).to_numpy()
    f = ds.df["frame_number"].astype(np.int64).to_numpy().astype(np.int32)
    return g, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/pvc/kvasir-capsule")
    ap.add_argument("--logits-dir", default="/pvc/results/logits")
    ap.add_argument("--exp", default="densenet201_swint_otdecode_official")
    ap.add_argument("--split-mode", default="official")
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--seeds", default="0,1,42")
    a = ap.parse_args()

    for fold in [int(x) for x in a.folds.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            fp = os.path.join(a.logits_dir, f"{a.exp}_f{fold}_seed{seed}.npz")
            if not os.path.exists(fp):
                print(f"SKIP missing {fp}", flush=True); continue
            vds = KvasirDataset(a.root, split="val",  split_mode=a.split_mode, split_id=fold, seed=seed)
            tds = KvasirDataset(a.root, split="test", split_mode=a.split_mode, split_id=fold, seed=seed)
            vg, vf = df_tags(vds)
            tg, tf = df_tags(tds)
            z = dict(np.load(fp, allow_pickle=True))   # some Kvasir npz carry an object-dtype key
            assert len(vg) == z["vL"].shape[0], f"val misalign f{fold}s{seed}: {len(vg)} vs {z['vL'].shape[0]}"
            assert len(tg) == z["tL"].shape[0], f"test misalign f{fold}s{seed}: {len(tg)} vs {z['tL'].shape[0]}"
            z.update(vgroup=vg, vframe=vf, tgroup=tg, tframe=tf)
            np.savez(fp, **z)
            print(f"tagged {os.path.basename(fp)}  (val {len(vg)}, test {len(tg)})", flush=True)
    print("DONE tag_logits_kvasir", flush=True)


if __name__ == "__main__":
    main()
