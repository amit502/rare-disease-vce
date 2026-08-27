"""
Construct a single-label multiclass PATHOLOGY task from GALAR's raw per-video CSVs.

Kvasir analog with genuinely rare, hard classes (unlike the easy `section` anatomy).
Rules:
  - A frame is `normal` (0 findings), a pathology class (EXACTLY 1 finding, and that
    finding survives the class filter), or DROPPED (>=2 findings, or a filtered-out finding).
  - A pathology is kept as a class only if it appears (as a single finding) in
    >= --min-videos DISTINCT videos, so its videos can be placed on both sides of a
    video-disjoint 2-fold split.
  - Videos are partitioned into two sets, rarest-class-first greedy, so every kept
    class has at least one video on each side. 2-fold CV: fold k tests on set k,
    trains on the other; val is a seeded frame-level sample of train (for OT alpha
    tuning + prior only; the reported test set stays video-disjoint).

Outputs label,path CSVs (splits_k/{train,val,test}.csv), classes.txt, and
data_distribution.csv (per-class counts + per-video/per-split breakdown for the paper).

Run in a job (galar RO + amit RW):
  python analysis/build_galar_pathology.py --galar-root /galar/galar \\
      --out /pvc/results/galar_pathology_splits --min-videos 2 --val-frac 0.1
"""
import argparse
import glob
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

PATHOS_ALL = ["ulcer", "polyp", "active bleeding", "blood", "erythema", "erosion",
              "angiectasia", "IBD", "foreign body", "esophagitis", "varices",
              "hematin", "celiac", "cancer", "lymphangioectasis"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--galar-root", default="/galar/galar")
    ap.add_argument("--out", default="/pvc/results/galar_pathology_splits")
    ap.add_argument("--min-videos", default=2, type=int)
    ap.add_argument("--min-frames", default=0, type=int,
                    help="OFF by default: rarity in frames is the POINT (Kvasir 'Blood-fresh' is ~22 "
                         "frames). Only >=1 finding-video (--min-videos) is a real constraint. Keep 0.")
    ap.add_argument("--val-frac", default=0.1, type=float)
    ap.add_argument("--seed", default=0, type=int)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    # ── pass 1: per-frame (video, path, tag) for <=1 finding; counts + video sets ──
    counts = Counter()               # single-finding frame count per pathology
    vids_of = defaultdict(set)       # distinct videos per pathology (single-finding)
    recs = []
    for vcsv in sorted(glob.glob(os.path.join(a.galar_root, "*.csv"))):
        base = os.path.basename(vcsv)
        if base == "metadata.csv":
            continue
        vid = base[:-4]
        df = pd.read_csv(vcsv)
        pcols = [c for c in PATHOS_ALL if c in df.columns]
        n = df[pcols].sum(axis=1)
        finding = df[pcols].idxmax(axis=1).where(n == 1, "")
        single = finding[n == 1]
        counts.update(single.tolist())
        for p in single.unique():
            vids_of[p].add(vid)
        recs.append(pd.DataFrame({
            "video": vid,
            "path": vid + "/frame_" + df["frame"].astype(int).map("{:06d}".format) + ".PNG",
            "tag": finding,
        })[n <= 1])
    rec = pd.concat(recs, ignore_index=True)

    print("pathology single-finding counts (videos):")
    for p in sorted(PATHOS_ALL, key=lambda k: -counts.get(k, 0)):
        print(f"  {p:<18} frames={counts.get(p, 0):<9} videos={len(vids_of[p])}")

    # keep classes present in >= min_videos distinct videos AND >= min_frames frames
    kept = [p for p in PATHOS_ALL
            if len(vids_of[p]) >= a.min_videos and counts.get(p, 0) >= a.min_frames]

    # ── greedy 2-set video partition: each kept class gets videos on both sides ──
    S = {}
    for p in sorted(kept, key=lambda p: len(vids_of[p])):     # rarest (fewest videos) first
        c = [0, 0]
        for v in vids_of[p]:
            if v in S:
                c[S[v]] += 1
        for v in sorted(vids_of[p]):
            if v in S:
                continue
            s = 0 if c[0] <= c[1] else 1
            S[v] = s; c[s] += 1
    frames_pv = rec.groupby("video").size()
    tot = [0, 0]
    for v, s in S.items():
        tot[s] += int(frames_pv.get(v, 0))
    for v in sorted(set(frames_pv.index) - set(S), key=lambda v: -int(frames_pv.get(v, 0))):
        s = 0 if tot[0] <= tot[1] else 1
        S[v] = s; tot[s] += int(frames_pv[v])

    # finalize: drop classes that did not land on both sides
    final = []
    for p in sorted(kept, key=lambda k: -counts.get(k, 0)):
        s0 = any(S.get(v) == 0 for v in vids_of[p])
        s1 = any(S.get(v) == 1 for v in vids_of[p])
        if s0 and s1:
            final.append(p)
        else:
            print(f"  drop {p}: videos landed on one side only")
    CLASSES = ["normal"] + final
    idx = {t: i + 1 for i, t in enumerate(final)}
    rec["label"] = rec["tag"].map(lambda t: 0 if t == "" else idx.get(t, -1))
    rec["set"] = rec["video"].map(S)
    rdf = rec[rec["label"] >= 0].reset_index(drop=True)
    print(f"\nCLASSES ({len(CLASSES)}): {CLASSES}")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "classes.txt"), "w") as f:
        f.write("\n".join(CLASSES))

    # ── 2-fold CV: fold k tests on set k, trains on the other; val = frame sample of train ──
    dist_rows = []
    for k in (0, 1):
        te = rdf[rdf["set"] == k]
        tr_all = rdf[rdf["set"] == (1 - k)]
        vmask = rng.random(len(tr_all)) < a.val_frac
        val = tr_all[vmask]; train = tr_all[~vmask]
        d = os.path.join(a.out, f"split_{k}"); os.makedirs(d, exist_ok=True)
        for name, part in [("train", train), ("val", val), ("test", te)]:
            part[["label", "path"]].to_csv(os.path.join(d, f"{name}.csv"), index=False)
            for lab, cnt in part["label"].value_counts().items():
                dist_rows.append({"fold": k, "split": name, "class": CLASSES[lab], "frames": int(cnt)})
        print(f"fold {k}: train {len(train)} val {len(val)} test {len(te)}")

    # ── data distribution for the paper (per-class frames/videos + per-split counts) ──
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(os.path.join(a.out, "data_distribution_splits.csv"), index=False)
    overall = pd.DataFrame([{
        "class": CLASSES[i],
        "total_frames": int((rdf["label"] == i).sum()),
        "n_videos": (1 if i == 0 else len(vids_of[CLASSES[i]])),
        "set0_videos": sum(1 for v in (set(rdf.video) if i == 0 else vids_of[CLASSES[i]]) if S.get(v) == 0),
        "set1_videos": sum(1 for v in (set(rdf.video) if i == 0 else vids_of[CLASSES[i]]) if S.get(v) == 1),
    } for i in range(len(CLASSES))])
    overall.to_csv(os.path.join(a.out, "data_distribution.csv"), index=False)
    print("\n=== data_distribution.csv ===")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
