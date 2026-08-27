"""
Dataset class-distribution figure: test-set prevalence per class, Kvasir-Capsule and GALAR,
log-scale (the imbalance spans ~3 orders of magnitude). Rare classes (bottom third by
prevalence, same definition used everywhere else in this project) highlighted in the same red
used for rare-class labels in the confusion-matrix figures, for a consistent visual language
across the paper's figures.

Source: tables/report/per_class_{kvasir,galar_pathology}_single_hcg.csv (support/prevalence are
identical across methods within a dataset, so just read the argmax rows).

Run: python analysis/plot_dataset_dist.py
Writes: figures/dataset_distribution.pdf
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

KVASIR_NAMES = ["Angiectasia", "Blood-fresh", "Erosion", "Erythema", "Foreign Body",
                "Ileocecal valve", "Lymphangiect.", "Normal mucosa", "Pylorus",
                "Reduced view", "Ulcer"]
RARE_COLOR = "#E63969"
COMMON_COLOR = "#3E5C8A"

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 12, "font.family": "DejaVu Sans",
    "axes.edgecolor": "#555", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": "#EAEAEA", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 14, "axes.labelsize": 12.5,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
})


def galar_names():
    names = [ln.strip().capitalize() for ln in open("tables/report/galar_classes.txt") if ln.strip()]
    return [n.replace("Lymphangioectasis", "Lymphangiect.").replace("Foreign body", "Foreign bd.")
            for n in names]


def _compact(n):
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}k"
    return str(n)


def load_dist(ds, names):
    rows = list(csv.DictReader(open(f"tables/report/per_class_{ds}_single_hcg.csv")))
    rows = [r for r in rows if r["method"] == "argmax"]
    rows.sort(key=lambda r: int(r["class"]))
    # The "single" scope's support column is summed over 2 folds x 3 seeds
    # (hcg_decode.py's _runs(ensemble=False)); the two folds are complementary
    # video-disjoint test partitions (each frame is a test frame in exactly one
    # fold), so pooling folds is correct and gives the complete dataset, but seed
    # doesn't change fold membership, so each fold's true count is tripled.
    # Divide by 3 to report the actual unique test-frame count.
    support = [int(r["support"]) // 3 for r in rows]
    prev = [float(r["prevalence"]) for r in rows]  # ratio, unaffected by the x3
    rare = [int(r["rare"]) for r in rows]
    assert len(names) == len(rows)
    order = np.argsort(prev)[::-1]
    return ([names[i] for i in order], [prev[i] for i in order],
            [support[i] for i in order], [rare[i] for i in order])


def panel(ax, names, prev, support, rare, title):
    colors = [RARE_COLOR if r else COMMON_COLOR for r in rare]
    y = np.arange(len(names))
    bars = ax.barh(y, prev, color=colors, edgecolor="white", linewidth=0.6, height=0.68)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10.5)
    for lbl, r in zip(ax.get_yticklabels(), rare):
        if r:
            lbl.set_fontweight("bold"); lbl.set_color(RARE_COLOR)
    ax.invert_yaxis()
    for yi, (p, s) in enumerate(zip(prev, support)):
        pct = f"{p*100:.2f}%" if p * 100 >= 0.01 else f"{p*100:.4f}%"
        ax.text(p * 1.25, yi, f"{pct} (n={_compact(s)})", va="center", fontsize=8.8,
                color="#333333")
    ax.set_xlabel("Prevalence (log scale)")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlim(min(prev) * 0.4, max(prev) * 6)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))

    names_k, prev_k, sup_k, rare_k = load_dist("kvasir", KVASIR_NAMES)
    panel(axes[0], names_k, prev_k, sup_k, rare_k, "Kvasir-Capsule (11 classes)")

    names_g, prev_g, sup_g, rare_g = load_dist("galar_pathology", galar_names())
    panel(axes[1], names_g, prev_g, sup_g, rare_g, "GALAR (12 classes)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=COMMON_COLOR, label="Common (top 2/3)"),
               plt.Rectangle((0, 0), 1, 1, color=RARE_COLOR, label="Rare (bottom 1/3 by prevalence)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04), fontsize=12)

    fig.tight_layout(rect=[0, 0.05, 1, 1], w_pad=4.0)
    out = "figures/dataset_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
