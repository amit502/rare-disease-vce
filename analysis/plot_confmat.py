"""
Confusion-matrix figures for the qualitative "what gets rescued" story. Two figure types per
dataset x scope:

  1. confmat_{ds}_{scope}.pdf -- row-normalized (recall-style) confusion matrices side by side
     for Argmax, OT, Temporal, HCE (ours). sns "mako" (blue/purple/teal, perceptually
     uniform), diagonal is the recall for that class under that method.
  2. confmat_rescue_{ds}_{scope}.pdf -- delta heatmap, HCE minus Argmax (row-normalized), blue
     ->white->purple diverging: blue = HCE reduces that confusion, purple = HCE adds correct
     (or off-diagonal wrong-class) mass. The diagonal row of this map answers "which classes did
     HCE actually rescue" directly.

Source: tables/report/confmat_{ds}_{scope}.npz (pooled per-method confusion matrices dumped by
hcg_decode.py), tables/report/galar_classes.txt for GALAR class names.

Run: python analysis/plot_confmat.py
Writes: figures/confmat_{ds}_{scope}.pdf, figures/confmat_rescue_{ds}_{scope}.pdf
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

KVASIR_NAMES = ["Angiectasia", "Blood-fresh", "Erosion", "Erythema", "Foreign Body",
                "Ileocecal valve", "Lymphangiect.", "Normal mucosa", "Pylorus",
                "Reduced view", "Ulcer"]
PANELS = [("argmax", "Argmax"), ("ot_alpha", "OT"), ("mean_temporal", "Temporal"),
          ("hce", "HCE (ours)")]
DATASETS = [("kvasir", "Kvasir-Capsule"), ("galar_pathology", "GALAR")]
SCOPES = [("single", "single model"), ("ens", "3-seed ensemble")]

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 11.5, "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.labelsize": 11.5,
})


def class_names(ds):
    if ds == "kvasir":
        return KVASIR_NAMES
    names = [ln.strip() for ln in open("tables/report/galar_classes.txt") if ln.strip()]
    return [n.replace("lymphangioectasis", "lymphangiect.").replace("foreign body", "foreign bd.")
            .capitalize() for n in names]


def row_normalize(cm):
    cm = cm.astype(float)
    row_sums = cm.sum(1, keepdims=True)
    return np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)


def side_by_side(ds, ds_title, scope, scope_title):
    z = np.load(f"tables/report/confmat_{ds}_{scope}.npz", allow_pickle=True)
    K = int(z["K"]); names = class_names(ds)
    rare = set(z["rare"].tolist())
    assert len(names) == K, (ds, len(names), K)

    fig = plt.figure(figsize=(4.6 * 4 + 0.9, 5.0))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.045], wspace=0.22,
                          left=0.045, right=0.965, top=0.85, bottom=0.20)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    cax = fig.add_subplot(gs[0, 4])
    for ax, (method, mlabel) in zip(axes, PANELS):
        cm = row_normalize(z[method])
        sns.heatmap(cm, ax=ax, cmap="mako", vmin=0, vmax=1, cbar=(ax is axes[-1]), cbar_ax=(cax if ax is axes[-1] else None),
                    cbar_kws={"label": "Recall (row-normalized)"} if ax is axes[-1] else None,
                    square=True, linewidths=0.3, linecolor="#22222220",
                    annot=(K <= 12), fmt=".2f", annot_kws={"size": 6.6, "color": "white"})
        ax.set_title(mlabel, fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(np.arange(K) + 0.5)
        ax.set_xticklabels(names, rotation=48, ha="right", fontsize=8.2)
        if ax is axes[0]:
            ax.set_yticks(np.arange(K) + 0.5)
            ax.set_yticklabels(names, rotation=0, fontsize=8.2)
            for i in rare:
                ax.get_yticklabels()[i].set_fontweight("bold")
                ax.get_yticklabels()[i].set_color("#E63969")
            ax.set_ylabel("True class")
        else:
            ax.set_yticks([])
            ax.set_ylabel("")
        ax.set_xlabel("Predicted class")

    fig.suptitle(f"{ds_title} — {scope_title}", fontsize=15, fontweight="bold", y=0.985)
    out = f"figures/confmat_{ds}_{scope}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def rescue_delta(ds, ds_title, scope, scope_title):
    z = np.load(f"tables/report/confmat_{ds}_{scope}.npz", allow_pickle=True)
    K = int(z["K"]); names = class_names(ds)
    rare = set(z["rare"].tolist())

    base = row_normalize(z["argmax"])
    hce = row_normalize(z["hce"])
    delta = hce - base

    cmap = sns.diverging_palette(240, 300, s=75, l=45, as_cmap=True)  # blue -> white -> purple
    vmax = max(abs(delta.min()), abs(delta.max()), 0.05)

    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    sns.heatmap(delta, ax=ax, cmap=cmap, vmin=-vmax, vmax=vmax, center=0,
                square=True, linewidths=0.3, linecolor="#22222220",
                annot=(K <= 12), fmt="+.2f", annot_kws={"size": 7.4},
                cbar_kws={"label": "HCE $-$ Argmax recall (row-normalized)"})
    ax.set_xticks(np.arange(K) + 0.5); ax.set_yticks(np.arange(K) + 0.5)
    ax.set_xticklabels(names, rotation=48, ha="right", fontsize=9)
    ax.set_yticklabels(names, rotation=0, fontsize=9)
    for i in rare:
        ax.get_yticklabels()[i].set_fontweight("bold")
        ax.get_yticklabels()[i].set_color("#E63969")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"{ds_title} — {scope_title}: HCE $-$ Argmax", fontsize=13.5,
                fontweight="bold", pad=10)

    fig.tight_layout()
    out = f"figures/confmat_rescue_{ds}_{scope}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main():
    os.makedirs("figures", exist_ok=True)
    for ds, ds_title in DATASETS:
        for scope, scope_title in SCOPES:
            side_by_side(ds, ds_title, scope, scope_title)
            rescue_delta(ds, ds_title, scope, scope_title)


if __name__ == "__main__":
    main()
