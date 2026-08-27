"""
Table 1: Kvasir-Capsule ARCHITECTURE comparison, all weighted CE, official split.
This is the first step of the model-selection story: sweep architectures before touching
loss/sampling. Source: tables/summary.csv, exp_name pattern "{model}_kvasir_official_f{fold}_seed{seed}".

Aggregation: average the 2 folds within each seed, then mean+-sd over the 3 seeds
(per-seed fold-average protocol, matches make_baseline_kvasir.py's loss-comparison table).

Best value per column is bolded. No significance markers here: this table compares different
TRAINED MODELS (each seed is an independently trained checkpoint), not decoders on a shared
frozen model, so the per-video cluster bootstrap used elsewhere in this project doesn't apply
directly, and a seed-level test wasn't run for this table. Significance markers are reserved
for tables where the per-video bootstrap was actually computed (main_results.tex, ablation).

vit_large is excluded: missing the f1/seed1 run (5/6 present), can't be aggregated on the
same protocol as the rest without a different n.

Run: python tables/make_arch_baseline.py
Writes: tables/report/arch_baseline_kvasir.tex
"""
import csv
import statistics as st

SUMMARY = "tables/summary.csv"
OUT = "tables/report/arch_baseline_kvasir.tex"

ARCHS = [
    ("densenet161", "DenseNet-161"),
    ("densenet201_swint", "DenseNet201 + Swin-T"),
    ("dinov2_vitl", "DINOv2 ViT-L/14"),
    ("dinov2_vits", "DINOv2 ViT-S/14"),
    ("efficientnet_b2", "EfficientNet-B2"),
    ("efficientnet_b4", "EfficientNet-B4"),
    ("focalconvnet", "FocalConvNet"),
    ("resnet152", "ResNet-152"),
    ("vit_small", "ViT-Small/16"),
]
SEEDS = [0, 1, 42]
METRICS = ["f1", "mac_rec", "mcc", "rr", "rp"]

RARE_REC = ["recall_blood___fresh", "recall_erosion", "recall_erythema"]
RARE_PREC = ["precision_blood___fresh", "precision_erosion", "precision_erythema"]


def get(rows, model, fold, seed):
    name = f"{model}_kvasir_official_f{fold}_seed{seed}"
    m = [r for r in rows if r["exp_name"] == name and r["seed"] == str(seed)]
    assert len(m) == 1, (name, len(m))
    return m[0]


def fmean(vals):
    return sum(vals) / len(vals)


def ms(vals, bold=False):
    s = f"{st.mean(vals):.4f}$\\pm${st.stdev(vals):.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def main():
    rows = list(csv.DictReader(open(SUMMARY)))
    data = {}
    for model, label in ARCHS:
        vals = {m: [] for m in METRICS}
        for seed in SEEDS:
            r0, r1 = get(rows, model, 0, seed), get(rows, model, 1, seed)
            vals["f1"].append(fmean([float(r0["f1_macro"]), float(r1["f1_macro"])]))
            vals["mac_rec"].append(fmean([float(r0["recall_macro"]), float(r1["recall_macro"])]))
            vals["mcc"].append(fmean([float(r0["mcc"]), float(r1["mcc"])]))
            rr0 = fmean([float(r0[c]) for c in RARE_REC]); rr1 = fmean([float(r1[c]) for c in RARE_REC])
            rp0 = fmean([float(r0[c]) for c in RARE_PREC]); rp1 = fmean([float(r1[c]) for c in RARE_PREC])
            vals["rr"].append(fmean([rr0, rr1])); vals["rp"].append(fmean([rp0, rp1]))
        data[label] = vals

    best = {m: max(data, key=lambda lbl: st.mean(data[lbl][m])) for m in METRICS}

    lines = []
    for model, label in ARCHS:
        cells = [ms(data[label][m], bold=(label == best[m])) for m in METRICS]
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")

    tex = (
        "\n% ===== Table 1: Kvasir-Capsule architecture comparison (weighted CE, official split) =====\n"
        "\\begin{table*}[t]\n  \\centering\n  \\begin{tabular}{lccccc}\n    \\toprule\n"
        "    Architecture & Macro-F1 & Macro Recall & MCC & Rare Recall & Rare Precision \\\\\n    \\midrule\n"
        + "\n".join(lines) +
        "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Kvasir-Capsule architecture comparison, weighted CE, official split, single model. "
        "Mean$\\pm$sd over 3 seeds (2 folds averaged per seed). Rare = Blood - fresh, Erosion, Erythema. "
        "\\textbf{Bold} = best per column. No significance testing (compares independently trained "
        "models, not decoders on a shared frozen model; see main results tables for per-video "
        "cluster bootstrap significance). DenseNet201+Swin-T is carried forward to the "
        "loss/sampling sweep.}\n"
        "  \\label{tab:arch_baseline_kvasir}\n\\end{table*}\n"
    )
    with open(OUT, "w") as f:
        f.write(tex)
    print(tex)


if __name__ == "__main__":
    main()
