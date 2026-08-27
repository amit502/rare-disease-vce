"""
Kvasir-Capsule baseline / loss-function comparison table (densenet201_swint backbone,
official split), sourced from tables/summary.csv. Aggregation: average the 2 folds within
each seed, then mean+-sd over the 3 seeds (per-seed fold-average protocol).

Rare classes = Blood - fresh, Erosion, Erythema (indices 1,2,3 in datasets/kvasir.py's
CLASSES order -- confirmed the bottom-3-by-prevalence set used throughout the decoding
pipeline, per analysis/hcg_decode.py's `rare = argsort(prev)[:K//3]`).

Run: python tables/make_baseline_kvasir.py
Writes: tables/report/baseline_kvasir.tex
"""
import csv
import statistics as st

SUMMARY = "tables/summary.csv"
OUT = "tables/report/baseline_kvasir.tex"

LOSSES = [
    ("weighted_ce", "Weighted CE (used for decoding)"),
    ("focal", "Focal loss"),
    ("ldam", "LDAM+DRW"),
]
SEEDS = [0, 1, 42]

RARE_REC = ["recall_blood___fresh", "recall_erosion", "recall_erythema"]
RARE_PREC = ["precision_blood___fresh", "precision_erosion", "precision_erythema"]
RARE_F1 = ["f1_blood___fresh", "f1_erosion", "f1_erythema"]


def get(rows, loss, fold, seed):
    name = (f"densenet201_swint_kvasir_official_f{fold}_seed{seed}" if loss == "weighted_ce"
            else f"densenet201_swint_{loss}_official_f{fold}")
    m = [r for r in rows if r["exp_name"] == name and r["seed"] == str(seed) and r["loss"] == loss]
    assert len(m) == 1, (name, seed, len(m))
    return m[0]


def fmean(vals):
    return sum(vals) / len(vals)


def ms(vals):
    return f"{st.mean(vals):.4f}$\\pm${st.stdev(vals):.4f}"


def main():
    rows = list(csv.DictReader(open(SUMMARY)))
    lines = []
    for loss, label in LOSSES:
        seed_f1, seed_mcc, seed_rr, seed_rp, seed_rf = [], [], [], [], []
        for seed in SEEDS:
            r0, r1 = get(rows, loss, 0, seed), get(rows, loss, 1, seed)
            seed_f1.append(fmean([float(r0["f1_macro"]), float(r1["f1_macro"])]))
            seed_mcc.append(fmean([float(r0["mcc"]), float(r1["mcc"])]))
            rr0 = fmean([float(r0[c]) for c in RARE_REC]); rr1 = fmean([float(r1[c]) for c in RARE_REC])
            rp0 = fmean([float(r0[c]) for c in RARE_PREC]); rp1 = fmean([float(r1[c]) for c in RARE_PREC])
            rf0 = fmean([float(r0[c]) for c in RARE_F1]); rf1 = fmean([float(r1[c]) for c in RARE_F1])
            seed_rr.append(fmean([rr0, rr1])); seed_rp.append(fmean([rp0, rp1])); seed_rf.append(fmean([rf0, rf1]))
        lines.append(f"    {label} & {ms(seed_f1)} & {ms(seed_mcc)} & {ms(seed_rr)} & "
                     f"{ms(seed_rp)} & {ms(seed_rf)} \\\\")

    tex = (
        "\n% ===== Kvasir-Capsule baselines (densenet201_swint, official split) =====\n"
        "\\begin{table}[t]\n  \\centering\n  \\begin{tabular}{lccccc}\n    \\toprule\n"
        "    Loss & Macro-F1 & MCC & Rare Recall & Rare Precision & Rare F1 \\\\\n    \\midrule\n"
        + "\n".join(lines) +
        "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Kvasir-Capsule baselines, DenseNet201+Swin-T, official split. "
        "Mean$\\pm$sd over 3 seeds (2 folds averaged per seed). Rare = Blood - fresh, Erosion, Erythema. "
        "Weighted CE is the checkpoint used for all subsequent decoding experiments.}\n"
        "  \\label{tab:baseline_kvasir}\n\\end{table}\n"
    )
    with open(OUT, "w") as f:
        f.write(tex)
    print(tex)


if __name__ == "__main__":
    main()
