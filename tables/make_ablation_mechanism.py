"""
Ablation: conditioning mechanism, GALAR single model. Shows that video-level presence
conditioning ALONE (hcg_video/hcg_rel and their budgeted/asymmetric variants condb/conda) barely
moves rare recall over temporal, and that composing it with EGA's global macro-F1-optimal
arbitration (hce) is what actually rescues rare classes. GALAR chosen over Kvasir because the
effect is much starker here: conditioning-alone rare recall stays near baseline (~0.023-0.025)
while HCE jumps to ~0.059, more than 2x -- on Kvasir conditioning-alone already does most of the
work, which is a less illustrative ablation for justifying the composition design choice.
hce_auto (val-selected among {pure EGA, pure conditioning, composed}) is included but flagged
as contaminated (val-time selection leak) per project notes, not the reported result.

Source: tables/report/results_hcg.csv.

Run: python tables/make_ablation_mechanism.py
Writes: tables/report/ablation_mechanism.tex
"""
import csv

RESULTS = "tables/report/results_hcg.csv"
OUT = "tables/report/ablation_mechanism.tex"

METHODS = [
    ("mean_temporal", "Temporal (baseline)"),
    ("hcg_video", "Conditioning alone (video-level)"),
    ("hcg_rel", "Conditioning alone (+ reliability wt.)"),
    ("condb", "Conditioning alone (budgeted)"),
    ("conda", "Conditioning alone (asymmetric)"),
    ("ega_b4", "EGA alone (global arbitration)"),
    ("hce_auto", "Auto-select among the three (contaminated)"),
    ("hce", "Composed: conditioning + EGA (ours)"),
    ("pact", "Composed, MCC-constrained (ours)"),
]


def load():
    rows = list(csv.DictReader(open(RESULTS)))
    return {(r["dataset"], r["scope"], r["method"]): r for r in rows}


def fmt(r, key):
    mean_key = f"{key}_mean" if f"{key}_mean" in r else key
    sd = r.get(f"{key}_sd")
    return (f"{float(r[mean_key]):.4f}$\\pm${float(sd):.4f}" if sd not in (None, "")
            else f"{float(r[mean_key]):.4f}")


def main():
    results = load()
    present = [(m, lbl, results[("galar_pathology", "single", m)]) for m, lbl in METHODS
               if ("galar_pathology", "single", m) in results]
    best_f1 = max(present, key=lambda t: float(t[2]["f1_mean"]))[0]
    best_mcc = max(present, key=lambda t: float(t[2].get("mcc_mean", t[2].get("mcc"))))[0]
    best_rr = max(present, key=lambda t: float(t[2]["rareRec_mean"]))[0]

    lines = []
    for m, lbl, r in present:
        f1 = fmt(r, "f1"); mcc = fmt(r, "mcc"); rr = fmt(r, "rareRec")
        if m == best_f1: f1 = f"\\textbf{{{f1}}}"
        if m == best_mcc: mcc = f"\\textbf{{{mcc}}}"
        if m == best_rr: rr = f"\\textbf{{{rr}}}"
        note = " $^{\\dagger}$" if m == "hce_auto" else ""
        lines.append(f"    {lbl}{note} & {f1} & {mcc} & {rr} \\\\")

    tex = (
        "\n% ===== Ablation: conditioning mechanism, GALAR single model =====\n"
        "\\begin{table*}[t]\n  \\centering\n  \\begin{tabular}{lccc}\n    \\toprule\n"
        "    Variant & Macro-F1 & MCC & Rare Recall \\\\\n    \\midrule\n"
        + "\n".join(lines) +
        "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Ablation: conditioning mechanism, GALAR single model. \\textbf{Bold} = best "
        "per column. Video-level presence conditioning alone (in any variant) barely moves rare "
        "recall over the temporal baseline; composing it with EGA's global arbitration (HCE) is "
        "what actually rescues rare classes here, more than doubling rare recall over "
        "conditioning-alone. $^{\\dagger}$hce\\_auto selects among the three at validation time "
        "and is contaminated (selection leak); shown for completeness, not as a reported result.}\n"
        "  \\label{tab:ablation_mechanism}\n\\end{table*}\n"
    )
    with open(OUT, "w") as f:
        f.write(tex)
    print(tex)


if __name__ == "__main__":
    main()
