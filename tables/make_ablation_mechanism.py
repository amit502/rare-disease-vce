"""
Ablation: conditioning mechanism, ALL FOUR evaluation settings (Kvasir/GALAR x single/ensemble).

Previously reported GALAR-single only, on the grounds that the mechanism separation is starkest
there. Reporting one block invites the (fair) reading that the block was chosen for its result,
especially since results_hcg.csv ships in the supplementary release and any reader can rebuild
the other three in two minutes. All four are shown instead, which is also more informative: the
two datasets tell genuinely different mechanistic stories, and that difference is the paper's
presence-variance argument showing up empirically.

  - On Kvasir, conditioning ALONE already contributes substantially (+0.011 single, +0.036 ens
    over temporal): videos do not contain most classes, so q_c(v) varies and conditioning has
    leverage.
  - On GALAR, conditioning alone is inert (+0.003, +0.0025): videos contain most classes, q_c(v)
    is near-constant, and the offset degenerates toward a global one.
  - Composition exceeds both components on 3 of 4 settings; on Kvasir-single EGA alone edges out
    the composition on rare recall. Reported as-is.

Mean columns are the plain mean over the four settings. NO sd is reported on them on purpose: the
spread across settings is dominated by the Kvasir-vs-GALAR difference in rare-class prevalence
(e.g. conditioning-alone differs by +0.034 between datasets), so a "+-" there would be read as
measurement noise when it is actually structural, and would understate the evidence rather than
qualify it. Per-setting columns are the honest uncertainty picture; see the main results tables
for per-video cluster bootstrap CIs.

hce_auto (val-selected among {pure EGA, pure conditioning, composed}) is included but flagged as
contaminated (val-time selection leak) and excluded from best-per-column bolding.

Source: tables/report/results_hcg.csv.

Run: python tables/make_ablation_mechanism.py
Writes: tables/report/ablation_mechanism.tex
"""
import csv

RESULTS = "tables/report/results_hcg.csv"
OUT = "tables/report/ablation_mechanism.tex"

METHODS = [
    ("mean_temporal", "Temporal"),
    ("hcg_video", "Cond.\\ (video)"),
    ("hcg_rel", "Cond.\\ (+rel.\\ wt.)"),
    ("condb", "Cond.\\ (budgeted)"),
    ("conda", "Cond.\\ (asym.)"),
    ("ega_b4", "EGA alone"),
    ("hce_auto", "Auto-select"),
    ("hce", "HCE (composed)"),
    ("pact", "PACT (constrained)"),
]
BLOCKS = [
    ("kvasir", "single", "Kv-S"),
    ("kvasir", "ens", "Kv-E"),
    ("galar_pathology", "single", "GA-S"),
    ("galar_pathology", "ens", "GA-E"),
]
# values this close are a tie, not a win: bolding a 0.0001 gap implies a distinction the
# data does not support (mean MCC separates HCE and asymmetric conditioning by exactly that).
TIE_EPS = 0.0015
CONTAMINATED = {"hce_auto"}


def load():
    rows = list(csv.DictReader(open(RESULTS)))
    return {(r["dataset"], r["scope"], r["method"]): r for r in rows}


def val(results, ds, scope, method, key):
    r = results.get((ds, scope, method))
    if r is None:
        return None
    mean_key = f"{key}_mean" if f"{key}_mean" in r else key
    return float(r[mean_key])


def main():
    results = load()
    present = [(m, lbl) for m, lbl in METHODS
               if all(val(results, ds, sc, m, "rareRec") is not None for ds, sc, _ in BLOCKS)]

    # per-setting rare recall + mean rare recall + mean F1 + mean MCC
    cells = {}
    for m, _ in present:
        rr = [val(results, ds, sc, m, "rareRec") for ds, sc, _ in BLOCKS]
        f1 = [val(results, ds, sc, m, "f1") for ds, sc, _ in BLOCKS]
        mcc = [val(results, ds, sc, m, "mcc") for ds, sc, _ in BLOCKS]
        cells[m] = rr + [sum(rr) / len(rr), sum(f1) / len(f1), sum(mcc) / len(mcc)]

    ncol = 7
    scoreable = [m for m, _ in present if m not in CONTAMINATED]
    colmax = [max(cells[m][j] for m in scoreable) for j in range(ncol)]

    lines = []
    for m, lbl in present:
        vals = []
        for j in range(ncol):
            s = f"{cells[m][j]:.3f}"
            if m not in CONTAMINATED and cells[m][j] >= colmax[j] - TIE_EPS:
                s = f"\\textbf{{{s}}}"
            vals.append(s)
        note = "$^{\\dagger}$" if m in CONTAMINATED else ""
        lines.append(f"    {lbl}{note} & " + " & ".join(vals) + " \\\\")

    hdr = " & ".join(t for _, _, t in BLOCKS)
    tex = (
        "\n% ===== Ablation: conditioning mechanism, all four evaluation settings =====\n"
        "\\begin{table*}[t]\n  \\centering\n  \\small\n"
        "  \\begin{tabular}{lcccc|ccc}\n    \\toprule\n"
        "    & \\multicolumn{4}{c|}{Rare Recall} & \\multicolumn{3}{c}{Mean over settings} \\\\\n"
        f"    Variant & {hdr} & Rare Rec. & F1 & MCC \\\\\n    \\midrule\n"
        + "\n".join(lines) +
        "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Ablation: conditioning mechanism, all four evaluation settings "
        "(Kv/GA = Kvasir-Capsule/GALAR, S/E = single model/3-seed ensemble). "
        "\\textbf{Bold} = best per column, ties within 0.0015 bolded jointly. "
        "The two datasets separate the mechanism differently: "
        "on Kvasir, conditioning alone already lifts rare recall over the temporal baseline, "
        "while on GALAR it is inert, consistent with GALAR videos containing most classes so that "
        "the video-level presence estimate barely varies. Composition exceeds both components on "
        "three of the four settings; on Kvasir single-model, EGA alone edges out the composition "
        "on rare recall. Mean columns are plain means over the four settings; no sd is given "
        "because the spread is dominated by the between-dataset prevalence difference rather than "
        "by measurement noise. $^{\\dagger}$hce\\_auto selects among the three at validation time "
        "and is contaminated (selection leak); shown for completeness, excluded from bolding, and "
        "not a reported result.}\n"
        "  \\label{tab:ablation_mechanism}\n\\end{table*}\n"
    )
    with open(OUT, "w") as f:
        f.write(tex)
    print(tex)


if __name__ == "__main__":
    main()
