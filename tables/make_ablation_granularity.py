"""
Ablation: conditioning granularity (hcg_L* family). Video-level conditioning (hcg_Lvid) vs.
splitting each video into contiguous chunks of L frames and conditioning per-chunk instead
(L=1000 down to L=3). This is the FAILED granularity extension (see memory
project_hcg_granularity.md): granularity is inert on real data, numbers barely move regardless
of L. Reported honestly as a negative result, not cut from the paper.

Source: tables/report/results_hcg.csv (method column hcg_L{vid,1000,300,100,30,10,3}).

Run: python tables/make_ablation_granularity.py
Writes: tables/report/ablation_granularity.tex
"""
import csv

RESULTS = "tables/report/results_hcg.csv"
OUT = "tables/report/ablation_granularity.tex"

GRANS = [("hcg_Lvid", "Whole video (L=$\\infty$)"), ("hcg_L1000", "L=1000"), ("hcg_L300", "L=300"),
         ("hcg_L100", "L=100"), ("hcg_L30", "L=30"), ("hcg_L10", "L=10"), ("hcg_L3", "L=3")]
DATASETS = [("kvasir", "Kvasir-Capsule"), ("galar_pathology", "GALAR")]


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
    lines = []
    for ds, ds_title in DATASETS:
        lines.append(f"    \\multicolumn{{5}}{{l}}{{\\textit{{{ds_title}, single model}}}} \\\\\n    \\midrule")
        for method, glabel in GRANS:
            r = results.get((ds, "single", method))
            if r is None:
                continue
            lines.append(f"    {glabel} & {fmt(r,'f1')} & {fmt(r,'mcc')} & {fmt(r,'rareRec')} & "
                         f"{fmt(r,'rarePrec')} \\\\")

    tex = (
        "\n% ===== Ablation: conditioning granularity (FAILED extension) =====\n"
        "\\begin{table*}[t]\n  \\centering\n  \\begin{tabular}{lcccc}\n    \\toprule\n"
        "    Granularity & Macro-F1 & MCC & Rare Recall & Rare Precision \\\\\n    \\midrule\n"
        + "\n".join(lines) +
        "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Ablation: video-level vs. sub-video-chunk conditioning granularity (L = "
        "frames per conditioning unit). All metrics stay within noise of each other across "
        "$L=\\infty$ (whole video) down to $L=3$ on both datasets: granularity is inert on real "
        "data, the video-level unit is not doing meaningful work relative to any finer split. "
        "Reported as a negative result; whole-video conditioning is used throughout the rest of "
        "the paper for simplicity, not because finer granularity was harmful.}\n"
        "  \\label{tab:ablation_granularity}\n\\end{table*}\n"
    )
    with open(OUT, "w") as f:
        f.write(tex)
    print(tex)


if __name__ == "__main__":
    main()
