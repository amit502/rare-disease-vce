"""
Ablation: class-balanced sampler vs focal on Kvasir, for HCE/PACT, both scopes. This is a
NEGATIVE/inconclusive result -- cbsampler's higher point-estimate rare recall is driven by a
couple of outlier seeds, not a consistent effect, and doesn't survive a direct per-video
bootstrap comparison against focal (see compare_cbsampler_vs_focal.csv). Reported honestly as
such, not as a win.

Sources: results_hcg.csv (focal), results_hcg_kvasir_cbsampler.csv (cbsampler),
compare_cbsampler_vs_focal.csv (the direct head-to-head test -- this is the number that matters
for the ablation's claim, not either config's own vs-temporal significance).

Run: python tables/make_ablation_cbsampler.py
Writes: tables/report/ablation_cbsampler.tex
"""
import csv

FOCAL = "tables/report/results_hcg.csv"
CBSAMPLER = "tables/report/results_hcg_kvasir_cbsampler.csv"
COMPARE = "tables/report/compare_cbsampler_vs_focal.csv"
OUT = "tables/report/ablation_cbsampler.tex"

METHODS = [("hce", "HCE"), ("pact", "PACT")]
MIN_DELTA = {"F1": 0.005, "rareRec": 0.003}


def load_results(path, ds):
    rows = list(csv.DictReader(open(path)))
    return {(r["scope"], r["method"]): r for r in rows if r["dataset"] == ds}


def load_compare():
    rows = list(csv.DictReader(open(COMPARE)))
    out = {}
    for r in rows:
        out[(r["scope"], r["method"], r["metric"])] = (float(r["delta"]), float(r["pvalue"]),
                                                          float(r["ci_lo"]), float(r["ci_hi"]))
    return out


def val(r, key_mean):
    mean_key = f"{key_mean}_mean" if f"{key_mean}_mean" in r else key_mean
    return float(r[mean_key])


def fmt(r, key_mean, bold=False):
    sd = r.get(f"{key_mean}_sd")
    mean_key = f"{key_mean}_mean" if f"{key_mean}_mean" in r else key_mean
    s = (f"{float(r[mean_key]):.4f}$\\pm${float(sd):.4f}" if sd not in (None, "")
         else f"{float(r[mean_key]):.4f}")
    return f"\\textbf{{{s}}}" if bold else s


def marker(cmp, scope, method, metric, key):
    d = cmp.get((scope, method, metric))
    if d is None:
        return ""
    delta, p, _, _ = d
    if p >= 0.05 or abs(delta) < MIN_DELTA[key]:
        return ""
    return "$^{\\uparrow}$" if delta > 0 else "$^{\\downarrow}$"


def main():
    focal = load_results(FOCAL, "kvasir")
    cbsamp = load_results(CBSAMPLER, "kvasir_cbsampler")
    cmp = load_compare()

    lines = []
    for scope, slabel in (("single", "Single model"), ("ens", "3-seed ensemble")):
        lines.append(f"    \\multicolumn{{6}}{{l}}{{\\textit{{{slabel}}}}} \\\\\n    \\midrule")
        for method, mlabel in METHODS:
            rf = focal[(scope, method)]
            rc = cbsamp[(scope, method)]
            mf1 = marker(cmp, scope, method, "F1", "F1")
            mrr = marker(cmp, scope, method, "rareRec", "rareRec")
            bf = {k: val(rf, k) >= val(rc, k) for k in ("f1", "recall", "mcc", "rareRec", "rarePrec")}
            lines.append(f"    {mlabel}, focal & {fmt(rf,'f1',bf['f1'])} & {fmt(rf,'recall',bf['recall'])} & "
                         f"{fmt(rf,'mcc',bf['mcc'])} & {fmt(rf,'rareRec',bf['rareRec'])} & "
                         f"{fmt(rf,'rarePrec',bf['rarePrec'])} \\\\")
            lines.append(f"    {mlabel}, cbsampler & {fmt(rc,'f1',not bf['f1'])}{mf1} & "
                         f"{fmt(rc,'recall',not bf['recall'])} & {fmt(rc,'mcc',not bf['mcc'])} & "
                         f"{fmt(rc,'rareRec',not bf['rareRec'])}{mrr} & "
                         f"{fmt(rc,'rarePrec',not bf['rarePrec'])} \\\\")

    tex = (
        "\n% ===== Ablation: class-balanced sampler vs focal (Kvasir), HCE/PACT =====\n"
        "\\begin{table*}[t]\n  \\centering\n  \\begin{tabular}{lccccc}\n    \\toprule\n"
        "    Config & Macro-F1 & Macro Recall & MCC & Rare Recall & Rare Precision \\\\\n    \\midrule\n"
        + "\n".join(lines) +
        "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Ablation: class-balanced sampler (cbsampler) vs. focal on Kvasir, HCE/PACT. "
        "\\textbf{Bold} = numerically higher of the pair per column (not a significance claim). "
        "$\\uparrow$/$\\downarrow$ marks a DIRECT per-video bootstrap comparison between the two "
        "configs (p$<$0.05 and $|\\Delta|\\geq$0.003-0.005), not either config's own significance "
        "vs. its temporal baseline. cbsampler's higher point-estimate rare recall is driven by 2 "
        "of 6 outlier seeds and is not significant in this direct comparison (all p$\\geq$0.44); "
        "reported as an honest negative/inconclusive ablation, not a win. Focal remains the "
        "reported main-result configuration.}\n"
        "  \\label{tab:ablation_cbsampler}\n\\end{table*}\n"
    )
    with open(OUT, "w") as f:
        f.write(tex)
    print(tex)


if __name__ == "__main__":
    main()
