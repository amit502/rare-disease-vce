"""
Table 3/4: decoding-method comparison, one table per dataset (Kvasir, GALAR), single model and
3-seed ensemble in the SAME table separated by a midrule. Sourced from
tables/report/results_hcg.csv + results_hcg_significance.csv (downloaded from the vce-hcg
k8s job's PVC output).

Significance marking: up/down arrow shown only if BOTH the bootstrap CI (2.5/97.5 percentile,
per-video cluster bootstrap vs. mean_temporal) excludes zero AND the delta clears a
practical-significance floor (MIN_DELTA). CI-exclusion is used instead of the raw Fisher-combined
p-value in results_hcg_significance.csv: that p-value underflows to physically implausible values
(e.g. 1e-295) whenever a single CV fold's bootstrap resamples land 100% on one side, which
disagrees with the (wider, honest) CI on the same row -- CI-exclusion is the conservative,
defensible criterion. Not shown as extra table columns here, the table is already at 5 columns.

Run: python tables/make_main_results.py
Writes: tables/report/main_results.tex
"""
import csv

RESULTS = "tables/report/results_hcg.csv"
SIG = "tables/report/results_hcg_significance.csv"

METHODS = [
    ("argmax", "Argmax"),
    ("la_tau", "LA"),
    ("ot_alpha", "OT"),
    ("mean_temporal", "Temporal (baseline)"),
    ("ega_b4", "EGA (ours)"),
    ("hce", "HCE (ours)"),
]
DATASETS = [
    ("kvasir", "Kvasir-Capsule, decoding methods", "main_kvasir"),
    ("galar_pathology", "GALAR, decoding methods", "main_galar_pathology"),
]
MIN_DELTA = {"f1": 0.005, "recall": 0.005, "mcc": 0.005, "rareRec": 0.003, "rarePrec": 0.005}


def load_results():
    rows = list(csv.DictReader(open(RESULTS)))
    return {(r["dataset"], r["scope"], r["method"]): r for r in rows}


def load_sig():
    rows = list(csv.DictReader(open(SIG)))
    out = {}
    for r in rows:
        if r["test"] != "video_cluster_bootstrap":
            continue
        out[(r["dataset"], r["scope"], r["method"], r["metric"])] = (
            float(r["delta"]), float(r["ci_lo"]), float(r["ci_hi"]))
    return out


def marker(ds, scope, method, metric, key):
    d_ci = sig.get((ds, scope, method, metric))
    if d_ci is None:
        return ""
    delta, ci_lo, ci_hi = d_ci
    if abs(delta) < MIN_DELTA[key]:
        return ""
    if ci_lo > 0:
        return "$^{\\uparrow}$"
    if ci_hi < 0:
        return "$^{\\downarrow}$"
    return ""


def rows_for(ds, scope):
    present = [(method, mlabel, results[(ds, scope, method)]) for method, mlabel in METHODS
               if (ds, scope, method) in results]
    best_f1 = max(present, key=lambda t: float(t[2]["f1_mean"]))[0]
    best_rec = max(present, key=lambda t: float(t[2].get("recall_mean", t[2].get("recall"))))[0]
    best_mcc = max(present, key=lambda t: float(t[2].get("mcc_mean", t[2].get("mcc"))))[0]
    best_rr = max(present, key=lambda t: float(t[2]["rareRec_mean"]))[0]
    best_rp = max(present, key=lambda t: float(t[2]["rarePrec_mean"]))[0]

    def cell(val_str, is_best):
        return f"\\textbf{{{val_str}}}" if is_best else val_str

    lines = []
    for method, mlabel, r in present:
        f1 = f"{float(r['f1_mean']):.4f}$\\pm${float(r['f1_sd']):.4f}"
        rec_sd = r.get("recall_sd")
        recm = (f"{float(r['recall_mean']):.4f}$\\pm${float(rec_sd):.4f}" if rec_sd not in (None, "")
                else f"{float(r.get('recall_mean', r.get('recall'))):.4f}")
        mcc_sd = r.get("mcc_sd")
        mcc = (f"{float(r['mcc_mean']):.4f}$\\pm${float(mcc_sd):.4f}" if mcc_sd not in (None, "")
               else f"{float(r.get('mcc_mean', r.get('mcc'))):.4f}")
        rr = f"{float(r['rareRec_mean']):.4f}$\\pm${float(r['rareRec_sd']):.4f}"
        rp = f"{float(r['rarePrec_mean']):.4f}$\\pm${float(r['rarePrec_sd']):.4f}"
        mf1 = marker(ds, scope, method, "F1", "f1")
        mrec = marker(ds, scope, method, "recall", "recall")
        mmcc = marker(ds, scope, method, "MCC", "mcc")
        mr = marker(ds, scope, method, "rareRec", "rareRec")
        mp = marker(ds, scope, method, "rarePrec", "rarePrec")
        f1 = cell(f1 + mf1, method == best_f1)
        recm = cell(recm + mrec, method == best_rec)
        mcc = cell(mcc + mmcc, method == best_mcc)
        rr = cell(rr + mr, method == best_rr)
        rp = cell(rp + mp, method == best_rp)
        lines.append(f"    {mlabel} & {f1} & {recm} & {mcc} & {rr} & {rp} \\\\")
    return lines


results = load_results()
sig = load_sig()

out_parts = []
for ds, title, label in DATASETS:
    single_lines = rows_for(ds, "single")
    ens_lines = rows_for(ds, "ens")
    tex = (
        f"\n% ===== {title} =====\n"
        "\\begin{table*}[t]\n  \\centering\n  \\begin{tabular}{lccccc}\n    \\toprule\n"
        "    Method & Macro-F1 & Macro Recall & MCC & Rare Recall & Rare Precision \\\\\n    \\midrule\n"
        "    \\multicolumn{6}{l}{\\textit{Single model}} \\\\\n    \\midrule\n"
        + "\n".join(single_lines) +
        "\n    \\midrule\n    \\multicolumn{6}{l}{\\textit{3-seed ensemble}} \\\\\n    \\midrule\n"
        + "\n".join(ens_lines) +
        "\n    \\bottomrule\n  \\end{tabular}\n"
        f"  \\caption{{{title}. Mean$\\pm$sd over 3 seeds $\\times$ 2 folds (single) / 2 folds (ensemble). "
        "\\textbf{Bold} = best per column (within this block). "
        "$\\uparrow$/$\\downarrow$: bootstrap 95\\% CI excludes zero (per-video cluster bootstrap) AND "
        "practically meaningful ($|\\Delta|\\geq$0.003-0.005) vs. temporal baseline. Unmarked = "
        "CI includes zero or delta below the practical floor.}\n"
        f"  \\label{{tab:{label}}}\n\\end{{table*}}\n"
    )
    out_parts.append(tex)

with open("tables/report/main_results.tex", "w") as f:
    f.write("".join(out_parts))
print("".join(out_parts))
