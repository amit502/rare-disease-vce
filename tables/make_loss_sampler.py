"""
Table 2: Kvasir-Capsule loss/sampling comparison, DenseNet201+Swin-T (the architecture Table 1
selected), official split. Single model AND 3-seed ensemble, one table, split by a midrule.

Single-model: mean+-sd over 3 seeds (2 folds averaged per seed), same protocol as Table 1;
best per column bolded. Ensemble: only 2 data points exist (one per fold, seed dimension
already collapsed by ensembling) -- mean+-sd over n=2, best per column bolded. No significance
markers anywhere in this table: it compares independently trained models, not decoders on a
shared frozen model, so the per-video cluster bootstrap used elsewhere doesn't apply, and no
seed-level test was run for it (see main results tables for per-video significance).

Candidates restricted to the 4 losses/samplers with BOTH single and ensemble runs on record
(weighted CE, focal, LDAM+DRW, class-balanced sampler). LA-loss and CRT (classifier
re-training) were also tried but have no ensemble run and are omitted here for a clean
single/ensemble comparison.

Run: python tables/make_loss_sampler.py
Writes: tables/report/loss_sampler_kvasir.tex
"""
import csv
import statistics as st

SUMMARY = "tables/summary.csv"
OUT = "tables/report/loss_sampler_kvasir.tex"

# (label, loss, single_base, ens_base)
CANDS = [
    ("Weighted CE", "weighted_ce", "densenet201_swint_kvasir_official", "densenet201_swint_argmaxens_official"),
    ("Focal loss", "focal", "densenet201_swint_focal_official", "densenet201_swint_focal_argmaxens_official"),
    ("LDAM+DRW", "ldam", "densenet201_swint_ldam_official", "densenet201_swint_ldam_argmaxens_official"),
    ("Class-balanced sampler + CE", "ce", "densenet201_swint_cbsampler_official", "densenet201_swint_cbsampler_argmaxens_official"),
]
SEEDS = [0, 1, 42]
METRICS = ["f1", "mac_rec", "mcc", "rr", "rp"]
RARE_REC = ["recall_blood___fresh", "recall_erosion", "recall_erythema"]
RARE_PREC = ["precision_blood___fresh", "precision_erosion", "precision_erythema"]


def fmean(vals):
    return sum(vals) / len(vals)


def ms(vals, bold=False):
    if len(vals) < 2:
        s = f"{vals[0]:.4f}"
    else:
        s = f"{st.mean(vals):.4f}$\\pm${st.stdev(vals):.4f}"
    return f"\\textbf{{{s}}}" if bold else s


def single_vals(rows, base, loss):
    out = {m: [] for m in METRICS}
    for seed in SEEDS:
        def find(fold):
            for name in (f"{base}_f{fold}_seed{seed}", f"{base}_f{fold}"):
                m = [r for r in rows if r["exp_name"] == name and r["seed"] == str(seed) and r["loss"] == loss]
                if m:
                    assert len(m) == 1, (name, seed)
                    return m[0]
            raise AssertionError((base, fold, seed))
        r0, r1 = find(0), find(1)
        out["f1"].append(fmean([float(r0["f1_macro"]), float(r1["f1_macro"])]))
        out["mac_rec"].append(fmean([float(r0["recall_macro"]), float(r1["recall_macro"])]))
        out["mcc"].append(fmean([float(r0["mcc"]), float(r1["mcc"])]))
        rr0 = fmean([float(r0[c]) for c in RARE_REC]); rr1 = fmean([float(r1[c]) for c in RARE_REC])
        rp0 = fmean([float(r0[c]) for c in RARE_PREC]); rp1 = fmean([float(r1[c]) for c in RARE_PREC])
        out["rr"].append(fmean([rr0, rr1])); out["rp"].append(fmean([rp0, rp1]))
    return out


def ens_vals(rows, base):
    out = {m: [] for m in METRICS}
    for fold in (0, 1):
        m = [r for r in rows if r["exp_name"] == f"{base}_f{fold}"]
        assert len(m) == 1, (base, fold)
        r = m[0]
        out["f1"].append(float(r["f1_macro"])); out["mac_rec"].append(float(r["recall_macro"]))
        out["mcc"].append(float(r["mcc"]))
        out["rr"].append(fmean([float(r[c]) for c in RARE_REC]))
        out["rp"].append(fmean([float(r[c]) for c in RARE_PREC]))
    return out


DECODERS = [("argmax", "Argmax"), ("la", "LA"), ("otalpha", "OT-alpha"), ("otreg", "OT-reg")]
DECODE_BASES = {
    "Weighted CE": "densenet201_swint",
    "Focal loss": "densenet201_swint_focal",
    "LDAM+DRW": "densenet201_swint_ldam",
    "Class-balanced sampler + CE": "densenet201_swint_cbsampler",
}


def decoder_cell(rows, base, decoder, loss, bold_f1=False, bold_mcc=False):
    f1s, mccs = [], []
    for fold in (0, 1):
        name = f"{base}_{decoder}_official_f{fold}"
        m = [r for r in rows if r["exp_name"] == name and r["loss"] == loss]
        assert len(m) >= 1, name
        f1s.extend(float(r["f1_macro"]) for r in m)
        mccs.extend(float(r["mcc"]) for r in m)
    f1_s = f"\\textbf{{{fmean(f1s):.4f}}}" if bold_f1 else f"{fmean(f1s):.4f}"
    mcc_s = f"\\textbf{{{fmean(mccs):.4f}}}" if bold_mcc else f"{fmean(mccs):.4f}"
    return f"{f1_s}/{mcc_s}"


def build_rows(cand_vals):
    best = {m: max(cand_vals, key=lambda lbl: st.mean(cand_vals[lbl][m])) for m in METRICS}
    lines = []
    for label in cand_vals:
        cells = [ms(cand_vals[label][m], bold=(label == best[m])) for m in METRICS]
        lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    return lines, best


def main():
    rows = list(csv.DictReader(open(SUMMARY)))
    single_data = {label: single_vals(rows, sbase, loss) for label, loss, sbase, ebase in CANDS}
    ens_data = {label: ens_vals(rows, ebase) for label, loss, sbase, ebase in CANDS}

    single_lines, _ = build_rows(single_data)
    ens_lines, _ = build_rows(ens_data)

    tex = (
        "\n% ===== Table 2: Kvasir-Capsule loss/sampling comparison, DenseNet201+Swin-T =====\n"
        "\\begin{table*}[t]\n  \\centering\n  \\begin{tabular}{lccccc}\n    \\toprule\n"
        "    Loss / Sampler & Macro-F1 & Macro Recall & MCC & Rare Recall & Rare Precision \\\\\n    \\midrule\n"
        "    \\multicolumn{6}{l}{\\textit{Single model}} \\\\\n    \\midrule\n"
        + "\n".join(single_lines) +
        "\n    \\midrule\n    \\multicolumn{6}{l}{\\textit{3-seed ensemble}} \\\\\n    \\midrule\n"
        + "\n".join(ens_lines) +
        "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Kvasir-Capsule loss/sampling comparison, DenseNet201+Swin-T, official split. "
        "Single model: mean$\\pm$sd over 3 seeds (2 folds averaged per seed). Ensemble: mean$\\pm$sd "
        "over 2 folds (n=2, seed variation already collapsed by ensembling). \\textbf{Bold} = best "
        "per column in each block. No significance testing (see main results tables for per-video "
        "cluster bootstrap significance). Rare = Blood - fresh, Erosion, Erythema. LA-loss and CRT "
        "were also tried but have no ensemble run and are omitted. Focal loss is carried forward to "
        "decoding.}\n"
        "  \\label{tab:loss_sampler_kvasir}\n\\end{table*}\n"
    )

    # decoder-verification sub-table: for each loss/sampler config, what does argmax/LA/OT-alpha/
    # OT-reg give (F1/MCC per cell)? Justifies why focal was carried forward: it wins under the
    # OT-alpha decoder specifically, which is what the selection was actually based on, even
    # though it isn't clearly best under plain argmax (see the table above).
    dec_cells_by_col = {dec: [] for dec, _ in DECODERS}
    raw = {}
    for label, loss, sbase, ebase in CANDS:
        base = DECODE_BASES[label]
        for dec, _ in DECODERS:
            f1s, mccs = [], []
            for fold in (0, 1):
                name = f"{base}_{dec}_official_f{fold}"
                m = [r for r in rows if r["exp_name"] == name and r["loss"] == loss]
                f1s.extend(float(r["f1_macro"]) for r in m)
                mccs.extend(float(r["mcc"]) for r in m)
            raw[(label, dec)] = (fmean(f1s), fmean(mccs))

    best_f1_by_dec = {dec: max(CANDS, key=lambda c: raw[(c[0], dec)][0])[0] for dec, _ in DECODERS}
    best_mcc_by_dec = {dec: max(CANDS, key=lambda c: raw[(c[0], dec)][1])[0] for dec, _ in DECODERS}

    dec_lines = []
    for label, loss, sbase, ebase in CANDS:
        cells = []
        for dec, _ in DECODERS:
            f1v, mccv = raw[(label, dec)]
            f1_s = f"\\textbf{{{f1v:.4f}}}" if best_f1_by_dec[dec] == label else f"{f1v:.4f}"
            mcc_s = f"\\textbf{{{mccv:.4f}}}" if best_mcc_by_dec[dec] == label else f"{mccv:.4f}"
            cells.append(f"{f1_s}/{mcc_s}")
        dec_lines.append(f"    {label} & " + " & ".join(cells) + " \\\\")
    dec_header = " & ".join(dlabel for _, dlabel in DECODERS)
    dec_tex = (
        "\n% ===== Table 2b: decoder verification (F1/MCC), single model =====\n"
        "\\begin{table*}[t]\n  \\centering\n  \\begin{tabular}{lcccc}\n    \\toprule\n"
        f"    Loss / Sampler & {dec_header} \\\\\n    \\midrule\n"
        + "\n".join(dec_lines) +
        "\n    \\bottomrule\n  \\end{tabular}\n"
        "  \\caption{Kvasir-Capsule, DenseNet201+Swin-T: Macro-F1/MCC under each decoder, single "
        "model, official split (2 folds averaged, no seed-level sd shown). \\textbf{Bold} = best "
        "config per decoder per metric (F1 or MCC independently). Verifies the focal-loss choice "
        "against the decoder actually used for selection (OT-alpha), not just raw argmax: focal "
        "wins both F1 and MCC under OT-alpha even though it is not the argmax-best config.}\n"
        "  \\label{tab:loss_sampler_decoder_verify}\n\\end{table*}\n"
    )

    with open(OUT, "w") as f:
        f.write(tex + dec_tex)
    print(tex + dec_tex)


if __name__ == "__main__":
    main()
