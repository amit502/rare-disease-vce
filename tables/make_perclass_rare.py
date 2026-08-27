"""
Per-rare-class recall tables (Kvasir + GALAR, single/ensemble), sourced from the
per_class_*_hcg.csv files downloaded from the vce-hcg job's PVC output.

Kvasir class names are hardcoded (analysis/per_class_report.py's convention, matching
datasets/kvasir.py's CLASSES order). GALAR class names come from
tables/report/galar_classes.txt (downloaded from /pvc/results/galar_pathology_splits/classes.txt).

Run: python tables/make_perclass_rare.py
Writes: tables/report/perclass_rare.tex
"""
import csv

KVASIR_NAMES = ["Angiectasia", "Blood - fresh", "Erosion", "Erythema", "Foreign Body",
                "Ileocecal valve", "Lymphangiectasia", "Normal clean mucosa", "Pylorus",
                "Reduced Mucosal View", "Ulcer"]
GALAR_NAMES = [ln.strip() for ln in open("tables/report/galar_classes.txt") if ln.strip()]

METHODS = [("argmax", "Argmax"), ("ot_alpha", "OT"), ("mean_temporal", "Temporal"),
           ("ega_b4", "EGA"), ("hce", "HCE")]

BLOCKS = [
    ("kvasir", "single", KVASIR_NAMES, "Kvasir-Capsule, single model", "perclass_kvasir_single"),
    ("kvasir", "ens", KVASIR_NAMES, "Kvasir-Capsule, ensemble", "perclass_kvasir_ens"),
    ("galar_pathology", "single", GALAR_NAMES, "GALAR, single model", "perclass_galar_pathology_single"),
    ("galar_pathology", "ens", GALAR_NAMES, "GALAR, ensemble", "perclass_galar_pathology_ens"),
]


def load(ds, scope):
    rows = list(csv.DictReader(open(f"tables/report/per_class_{ds}_{scope}_hcg.csv")))
    return {(r["method"], int(r["class"])): r for r in rows}


out_parts = []
for ds, scope, names, title, label in BLOCKS:
    data = load(ds, scope)
    rare_idx = sorted({c for (m, c) in data if data[(m, c)]["rare"] == "1"})
    rare_names = [names[i] for i in rare_idx]
    support = [data[(METHODS[0][0], i)]["support"] for i in rare_idx]

    ncols = "c" * len(rare_idx)
    header = " & ".join(rare_names)
    lines = []
    for method, mlabel in METHODS:
        vals = " & ".join(f"{float(data[(method, i)]['R_mean']):.3f}" for i in rare_idx)
        lines.append(f"    {mlabel} & {vals} \\\\")
    support_row = " & ".join(support)

    tex = (
        f"\n% ===== {title} per-RARE-class recall =====\n"
        f"\\begin{{table}}[t]\n  \\centering\n  \\begin{{tabular}}{{l{ncols}}}\n    \\toprule\n"
        f"    Method & {header} \\\\\n    \\midrule\n"
        + "\n".join(lines) +
        f"\n    \\midrule\n    Support (test) & {support_row} \\\\\n"
        "    \\bottomrule\n  \\end{tabular}\n"
        f"  \\caption{{{title}: recall on the {len(rare_idx)} rarest classes (bottom third by "
        f"prevalence). Full per-class P/R/F1 for all classes and methods available in "
        f"per\\_class\\_{ds}\\_{scope}\\_hcg.csv.}}\n"
        f"  \\label{{tab:{label}}}\n\\end{{table}}\n"
    )
    out_parts.append(tex)

with open("tables/report/perclass_rare.tex", "w") as f:
    f.write("".join(out_parts))
print("".join(out_parts))
