"""
Pareto-style figure: MCC (x) vs Rare Recall (y) for every decoding method, one panel per
dataset x scope (Kvasir/GALAR x single/ensemble). Poster/publication-quality vector PDF for
the WACV paper.

Design:
  - Two-tier visual hierarchy: baselines (argmax/LA/OT/temporal) are muted, small, semi-
    transparent; EGA/HCE ("ours") are large, vivid, fully-opaque, black-outlined. Ours
    never gets visually buried even when it isn't the single best on a raw axis.
  - Distinct marker SHAPE per method as well as color (never color-only).
  - No error bars -- exact sd is in the tables; this figure's job is the visual comparison,
    and bars across 7 close-together points read as clutter, not precision.
  - Automatic dodge-with-leader-line: when 2+ markers would visually overlap (checked in
    actual display pixels after layout), they're pushed apart along a small ring around their
    true centroid and connected back to their real data position with a thin leader line, so
    crowded clusters (e.g. GALAR-ensemble's ours cluster) separate instead of merging.
  - Generous margins, bold typography, minimal ink.

Source: tables/report/results_hcg.csv (downloaded from the vce-hcg k8s job's PVC output).

Run: python analysis/plot_pareto.py
Writes: figures/pareto_mcc_rarerecall.pdf
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS = "tables/report/results_hcg.csv"
OUT = "figures/pareto_mcc_rarerecall.pdf"

# method -> (color, marker, markersize, is_ours)
STYLE = {
    "argmax":        ("#8A8A8A", "o", 8.5, False),
    "la_tau":        ("#7CA6C7", "s", 8.5, False),
    "ot_alpha":      ("#C97B4A", "^", 9.0, False),
    "mean_temporal": ("#5B7C99", "D", 8.2, False),
    "ega_b4":        ("#00A878", "H", 15.0, True),
    "hce":           ("#FFB000", "*", 20.0, True),
}
LABEL = {
    "argmax": "Argmax", "la_tau": "LA", "ot_alpha": "OT", "mean_temporal": "Temporal",
    "ega_b4": "EGA (ours)", "hce": "HCE (ours)",
}
PANELS = [
    ("kvasir", "single", "Kvasir-Capsule, single model"),
    ("kvasir", "ens", "Kvasir-Capsule, 3-seed ensemble"),
    ("galar_pathology", "single", "GALAR, single model"),
    ("galar_pathology", "ens", "GALAR, 3-seed ensemble"),
]
LABEL_CANDIDATES = [
    (11, 8, "left"), (11, -15, "left"), (-11, -15, "right"), (-11, 8, "right"),
    (11, 20, "left"), (-11, 20, "right"), (11, -26, "left"), (-11, -26, "right"),
    (17, -4, "left"), (-17, -4, "right"),
]

plt.rcParams.update({
    "figure.dpi": 200, "font.size": 12.5, "axes.edgecolor": "#555", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": "#EEEEEE", "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "legend.fontsize": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "font.family": "DejaVu Sans",
})


def load():
    rows = list(csv.DictReader(open(RESULTS)))
    return {(r["dataset"], r["scope"], r["method"]): r for r in rows}


def get(r, key):
    mean_key = f"{key}_mean" if f"{key}_mean" in r else key
    return float(r[mean_key])


def pareto_frontier(points):
    frontier = []
    best_y = -1e18
    for mcc, rr, m in sorted(points, key=lambda p: -p[0]):
        if rr > best_y:
            frontier.append((mcc, rr, m))
            best_y = rr
    return sorted(frontier, key=lambda p: p[0])


def dodge_overlaps(ax, fig, pts):
    """pts: list of (mcc, rr, method). Returns {method: (draw_x, draw_y, is_dodged)} -- draw
    position may differ from true data position when markers would visually collide."""
    fig.canvas.draw()
    disp = {m: ax.transData.transform((mcc, rr)) for mcc, rr, m in pts}
    # STYLE[m][2] is a marker "size" in points (matplotlib scatter s=size**2, points^2 area);
    # transData.transform gives DISPLAY PIXELS at fig.dpi, so convert points -> pixels before
    # comparing distances, or the threshold is silently ~2.8x too small at 200 DPI and almost
    # never fires.
    pt_to_px = fig.dpi / 72.0
    radius_px = {m: STYLE[m][2] * 0.62 * pt_to_px for _, _, m in pts}

    groups, used = [], set()
    methods = [m for _, _, m in pts]
    for i, m1 in enumerate(methods):
        if m1 in used:
            continue
        grp = [m1]
        for m2 in methods[i + 1:]:
            if m2 in used:
                continue
            d = np.hypot(*(disp[m1] - disp[m2]))
            if d < (radius_px[m1] + radius_px[m2]) * 1.05:
                grp.append(m2)
        if len(grp) > 1:
            groups.append(grp)
            used.update(grp)

    draw_pos = {m: (mcc, rr, False) for mcc, rr, m in pts}
    data_of = {m: (mcc, rr) for mcc, rr, m in pts}
    for grp in groups:
        cx = np.mean([disp[m][0] for m in grp]); cy = np.mean([disp[m][1] for m in grp])
        n = len(grp)
        ring_r = sum(radius_px[m] for m in grp) / len(grp) * 2.6
        # Place each point along its OWN true direction from the centroid, pushed out to
        # ring_r, instead of an arbitrary evenly-spaced ring (angle = pi/2 + 2*pi*k/n). The old
        # scheme forced every 2-point collision onto the vertical axis (angles pi/2 and 3pi/2,
        # both with cos()=0), which put both markers at IDENTICAL x regardless of the real x
        # (e.g. MCC) difference, and ordered top-vs-bottom by iteration/sort order rather than
        # true y value -- silently misrepresenting both axes (seen concretely on GALAR ensemble:
        # EGA's higher MCC was erased to a tie with HCE, and HCE's higher rare recall got
        # rendered as EGA's before this fix). True-direction placement preserves both.
        # Points with (near-)identical true positions have no meaningful direction to preserve
        # (an exact tie), so those are fanned out symmetrically in a small arc instead.
        true_ang = {}
        for m in grp:
            dxp, dyp = disp[m][0] - cx, disp[m][1] - cy
            true_ang[m] = float(np.arctan2(dyp, dxp)) if np.hypot(dxp, dyp) > 1e-6 else None
        buckets = {}
        for m in grp:
            key = round(true_ang[m], 3) if true_ang[m] is not None else "tie"
            buckets.setdefault(key, []).append(m)
        for key, members in buckets.items():
            base_ang = true_ang[members[0]] if key != "tie" else np.pi / 2
            fan = np.pi / 6
            for j, m in enumerate(members):
                offset = 0.0 if len(members) == 1 else (j - (len(members) - 1) / 2) * (fan / max(len(members) - 1, 1))
                ang = base_ang + offset
                nx, ny = cx + ring_r * np.cos(ang), cy + ring_r * np.sin(ang)
                dx, dy = ax.transData.inverted().transform((nx, ny))
                draw_pos[m] = (dx, dy, True)
    return draw_pos, data_of


def place_label(ax, fig, x, y, text, placed_boxes, bold=False, fontsize=10.4, color="#2a2a2a"):
    renderer = fig.canvas.get_renderer()
    for dx, dy, ha in LABEL_CANDIDATES:
        t = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                         fontsize=fontsize, color=color, ha=ha, va="center", zorder=8,
                         fontweight="bold" if bold else "normal")
        fig.canvas.draw()
        bbox = t.get_window_extent(renderer=renderer).expanded(1.08, 1.22)
        if not any(bbox.overlaps(b) for b in placed_boxes):
            placed_boxes.append(bbox)
            return
        t.remove()
    dx, dy, ha = LABEL_CANDIDATES[-1]
    t = ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                     fontsize=fontsize, color=color, ha=ha, va="center", zorder=8,
                     fontweight="bold" if bold else "normal")
    fig.canvas.draw()
    placed_boxes.append(t.get_window_extent(renderer=renderer))


def main():
    results = load()
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.6))
    fig.patch.set_facecolor("white")

    for ax, (ds, scope, title) in zip(axes.flat, PANELS):
        ax.set_facecolor("#FCFCFC")
        pts = []
        for method in STYLE:
            r = results.get((ds, scope, method))
            if r is None:
                continue
            pts.append((get(r, "mcc"), get(r, "rareRec"), method))

        mccs = [p[0] for p in pts]; rrs = [p[1] for p in pts]
        xr = max(mccs) - min(mccs) or 0.05; yr = max(rrs) - min(rrs) or 0.02
        ax.set_xlim(min(mccs) - 0.32 * xr, max(mccs) + 0.32 * xr)
        ax.set_ylim(min(rrs) - 0.34 * yr, max(rrs) + 0.38 * yr)

        frontier = pareto_frontier(pts)
        if len(frontier) > 1:
            fx = [p[0] for p in frontier]; fy = [p[1] for p in frontier]
            ax.plot(fx, fy, "--", color="#C9A0DC", linewidth=2.0, zorder=1,
                    solid_capstyle="round")

        draw_pos, data_of = dodge_overlaps(ax, fig, pts)

        # leader lines for dodged points first (under everything)
        for method, (dx, dy, dodged) in draw_pos.items():
            if dodged:
                mx, my = data_of[method]
                ax.plot([mx, dx], [my, dy], "-", color="#BBBBBB", linewidth=0.9, zorder=2)
                ax.scatter([mx], [my], s=14, c="#BBBBBB", zorder=2)

        # baselines, then ours on top
        for method in sorted(draw_pos, key=lambda m: STYLE[m][3]):
            dx, dy, dodged = draw_pos[method]
            color, marker, size, ours = STYLE[method]
            ax.scatter([dx], [dy], s=size ** 2, c=color, marker=marker,
                       edgecolors=("black" if ours else "white"),
                       linewidths=(1.2 if ours else 0.9),
                       alpha=(1.0 if ours else 0.82), zorder=(5 if ours else 4))

        fig.canvas.draw()
        placed_boxes = []
        for method in sorted(draw_pos, key=lambda m: not STYLE[m][3]):
            dx, dy, _ = draw_pos[method]
            ours = STYLE[method][3]
            place_label(ax, fig, dx, dy, LABEL[method], placed_boxes,
                        bold=ours, fontsize=11.2 if ours else 10.0,
                        color="#111111" if ours else "#666666")

        ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("MCC (higher = better overall balance)")
        ax.set_ylabel("Rare Recall (higher = better)")

    handles = [plt.Line2D([0], [0], marker=STYLE[m][1], color="w", markerfacecolor=STYLE[m][0],
                           markeredgecolor=("black" if STYLE[m][3] else "white"),
                           markeredgewidth=1.2 if STYLE[m][3] else 0.8,
                           markersize=STYLE[m][2] * (0.58 if STYLE[m][1] == "*" else 0.70),
                           label=LABEL[m])
               for m in STYLE]
    handles.append(plt.Line2D([0], [0], linestyle="--", color="#C9A0DC", linewidth=2.2,
                               label="Pareto frontier"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.05), handletextpad=0.7, columnspacing=2.2)

    fig.suptitle("MCC vs. Rare Recall by decoding method", fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0.09, 1, 0.98], h_pad=4.2, w_pad=3.4)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
