"""
Head-to-head bootstrap comparison: cbsampler vs focal on Kvasir, for hce and pact, using the
companion cmp_preds_{ds}_{scope}.npz files hcg_decode.py's run() now dumps (per-fold predictions
+ labels + video groups). results_hcg.csv only has each config's aggregated mean/sd vs its OWN
temporal baseline -- this answers the actual question asked ("is cbsampler significantly better
than focal"), not "does the decoder help within each config."

Reuses the same per-video cluster bootstrap + Fisher-combine machinery as hcg_decode.py's
_boot_pvalue/_fisher_combine (copied here as standalone functions since they're nested closures
inside hcg_decode.run() and not importable as-is).

Run: python analysis/compare_cbsampler_vs_focal.py
Reads: tables/report/cmp_preds_kvasir_{single,ens}.npz, cmp_preds_kvasir_cbsampler_{single,ens}.npz
Writes: tables/report/compare_cbsampler_vs_focal.csv
"""
import csv
import numpy as np


def _metrics_from_cm(cm, K, rare):
    cm = cm.astype(np.float64)
    tp = np.diag(cm); P = cm.sum(0); T = cm.sum(1); N = cm.sum()
    f1c = np.where(P + T > 0, 2.0 * tp / np.maximum(P + T, 1e-12), 0.0)
    prec = np.where(P > 0, tp / np.maximum(P, 1e-12), 0.0)
    rec = np.where(T > 0, tp / np.maximum(T, 1e-12), 0.0)
    c = float(tp.sum())
    num = N * c - float(np.dot(P, T))
    den = np.sqrt(max(N * N - float(np.dot(P, P)), 1e-12) * max(N * N - float(np.dot(T, T)), 1e-12))
    mcc = num / max(den, 1e-12)
    return float(f1c.mean()), mcc, float(rec[rare].mean()), float(prec[rare].mean()), float(f1c[rare].mean())


def _boot_pvalue(y, pa, pb, grp, K, rare, nboot=800, seed=0):
    rng = np.random.default_rng(seed)
    uid = np.unique(grp)
    nV = len(uid)
    by_vid = {v: np.flatnonzero(grp == v) for v in uid}
    diffs = {"F1": [], "MCC": [], "rareRec": [], "rarePrec": [], "rareF1": []}
    for _ in range(nboot):
        samp = uid[rng.integers(0, nV, nV)]
        idx = np.concatenate([by_vid[v] for v in samp])
        ys, xa, xb = y[idx], pa[idx], pb[idx]
        cma = np.bincount(ys * K + xa, minlength=K * K).reshape(K, K)
        cmb = np.bincount(ys * K + xb, minlength=K * K).reshape(K, K)
        f1a, mcca, rra, rpa, rfa = _metrics_from_cm(cma, K, rare)
        f1b, mccb, rrb, rpb, rfb = _metrics_from_cm(cmb, K, rare)
        diffs["F1"].append(f1a - f1b)
        diffs["MCC"].append(mcca - mccb)
        diffs["rareRec"].append(rra - rrb)
        diffs["rarePrec"].append(rpa - rpb)
        diffs["rareF1"].append(rfa - rfb)
    out = {}
    for k, v in diffs.items():
        v = np.array(v)
        p_two = 2 * min((v <= 0).mean(), (v >= 0).mean())
        out[k] = (float(v.mean()), float(min(p_two, 1.0)),
                  float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


def _fisher_combine(pvals):
    pvals = [max(p, 1e-300) for p in pvals if p == p]
    if not pvals:
        return float("nan")
    from scipy import stats as st
    chi2 = -2 * sum(np.log(pvals))
    return float(st.chi2.sf(chi2, 2 * len(pvals)))


def load(ds, scope):
    z = np.load(f"tables/report/cmp_preds_{ds}_{scope}.npz", allow_pickle=True)
    return z["YT"], z["TG"], z["hce"], z["pact"], int(z["K"]), z["rare"].tolist()


def main():
    rows = []
    for scope in ("single", "ens"):
        YT_c, TG_c, hce_c, pact_c, K, rare = load("kvasir_cbsampler", scope)
        YT_f, TG_f, hce_f, pact_f, K2, rare2 = load("kvasir", scope)
        assert K == K2 and rare == rare2, "class/rare mismatch between configs"
        n = len(YT_c)
        assert n == len(YT_f), "fold count mismatch"
        for i in range(n):
            assert np.array_equal(YT_c[i], YT_f[i]), f"label mismatch fold {i} scope {scope}"

        for method, predc, predf in (("hce", hce_c, hce_f), ("pact", pact_c, pact_f)):
            per_fold = [_boot_pvalue(YT_c[i], np.asarray(predc[i]), np.asarray(predf[i]),
                                      TG_c[i], K, rare, seed=i) for i in range(n)]
            for metric in ("F1", "MCC", "rareRec", "rarePrec", "rareF1"):
                deltas = [r[metric][0] for r in per_fold]
                pvals = [r[metric][1] for r in per_fold]
                cilo = [r[metric][2] for r in per_fold]
                cihi = [r[metric][3] for r in per_fold]
                combined_p = _fisher_combine(pvals)
                rows.append(["kvasir", scope, method, "cbsampler_vs_focal", metric,
                             f"{np.mean(deltas):.4f}", f"{combined_p:.4g}",
                             f"{np.mean(cilo):.4f}", f"{np.mean(cihi):.4f}"])
                print(f"{scope:7} {method:5} {metric:9} delta={np.mean(deltas):+.4f} "
                      f"p={combined_p:.4g} CI=[{np.mean(cilo):+.4f},{np.mean(cihi):+.4f}]")

    with open("tables/report/compare_cbsampler_vs_focal.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "scope", "method", "test", "metric", "delta", "pvalue", "ci_lo", "ci_hi"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
