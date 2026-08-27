"""
Expected-Gain Arbitration (EGA) -- training-free decoding by expected macro-F1 improvement.

Replaces the frequency heuristic in the rare-class cascade (route rare classes to the plug-in)
with the quantity that heuristic was a proxy for. Macro-F1 decomposes over classes,
    F1_c = 2 TP_c / (P_c + T_c),   P_c = predicted count, T_c = true count,
so reassigning one frame has a closed-form expected effect:
    d F1_c / d TP_c = 2 T_c / (P_c + T_c)^2          (frame is truly c)
    d F1_c / d FP_c = -2 TP_c / (P_c + T_c)^2        (frame is not c)
With a calibrated posterior p_c(x) the expected gain of moving frame x from its current class b
to class c is
    E[Delta](x, b->c) = [ p_c g+_c + (1-p_c) g-_c ] - [ p_b g+_b + (1-p_b) g-_b ]
and we move only when E[Delta] > 0, taking the best c. Counts (TP_c, P_c, T_c) are estimated on
VAL under the base decoder; posteriors are temperature-calibrated on val.

Consequences, both of which fix observed failures of the cascade:
  * no rare set and no fitted fraction -> the high variance from the flip-flopping rare-set size
    disappears; rare classes are claimed automatically because their T_c/(P_c+T_c)^2 is large;
  * a claim is rejected when its expected gain is negative -> on many-class still datasets the
    detector stops over-claiming, which is exactly what collapsed precision there.

Iterated twice so the counts reflect the reassignments (greedy coordinate ascent on expected
macro-F1). Training-free, frozen logits, val-only calibration. Same frame-level metrics.
  python analysis/expected_gain_arbitration.py --dataset ... --device cuda [--ensemble]
"""
import argparse
import csv
import os
import sys

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_principled as C
import decode_sweep as D

W_TEMP = 7
VIDEO_DS = {"kvasir", "galar_pathology"}
TEMPS = (0.5, 1.0, 2.0, 4.0)      # posterior temperature, calibrated on val
ROUNDS = 5
BETAS = (1.0, 1.5, 2.0, 3.0, 4.0)   # macro-F-beta: beta>1 weights recall (rare-disease screening)
MCC_TOL = 0.002
SHAPES = ("global", "log", "rank", "tail")   # rarity-scaling shapes compared                # ega_auto: allow at most this much val-MCC loss vs the base decoder


def prefix_ddir(ds):
    return {
        "galar_pathology": ("densenet201_swint_galar_pathology_focal", "/pvc/results/logits"),
        "kvasir":          ("densenet201_swint_otdecode_official", "/pvc/results/logits_kvtemporal"),
        "hyperkvasir":     ("densenet201_swint_hyperkvasir_focal", "/pvc/results/logits"),
        "gastrovision":    ("densenet201_swint_gastrovision_focal", "/pvc/results/logits"),
    }[ds]


def load(ds, prefix, ddir, fold, seed, ensemble):
    vid = ds in VIDEO_DS
    if not ensemble:
        z = np.load(os.path.join(ddir, f"{prefix}_f{fold}_seed{seed}.npz"), allow_pickle=True)
        base = (z["vL"], z["vY"].astype(int), z["tL"], z["tY"].astype(int))
    else:
        zs = [np.load(os.path.join(ddir, f"{prefix}_f{fold}_seed{s}.npz"), allow_pickle=True) for s in C.SEEDS]
        z = zs[0]
        base = (C._log_mean_softmax([x["vL"] for x in zs]), z["vY"].astype(int),
                C._log_mean_softmax([x["tL"] for x in zs]), z["tY"].astype(int))
    if vid:
        return base + (z["vgroup"], z["tgroup"], z["vframe"].astype(np.int64), z["tframe"].astype(np.int64))
    return base + (None, None, None, None)


def _exact_gain(Pmat, pred, T, K, b2):
    """b2 may be a scalar or a per-class vector (rarity-scaled beta)."""
    """EXACT expected-F-beta deltas (not the first-order derivative).
    Moving a frame INTO c:   dGain_c(p) = (1+b2)(TP_c+p)/(b2 T_c + P_c + 1) - (1+b2)TP_c/(b2 T_c + P_c)
    Moving a frame OUT of b: dLoss_b(p) = (1+b2)(TP_b-p)/(b2 T_b + P_b - 1) - (1+b2)TP_b/(b2 T_b + P_b)
    Returns (gain_in [N,K], loss_out [N,K]) both already divided by K (macro average)."""
    b2 = np.broadcast_to(np.asarray(b2, dtype=float), (K,))
    Pc = np.bincount(pred, minlength=K).astype(float)
    TPc = np.array([Pmat[pred == c, c].sum() if (pred == c).any() else 0.0 for c in range(K)])
    den0 = b2 * T + Pc
    f_old = (1.0 + b2) * TPc / np.maximum(den0, 1e-12)
    # into c: denominator +1
    f_in = (1.0 + b2) * (TPc[None, :] + Pmat) / np.maximum(den0[None, :] + 1.0, 1e-12)
    gain_in = (f_in - f_old[None, :]) / K
    # out of b: denominator -1
    den_out = np.maximum(den0[None, :] - 1.0, 1e-12)
    f_out = (1.0 + b2) * (TPc[None, :] - Pmat) / den_out
    loss_out = (f_out - f_old[None, :]) / K
    return gain_in, loss_out


def beta_vector(prior, beta_max, mode="tail"):
    """Per-class beta from the VAL prior. Shapes differ in how sharply the recall weighting is
    concentrated on the rare tail -- the log shape saturates on long-tailed (K>20) datasets, where
    almost every class sits near the bottom of the log-prior range, so it degenerates to a global
    beta and costs MCC.
      global : beta_c = beta_max for all classes
      log    : interpolate in log-prior (v4; saturates when K is large)
      rank   : linear in rank, only the bottom HALF is boosted
      tail   : beta_max on the bottom THIRD only, 1 elsewhere (sharpest)
    """
    K = len(prior)
    if beta_max <= 1.0:
        return np.ones(K)
    if mode == "global":
        return np.full(K, beta_max, dtype=float)
    p = np.clip(prior, 1e-12, None)
    if mode == "log":
        s = (np.log(p.max()) - np.log(p)) / max((np.log(p.max()) - np.log(p[p > 0].min())), 1e-12)
    elif mode == "rank":
        rank = np.empty(K); rank[np.argsort(-p)] = np.arange(K)      # 0 = most common
        s = np.clip((rank - (K - 1) / 2.0) / max((K - 1) / 2.0, 1e-12), 0.0, 1.0)
    else:                                                             # "tail"
        rank = np.empty(K); rank[np.argsort(-p)] = np.arange(K)
        s = (rank >= K - max(1, K // 3)).astype(float)
    return 1.0 + (beta_max - 1.0) * np.clip(s, 0.0, 1.0)


def ega_fit(vP, v_base_pred, vY, K, beta=1.0, rounds=ROUNDS):
    """INDUCTIVE fit on VAL ONLY, using EXACT expected-F-beta deltas.
    Returns the per-class coefficients needed to score a single frame at test time."""
    pred = v_base_pred.copy()
    N = len(pred)
    T = np.bincount(vY, minlength=K).astype(float)
    b2 = np.asarray(beta, dtype=float) ** 2
    for _ in range(rounds):
        gain_in, loss_out = _exact_gain(vP, pred, T, K, b2)
        cur_loss = loss_out[np.arange(N), pred]                 # cost of vacating current class
        total = gain_in + cur_loss[:, None]
        total[np.arange(N), pred] = -np.inf
        best = total.argmax(1); g = total[np.arange(N), best]
        move = g > 0
        if not move.any():
            break
        pred = pred.copy(); pred[move] = best[move]
    # export the val-fitted class statistics; test frames are scored independently against them
    Pc = np.bincount(pred, minlength=K).astype(float)
    TPc = np.array([vP[pred == c, c].sum() if (pred == c).any() else 0.0 for c in range(K)])
    return (T / N, Pc / N, TPc / N, b2)


def ega_apply(P, base_pred, stats):
    """INDUCTIVE apply: score each frame independently against VAL-fitted class statistics
    (rates, rescaled to this cohort's size). No test aggregates are used."""
    T_r, Pc_r, TPc_r, b2 = stats
    N, K = P.shape
    T, Pc, TPc = T_r * N, Pc_r * N, TPc_r * N
    den0 = b2 * T + Pc
    f_old = (1.0 + b2) * TPc / np.maximum(den0, 1e-12)
    gain_in = ((1.0 + b2) * (TPc[None, :] + P) / np.maximum(den0[None, :] + 1.0, 1e-12) - f_old[None, :]) / K
    loss_out = ((1.0 + b2) * (TPc[None, :] - P) / np.maximum(den0[None, :] - 1.0, 1e-12) - f_old[None, :]) / K
    cur_loss = loss_out[np.arange(N), base_pred]
    total = gain_in + cur_loss[:, None]
    total[np.arange(N), base_pred] = -np.inf
    best = total.argmax(1)
    return np.where(total[np.arange(N), best] > 0, best, base_pred)


def run(ds, ensemble):
    prefix, ddir = prefix_ddir(ds)
    z0 = np.load(os.path.join(ddir, f"{prefix}_f0_seed0.npz"), allow_pickle=True)
    K = int(max(z0["vY"].max(), z0["tY"].max())) + 1
    D.CLASSES = [str(i) for i in range(K)]
    vid = ds in VIDEO_DS
    methods = ["argmax", "ot_alpha"] + (["mean_temporal"] if vid else []) + [f"{s}_b{b:g}" for s in SHAPES for b in BETAS]
    YT = []; acc = {m: [] for m in methods}; chosen = []
    for fold, seed in C._runs(ensemble):
        vL, vY, tL, tY, vg, tg, vf, tf = load(ds, prefix, ddir, fold, seed, ensemble)
        prior = np.bincount(vY, minlength=K).astype(float); prior /= prior.sum()
        lp = np.log(prior + 1e-12)
        vlog, tlog = D.log_softmax(vL), D.log_softmax(tL)
        YT.append(tY)
        acc["argmax"].append(tL.argmax(1))
        acc["ot_alpha"].append(D.fit_ot_alpha(vL, vlog, vY, prior, lp, D.f1m)(tL, tlog))
        if vid:
            vbaseL = D.temporal_smooth(vL, vg, vf, W_TEMP); tbaseL = D.temporal_smooth(tL, tg, tf, W_TEMP)
            acc["mean_temporal"].append(tbaseL.argmax(1))
        else:
            vbaseL, tbaseL = vlog, tlog
        # optional LA pre-adjustment (INDUCTIVE: a fixed per-class bias fitted on val), chosen on val
        best_tau, best_tf1 = 0.0, D.f1m(vY, vbaseL.argmax(1))
        for tau in np.linspace(0.0, 2.0, 11):
            f = D.f1m(vY, (vbaseL - tau * lp[None, :]).argmax(1))
            if f > best_tf1:
                best_tf1, best_tau = f, tau
        if best_tau > 0:
            vbaseL = vbaseL - best_tau * lp[None, :]
            tbaseL = tbaseL - best_tau * lp[None, :]
        vbase, tbase = vbaseL.argmax(1), tbaseL.argmax(1)
        T_rate = np.bincount(vY, minlength=K).astype(float) / len(vY)     # val true-count rate
        for shape in SHAPES:
            for b in BETAS:
                bv = beta_vector(prior, b, shape)
                best = (-9.0, 1.0)
                for T in TEMPS:
                    vP = D._softmax(vbaseL / T)
                    st = ega_fit(vP, vbase, vY, K, beta=bv)
                    f = D.f1m(vY, ega_apply(vP, vbase, st))
                    if f > best[0]:
                        best = (f, T)
                Temp = best[1]
                st = ega_fit(D._softmax(vbaseL / Temp), vbase, vY, K, beta=bv)
                acc[f"{shape}_b{b:g}"].append(ega_apply(D._softmax(tbaseL / Temp), tbase, st))
        print(f"  fold{fold} seed{seed}: done (K={K})", flush=True)

    lab = list(range(K))
    support = np.bincount(np.concatenate(YT), minlength=K).astype(float)
    prev = support / max(support.sum(), 1)
    rare = sorted(np.argsort(prev)[:max(1, K // 3)].tolist())
    macro = {}; rareR = {}
    for m in methods:
        f1s, recs, mccs, rr = [], [], [], []
        for r, pr in enumerate(acc[m]):
            f1s.append(D.f1m(YT[r], pr)); recs.append(D.recm(YT[r], pr)); mccs.append(D.mccm(YT[r], pr))
            _, R_, _, _ = precision_recall_fscore_support(YT[r], pr, labels=lab, zero_division=0)
            rr.append(R_[rare].mean())
        macro[m] = (np.mean(f1s), np.std(f1s), np.mean(recs), np.mean(mccs))
        rareR[m] = (np.mean(rr), np.std(rr))

    tag = "ens" if ensemble else "single"
    os.makedirs("/pvc/results/experimental", exist_ok=True)
    outp = "/pvc/results/experimental/results_ega_v5.csv"
    hdr = not os.path.exists(outp)
    with open(outp, "a", newline="") as f:
        w = csv.writer(f)
        if hdr:
            w.writerow(["dataset", "scope", "method", "f1_mean", "f1_sd", "recall", "mcc", "rareRec_mean", "rareRec_sd"])
        for m in methods:
            a, b, c, d = macro[m]; rrm, rrsd = rareR[m]
            w.writerow([ds, tag, m, f"{a:.4f}", f"{b:.4f}", f"{c:.4f}", f"{d:.4f}", f"{rrm:.4f}", f"{rrsd:.4f}"])
    amf = macro["argmax"][0]; basef = macro["mean_temporal"][0] if vid else amf
    amrr = rareR["argmax"][0]
    print(f"=== {ds} [{tag}] K={K} runs={len(YT)} ===", flush=True)
    for m in methods:
        a, b, c, d = macro[m]; rrm, rrsd = rareR[m]
        print(f"  {m:<14} F1 {a:.4f}+-{b:.4f} (dArg {a-amf:+.4f}, dBase {a-basef:+.4f}) rec {c:.4f} mcc {d:.4f} | "
              f"rareRec {rrm:.4f} (dArg {rrm-amrr:+.4f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["galar_pathology", "kvasir", "hyperkvasir", "gastrovision"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--ensemble", action="store_true")
    a = ap.parse_args()
    if a.device == "cuda":
        assert torch is not None and torch.cuda.is_available()
        D._DEVICE = torch.device("cuda")
    run(a.dataset, a.ensemble)


if __name__ == "__main__":
    main()
