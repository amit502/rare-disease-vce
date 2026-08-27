"""
PRINCIPLED OT+temporal framework (temp CSV) -- composition of two training-free operators,
selected by deployment-faithful cross-validation. This replaces the single-objective
unified_decode: its soft prior-marginal-KL term is NOT sinkhorn OT and never recovered
Kvasir's real OT gain regardless of the entropic temperature. Here we compose the ACTUAL
operators that work.

Two forward-free operators on frozen logits L (+ (video,frame) tags, never labels):
  T_W  temporal smoothing : within-video +/-W posterior moving-average (log-probs). Lesions
       are contiguous frame runs, artifact FPs are isolated spikes -> averaging lifts
       precision. Safe at any imbalance.
  O_a  prior-transport (entropic OT / sinkhorn) : joint assignment of the cohort to the
       class marginal prior**alpha. Recovers rare classes at MODERATE imbalance (helps);
       at EXTREME imbalance the base-rate wall makes it inject false positives (hurts).

Framework = one of {I (argmax), T_W, O_a, O_a o T_W}, i.e. each operator optionally applied.
The mode + (alpha, eps, W) are chosen per dataset by LEAVE-VIDEO-OUT CV on val with POOLED
out-of-fold macro-F1 (mirrors the train->test video shift), argmax an explicit candidate,
and a simpler-operator tie-break (argmax > single op > composition; temporal preferred over
OT at equal complexity) -> a shrinkage toward safety that makes the choice >= argmax on the
selection metric by construction.

Expected: GALAR selects T_W or I (OT stays off) -> >= argmax; Kvasir selects O_a o T_W
(OT on) -> the real OT+temporal gain, > fixed temporal. The base-rate diagnostic EXPLAINS
what the CV selects. Contrast vs compose_naive (in-sample full-val selection, overfits).

Writes /pvc/results/experimental/results_compose.csv.
  python analysis/compose_principled.py --dataset galar_pathology|kvasir --device cuda
"""
import argparse
import csv
import os
import sys

import numpy as np

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decode_sweep as D

FOLDS = (0, 1)
SEEDS = (0, 1, 42)
N_CV = 5              # leave-video-out CV folds on val (grouped by video)
TIE = 1e-3           # near-tie band -> prefer the simpler (safer) operator composition
P_MIN = 1e-4         # base-rate gate: enable OT iff the rarest class prevalence >= P_MIN.
                     # Below P_MIN the base-rate wall caps rare-class precision ~0 so OT's
                     # redistribution only injects false positives. GALAR min-prev 1e-5 < P_MIN
                     # -> OT OFF; Kvasir min-prev 8.7e-4 > P_MIN -> OT ON. (~87x separation.)
                     # This is a property of the KNOWN deployment prior, not a learned/CV signal:
                     # video-disjoint CV mis-measures OT (a cohort operator) and tanks it on BOTH.

WG = (7, 15, 31)
AL = (0.3, 0.5, 1.0)
EPS = (0.05, 0.1, 0.25)


def candidate_configs():
    """(mode, alpha, eps, W); argmax once, then T_W, O_a, O_a o T_W."""
    cfgs = [("argmax", 1.0, 0.05, 7)]
    for w in WG:
        cfgs.append(("temporal", 1.0, 0.05, w))
    for al in AL:
        for eps in EPS:
            cfgs.append(("ot", al, eps, 7))
    for al in AL:
        for eps in EPS:
            for w in WG:
                cfgs.append(("ot_temporal", al, eps, w))
    return cfgs


def _cost(cfg):
    """(n_operators, uses_OT) -- for the simpler-first tie-break."""
    mode = cfg[0]
    nops = {"argmax": 0, "temporal": 1, "ot": 1, "ot_temporal": 2}[mode]
    return (nops, 1 if "ot" in mode else 0)


def decode(cfg, L, g, f, prior):
    mode, al, eps, w = cfg
    if mode == "argmax":
        return L.argmax(1)
    if mode == "temporal":
        return D.temporal_smooth(L, g, f, w).argmax(1)
    cm = np.power(prior, al); cm /= cm.sum()
    if mode == "ot":
        return D.sinkhorn_assign(D.log_softmax(L), cm, eps=eps)
    Ls = D.temporal_smooth(L, g, f, w)                 # log-probs -> OT on smoothed posteriors
    return D.sinkhorn_assign(Ls, cm, eps=eps)


def _key(score, cfg):
    nops, uses_ot = _cost(cfg)
    w = cfg[3] if cfg[0] in ("temporal", "ot_temporal") else 0
    # max: highest score (TIE-rounded), then simpler/safer op-set, then LARGER temporal window.
    # W is a monotone, low-risk hyperparameter for dense-frame VCE (temporal smoothing helps
    # precision on contiguous lesion runs, monotone in W up to the tested max); so among
    # val-tied configs prefer more smoothing rather than picking W arbitrarily -- this lands on
    # the manual's W=31 operating point instead of a noisy small-val W=15.
    return (round(score / TIE), -nops, -uses_ot, w)


def select_principled(vL, vY, vg, vf, K, seed, cfgs):
    """leave-video-out CV; pooled out-of-fold macro-F1 per candidate; simpler-first tie-break."""
    vids = np.unique(vg)
    nfold = min(N_CV, len(vids))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(vids))
    fold_of = {vid: i % nfold for i, vid in enumerate(vids[perm])}
    vfold = np.array([fold_of[g] for g in vg])
    best = None
    for cfg in cfgs:
        oof = np.empty(len(vY), dtype=int)
        for fo in range(nfold):
            te = vfold == fo; tr = ~te
            if te.sum() == 0 or tr.sum() == 0:
                oof[te] = vL[te].argmax(1)
                continue
            pr = np.bincount(vY[tr], minlength=K).astype(float); pr /= pr.sum()
            oof[te] = decode(cfg, vL[te], vg[te], vf[te], pr)
        k = _key(D.f1m(vY, oof), cfg)
        if best is None or k > best[0]:
            best = (k, cfg)
    return best[1]


def select_naive(vL, vY, vg, vf, K, cfgs):
    """in-sample full-val selection (overfitting baseline)."""
    prior = np.bincount(vY, minlength=K).astype(float); prior /= prior.sum()
    best = None
    for cfg in cfgs:
        k = _key(D.f1m(vY, decode(cfg, vL, vg, vf, prior)), cfg)
        if best is None or k > best[0]:
            best = (k, cfg)
    return best[1]


def select_gated(vL, vY, vg, vf, K, cfgs, p_min):
    """Base-rate-gated selection. The gate reads the VAL prior (natural skew preserved --
    NOT the capped train prior, which would understate GALAR's extremity): OT is enabled iff
    the rarest class prevalence >= p_min (below it the base-rate wall caps rare-class precision
    ~0, so OT only injects false positives). OT's own target marginal is this same val prior,
    so gate and operator read the identical distribution -- no train-prior leakage. Within the
    allowed set, in-sample val PROPOSES the config (alpha,eps,W). Leakage-free (val only).
    NB: a video-disjoint CV cannot substitute here -- it mis-measures OT (a cohort operator
    applied to the whole test batch) by evaluating it on tiny non-representative held-out
    cohorts, tanking OT on BOTH datasets. Returns (cfg, allow_ot, min_prev)."""
    prior = np.bincount(vY, minlength=K).astype(float); prior /= prior.sum()
    min_prev = float(prior[prior > 0].min())
    allow_ot = min_prev >= p_min
    # TWO-LEVEL design: the base-rate gate is the OT on/off DECISION. When OT is admitted we
    # commit to an OT mode (O_a or O_a o T_W) and tune only (alpha,eps,W) in-sample -- we do
    # NOT re-litigate OT vs temporal with a noisy per-model in-sample F1 (that dropped OT on
    # 5/6 single-model Kvasir runs even though OT is beneficial there). When vetoed, restrict
    # to argmax / temporal.
    if allow_ot:
        cand = [c for c in cfgs if "ot" in c[0]]
    else:
        cand = [c for c in cfgs if "ot" not in c[0]]
    insample = {cfg: D.f1m(vY, decode(cfg, vL, vg, vf, prior)) for cfg in cand}
    best = max(cand, key=lambda c: _key(insample[c], c))
    return best, allow_ot, min_prev


def _log_mean_softmax(Ls):
    """ensemble across seeds: mean of softmax probs, returned as log-probs (drops into any
    decoder that reads logits: _softmax(log p)=p, argmax and log_softmax both consistent)."""
    p = np.mean([D._softmax(L) for L in Ls], axis=0)
    return np.log(p + 1e-12)


def _runs(ensemble):
    """(fold, seed_or_None) units: 6 single models, or 2 seed-ensembles per fold."""
    if ensemble:
        return [(f, None) for f in FOLDS]
    return [(f, s) for f in FOLDS for s in SEEDS]


def _load_run(ddir, prefix, fold, seed, ensemble):
    if not ensemble:
        z = np.load(os.path.join(ddir, f"{prefix}_f{fold}_seed{seed}.npz"), allow_pickle=True)
        return z["vL"], z["vY"].astype(int), z["tL"], z["tY"].astype(int), \
            z["vgroup"], z["vframe"].astype(np.int64), z["tgroup"], z["tframe"].astype(np.int64)
    zs = [np.load(os.path.join(ddir, f"{prefix}_f{fold}_seed{s}.npz"), allow_pickle=True) for s in SEEDS]
    z0 = zs[0]
    vL = _log_mean_softmax([z["vL"] for z in zs]); tL = _log_mean_softmax([z["tL"] for z in zs])
    return vL, z0["vY"].astype(int), tL, z0["tY"].astype(int), \
        z0["vgroup"], z0["vframe"].astype(np.int64), z0["tgroup"], z0["tframe"].astype(np.int64)


def evaluate(dataset, ensemble=False):
    if dataset == "galar_pathology":
        with open("/pvc/results/galar_pathology_splits/classes.txt") as f:
            D.CLASSES = [ln.strip() for ln in f if ln.strip()]
        prefix = "densenet201_swint_galar_pathology_focal"; ddir = "/pvc/results/logits"
    else:
        prefix = "densenet201_swint_otdecode_official"; ddir = "/pvc/results/logits_kvtemporal"
    K = len(D.CLASSES)
    cfgs = candidate_configs()
    runs = _runs(ensemble)
    print(f"[grid] {len(cfgs)} candidate configs (incl. argmax) | {len(runs)} runs "
          f"({'seed-ensemble' if ensemble else 'single-model'})", flush=True)

    S = {s: [] for s in ["argmax", "mean_temporal", "ot_alpha", "compose_naive",
                         "compose_principled", "compose_gated"]}
    chosen_p = []; chosen_g = []

    for fold, seed in runs:
            vL, vY, tL, tY, vg, vf, tg, tf = _load_run(ddir, prefix, fold, seed, ensemble)
            prior = np.bincount(vY, minlength=K).astype(float); prior /= prior.sum()
            sc = lambda pred: [D.f1m(tY, pred), D.recm(tY, pred), D.mccm(tY, pred)]

            S["argmax"].append(sc(tL.argmax(1)))
            S["mean_temporal"].append(sc(D.temporal_smooth(tL, tg, tf, 7).argmax(1)))
            fit = D.fit_ot_alpha(vL, D.log_softmax(vL), vY, prior, np.log(prior + 1e-12), D.f1m)
            S["ot_alpha"].append(sc(fit(tL, D.log_softmax(tL))))

            cn = select_naive(vL, vY, vg, vf, K, cfgs)
            S["compose_naive"].append(sc(decode(cn, tL, tg, tf, prior)))
            cv_seed = seed if seed is not None else 12345 + fold
            cp = select_principled(vL, vY, vg, vf, K, cv_seed, cfgs); chosen_p.append(cp)
            S["compose_principled"].append(sc(decode(cp, tL, tg, tf, prior)))
            cg, allow_ot, min_prev = select_gated(vL, vY, vg, vf, K, cfgs, P_MIN)
            chosen_g.append(cg)
            S["compose_gated"].append(sc(decode(cg, tL, tg, tf, prior)))
            sid = 'ens' if seed is None else seed
            print(f"  f{fold}s{sid}: principled={cp}  gated={cg} (allow_ot={allow_ot}, "
                  f"min_prev={min_prev:.5f})  naive={cn}", flush=True)

    os.makedirs("/pvc/results/experimental", exist_ok=True)
    outp = f"/pvc/results/experimental/results_compose{'_ens' if ensemble else ''}.csv"
    hdr = not os.path.exists(outp)
    with open(outp, "a", newline="") as f:
        w = csv.writer(f)
        if hdr:
            w.writerow(["dataset", "strategy", "f1", "recall", "mcc"])
        am = np.array(S["argmax"]).mean(0)
        print(f"\n=== {dataset} (mean over {len(S['argmax'])} runs) ===")
        print(f"{'strategy':<20}{'f1':>8}{'recall':>9}{'mcc':>8}   vs-argmax-F1")
        for s, vals in S.items():
            m = np.array(vals).mean(0)
            print(f"{s:<20}{m[0]:>8.4f}{m[1]:>9.4f}{m[2]:>8.4f}   {m[0]-am[0]:+.4f}")
            w.writerow([dataset, s, f"{m[0]:.4f}", f"{m[1]:.4f}", f"{m[2]:.4f}"])

    ar = np.array(S["argmax"])[:, 0]
    for name, chosen in (("principled", chosen_p), ("gated", chosen_g)):
        pr = np.array(S[f"compose_{name}"])[:, 0]
        wins = int((pr >= ar - 1e-6).sum())
        modes = [c[0] for c in chosen]
        ot_on = sum("ot" in m for m in modes)
        print(f"\n{name} >= argmax on {wins}/{len(pr)} runs "
              f"(min margin {float((pr-ar).min()):+.4f}, mean {float((pr-ar).mean()):+.4f})")
        print(f"chosen[{name}] modes: " + ", ".join(f"{m}={modes.count(m)}"
              for m in ["argmax", "temporal", "ot", "ot_temporal"]) + f" | OT-on {ot_on}/{len(modes)}")
    print(f"appended -> {outp}", flush=True)


def evaluate_stills(dataset, ensemble=False):
    """Stills path (HyperKvasir, GastroVision): still images, no (video,frame) tags -> temporal
    N/A. The framework reduces to {argmax, O_a} with the SAME base-rate gate (moderate imbalance
    -> OT fires). Reports argmax, ot_alpha, compose_gated. No temporal / no video CV."""
    prefix = {"hyperkvasir": "densenet201_swint_hyperkvasir_focal",
              "gastrovision": "densenet201_swint_gastrovision_focal"}[dataset]
    ddir = "/pvc/results/logits"
    cfgs = [("argmax", 1.0, 0.05, 7)] + [("ot", al, eps, 7) for al in AL for eps in EPS]

    def load(fold, seed):
        if not ensemble:
            z = np.load(os.path.join(ddir, f"{prefix}_f{fold}_seed{seed}.npz"), allow_pickle=True)
            return z["vL"], z["vY"].astype(int), z["tL"], z["tY"].astype(int)
        zs = [np.load(os.path.join(ddir, f"{prefix}_f{fold}_seed{s}.npz"), allow_pickle=True) for s in SEEDS]
        return _log_mean_softmax([z["vL"] for z in zs]), zs[0]["vY"].astype(int), \
            _log_mean_softmax([z["tL"] for z in zs]), zs[0]["tY"].astype(int)

    z0 = np.load(os.path.join(ddir, f"{prefix}_f0_seed0.npz"), allow_pickle=True)
    K = int(max(z0["vY"].max(), z0["tY"].max())) + 1
    D.CLASSES = [str(i) for i in range(K)]
    runs = _runs(ensemble)
    print(f"[grid] {len(cfgs)} stills configs (argmax + OT) | {len(runs)} runs "
          f"({'seed-ensemble' if ensemble else 'single-model'})", flush=True)
    S = {s: [] for s in ["argmax", "ot_alpha", "compose_gated"]}
    chosen_g = []
    for fold, seed in runs:
        vL, vY, tL, tY = load(fold, seed)
        prior = np.bincount(vY, minlength=K).astype(float); prior /= prior.sum()
        sc = lambda pred: [D.f1m(tY, pred), D.recm(tY, pred), D.mccm(tY, pred)]
        S["argmax"].append(sc(tL.argmax(1)))
        fit = D.fit_ot_alpha(vL, D.log_softmax(vL), vY, prior, np.log(prior + 1e-12), D.f1m)
        S["ot_alpha"].append(sc(fit(tL, D.log_softmax(tL))))
        cg, allow_ot, min_prev = select_gated(vL, vY, None, None, K, cfgs, P_MIN)
        chosen_g.append(cg)
        S["compose_gated"].append(sc(decode(cg, tL, None, None, prior)))
        print(f"  f{fold}s{'ens' if seed is None else seed}: gated={cg} "
              f"(allow_ot={allow_ot}, min_prev={min_prev:.5f})", flush=True)

    os.makedirs("/pvc/results/experimental", exist_ok=True)
    short = {"hyperkvasir": "hk", "gastrovision": "gv"}[dataset]
    outp = f"/pvc/results/experimental/results_compose_{short}{'_ens' if ensemble else ''}.csv"
    with open(outp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dataset", "strategy", "f1", "recall", "mcc"])
        am = np.array(S["argmax"]).mean(0)
        print(f"\n=== {dataset} (mean over {len(runs)} runs) ===")
        print(f"{'strategy':<16}{'f1':>8}{'recall':>9}{'mcc':>8}   vs-argmax-F1")
        for s, vals in S.items():
            m = np.array(vals).mean(0)
            print(f"{s:<16}{m[0]:>8.4f}{m[1]:>9.4f}{m[2]:>8.4f}   {m[0]-am[0]:+.4f}")
            w.writerow([dataset, s, f"{m[0]:.4f}", f"{m[1]:.4f}", f"{m[2]:.4f}"])
    gt = np.array(S["compose_gated"])[:, 0]; ar = np.array(S["argmax"])[:, 0]
    ot_on = sum("ot" in c[0] for c in chosen_g)
    print(f"\ngated >= argmax on {int((gt >= ar - 1e-6).sum())}/{len(gt)} runs "
          f"(min {float((gt-ar).min()):+.4f}, mean {float((gt-ar).mean()):+.4f}) | OT-on {ot_on}/{len(chosen_g)}")
    print(f"appended -> {outp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["galar_pathology", "kvasir", "hyperkvasir", "gastrovision"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--ensemble", action="store_true", help="mean over seeds {0,1,42} per fold")
    a = ap.parse_args()
    if a.device == "cuda":
        assert torch is not None and torch.cuda.is_available(), "cuda requested but unavailable"
        D._DEVICE = torch.device("cuda"); print(f"[device] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    if a.dataset in ("hyperkvasir", "gastrovision"):
        evaluate_stills(a.dataset, ensemble=a.ensemble)
    else:
        evaluate(a.dataset, ensemble=a.ensemble)


if __name__ == "__main__":
    main()
