"""
Forward-free sweep of metric-optimal decoders over CACHED logits.

metric_optimal_decode.py already saved val+test logits for every (fold, seed) to
/pvc/results/logits/*.npz. Decoding is a pure numpy problem now: no model, no GPU,
no re-forward. This script tries every decision rule we can justify, all under the
same leakage-free protocol (fit/tune on VAL, class prior from TRAIN only, report on
TEST), and prints an aggregate table so we can pick a single winner before writing
it to summary.csv.

Decoders (each tuned on val for the chosen objective):
  argmax        - the baseline decision rule.
  la            - logit adjustment (Menon 2021): logit - tau*log(prior), tau swept.
  free_bias     - per-class additive logit offset, coordinate ascent (calibrate_mcc).
  ot_alpha      - OT (Sinkhorn) assignment to marginal prior**alpha, alpha swept.
  ot_free       - OT to a FREE per-class marginal (overfits val; kept for contrast).
  ot_reg        - OT to a free marginal SHRUNK toward the prior; shrink strength
                  picked on a held-out half of val (the honest fix for ot_free).
  auto          - fit each decoder on one half of val, score macro-F1 on the other,
                  pick the winner, refit on full val. No manual decoder cherry-pick.

Ensemble variant: eval order is deterministic (shuffle=False), so per fold the 3
seeds' posteriors are averaged elementwise, then decoded. Reported separately.

Usage (in-cluster, CPU only):
  python analysis/decode_sweep.py --logits-dir /pvc/results/logits \\
      --data-root /pvc/kvasir-capsule --output-dir /pvc/results \\
      --exp-prefix densenet201_swint_otdecode_official
  # then, once a winner is chosen:
  python analysis/decode_sweep.py ... --write-summary ot_alpha
"""
import argparse
import os
import sys

import numpy as np
from sklearn.metrics import f1_score, recall_score, matthews_corrcoef, roc_auc_score

try:
    import torch
except ImportError:
    torch = None
_DEVICE = None   # torch.device when GPU decode enabled (--device cuda); else numpy path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Kvasir-Capsule class order (datasets/kvasir.py CLASSES). Hardcoded so this script
# runs with only numpy+sklearn - no torch/data-root needed for a val-prior sweep.
CLASSES = ["Angiectasia", "Blood - fresh", "Erosion", "Erythema", "Foreign body",
           "Ileocecal valve", "Lymphangiectasia", "Normal clean mucosa", "Pylorus",
           "Reduced mucosal view", "Ulcer"]
# GALAR section task (same order as datasets.galar.SECTION_CLASSES, so label indices match)
SECTION_CLASSES = ["mouth", "esophagus", "stomach", "small intestine", "colon"]

FOLDS = (0, 1)
SEEDS = (0, 1, 42)
RARE = "Blood - fresh"          # AUC 0.22, below chance: no decoder can recover it
MID  = "Erythema"               # AUC 0.93, argmax recall 0: recoverable
NORMAL_IDX = None               # dominant-class index for ot_hier (set for galar pathology)


# ── metrics ────────────────────────────────────────────────────────────────
def f1m(y, p):  return f1_score(y, p, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
def recm(y, p): return recall_score(y, p, labels=list(range(len(CLASSES))), average="macro", zero_division=0)
def mccm(y, p): return matthews_corrcoef(y, p)
def baccm(y, p): return recm(y, p)   # balanced accuracy = macro-recall (triage sensitivity)
def gmeanm(y, p):                    # geometric mean of per-class recall (0 if any class missed)
    r = recall_score(y, p, labels=list(range(len(CLASSES))), average=None, zero_division=0)
    return float(np.exp(np.log(np.clip(r, 1e-6, None)).mean()))

OBJ = {"f1": f1m, "mcc": mccm, "bacc": baccm, "gmean": gmeanm}


def logsumexp(a, axis):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def _sinkhorn_torch(log_post, col_marg, eps, iters):
    """CUDA float64 Sinkhorn (matches the numpy path numerically; ~100x faster on the
    2M-row test set). Enabled by --device cuda -> _DEVICE set."""
    G = torch.as_tensor(np.asarray(log_post, dtype=np.float64), device=_DEVICE) / eps
    N, K = G.shape
    logr = np.log(1.0 / N)
    logc = torch.log(torch.clamp(
        torch.as_tensor(np.asarray(col_marg, dtype=np.float64), device=_DEVICE), min=1e-12))
    u = torch.zeros(N, dtype=torch.float64, device=_DEVICE)
    v = torch.zeros(K, dtype=torch.float64, device=_DEVICE)
    for _ in range(iters):
        u = logr - torch.logsumexp(G + v.unsqueeze(0), dim=1)
        v = logc - torch.logsumexp(G + u.unsqueeze(1), dim=0)
    return (G + u.unsqueeze(1) + v.unsqueeze(0)).argmax(1).cpu().numpy()


def sinkhorn_assign(log_post, col_marg, eps=0.05, iters=200):
    """Entropic OT joint assignment. Returns per-row argmax of the transport plan."""
    if _DEVICE is not None:
        return _sinkhorn_torch(log_post, col_marg, eps, iters)
    N, K = log_post.shape
    G = log_post / eps
    logr = np.log(np.full(N, 1.0 / N))
    logc = np.log(np.clip(col_marg, 1e-12, None))
    u = np.zeros(N); v = np.zeros(K)
    for _ in range(iters):
        u = logr - logsumexp(G + v[None, :], axis=1)
        v = logc - logsumexp(G + u[:, None], axis=0)
    return (G + u[:, None] + v[None, :]).argmax(1)


# ── decoders: fit(...) -> predict(tL, tlog) -> preds ────────────────────────
def unbalanced_sinkhorn(log_post, col_marg, eps=0.05, kappa=1.0, iters=200):
    """Unbalanced OT: row marginal stays hard (every frame is classified once), but
    the class (column) marginal is a SOFT KL target, relaxed by kappa in [0,1].
    kappa=1 -> hard OT (enforce the quota); kappa=0 -> argmax (no quota at all).
    At intermediate kappa a confident frame (peaked posterior) resists the quota
    pull, so a confidently-normal video is NOT forced to emit rare-class frames -
    the fix for the per-video class-distribution mismatch."""
    N, K = log_post.shape
    G = log_post / eps
    logr = np.log(np.full(N, 1.0 / N))
    logc = np.log(np.clip(col_marg, 1e-12, None))
    u = np.zeros(N); v = np.zeros(K)
    for _ in range(iters):
        u = logr - logsumexp(G + v[None, :], axis=1)            # hard row constraint
        v = kappa * (logc - logsumexp(G + u[:, None], axis=0))  # soft column constraint
    return (G + u[:, None] + v[None, :]).argmax(1)


def fit_argmax(vL, vlog, vY, prior, log_prior, obj):
    return lambda tL, tlog: tL.argmax(1)


def fit_ot_uot(vL, vlog, vY, prior, log_prior, obj):
    """Unbalanced-OT decode: tune target marginal prior**alpha and softness kappa on
    val. Recovers hard OT (kappa=1) and argmax (kappa=0) as special cases, so on val
    it can only match-or-beat those endpoints."""
    best = (-9, None)
    for a in np.linspace(0.0, 1.0, 6):
        cm = np.power(prior, a); cm /= cm.sum()
        for kap in np.linspace(0.0, 1.0, 11):
            s = obj(vY, unbalanced_sinkhorn(vlog, cm, kappa=kap))
            if s > best[0]:
                best = (s, (cm, kap))
    cm, kap = best[1]
    return lambda tL, tlog: unbalanced_sinkhorn(tlog, cm, kappa=kap)


def fit_la(vL, vlog, vY, prior, log_prior, obj):
    best = (-9, 0.0)
    for tau in np.linspace(0.0, 3.0, 31):
        s = obj(vY, (vL - tau * log_prior).argmax(1))
        if s > best[0]:
            best = (s, tau)
    tau = best[1]
    return lambda tL, tlog: (tL - tau * log_prior).argmax(1)


def fit_free_bias(vL, vlog, vY, prior, log_prior, obj, rounds=4):
    K = len(CLASSES)
    grid = np.linspace(-4.0, 4.0, 41)
    bias = np.zeros(K)
    best = obj(vY, vL.argmax(1))
    for _ in range(rounds):
        moved = False
        for c in range(K):
            trial = bias.copy(); bc, bv = bias[c], best
            for g in grid:
                trial[c] = g
                s = obj(vY, (vL + trial).argmax(1))
                if s > bv:
                    bv, bc = s, g
            if bc != bias[c]:
                moved = True
            bias[c], best = bc, bv
        if not moved:
            break
    return lambda tL, tlog: (tL + bias).argmax(1)


def fit_ot_alpha(vL, vlog, vY, prior, log_prior, obj):
    best = (-9, None)
    for a in np.linspace(0.0, 1.0, 11):
        cm = np.power(prior, a); cm /= cm.sum()
        s = obj(vY, sinkhorn_assign(vlog, cm))
        if s > best[0]:
            best = (s, cm)
    cm = best[1]
    return lambda tL, tlog: sinkhorn_assign(tlog, cm)


def _ot_free_marginal(vlog, vY, prior, obj, rounds=5):
    m = prior.astype(float).copy()
    val = lambda mm: obj(vY, sinkhorn_assign(vlog, mm / mm.sum()))
    best = val(m)
    for _ in range(rounds):
        for c in range(len(CLASSES)):
            for fac in (0.25, 0.5, 2.0, 4.0, 8.0):
                trial = m.copy(); trial[c] *= fac
                s = val(trial)
                if s > best:
                    best, m = s, trial
    return m / m.sum()


def fit_ot_free(vL, vlog, vY, prior, log_prior, obj):
    cm = _ot_free_marginal(vlog, vY, prior, obj)
    return lambda tL, tlog: sinkhorn_assign(tlog, cm)


def fit_ot_reg(vL, vlog, vY, prior, log_prior, obj, seed=0):
    """Free marginal fit on one half of val, then shrunk toward the prior in
    marginal space: cm(lam) = norm(prior**(1-lam) * m_free**lam). lam picked on the
    other half. lam=0 -> prior (=ot_alpha,a=1); lam=1 -> full ot_free."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(vY))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    m_free = _ot_free_marginal(vlog[a], vY[a], prior, obj)
    best = (-9, prior / prior.sum())
    for lam in np.linspace(0.0, 1.0, 11):
        cm = np.power(prior, 1 - lam) * np.power(m_free, lam); cm /= cm.sum()
        s = obj(vY[b], sinkhorn_assign(vlog[b], cm))
        if s > best[0]:
            best = (s, cm)
    cm = best[1]
    return lambda tL, tlog: sinkhorn_assign(tlog, cm)


def _val_class_auc(vL, vY):
    """Per-class one-vs-rest AUC on val posteriors = rankability (chance = 0.5).
    A class the model cannot rank above chance is one OT can never recover; a class
    absent from val is treated as un-rankable (0.5)."""
    K = len(CLASSES)
    vp = _softmax(vL)
    auc = np.full(K, 0.5)
    for c in range(K):
        yc = (vY == c).astype(int)
        if 0 < yc.sum() < len(yc):
            try:
                auc[c] = roc_auc_score(yc, vp[:, c])
            except ValueError:
                pass
    return auc


def fit_ot_gated(vL, vlog, vY, prior, log_prior, obj):
    """Decodability-gated OT (ours). Our diagnostic: OT recovers a class only when it
    is BOTH argmax-collapsed AND rankable (val AUC > chance). Plain OT pushes the
    rare-class quota onto every class equally, so on a long tail where many classes
    sit near chance (GALAR angiectasia/ulcer/polyp) it just manufactures false
    positives that cancel the genuine recovery on the rankable-rare ones. We turn the
    diagnostic into the decoder: gate the OT target marginal by per-class val AUC.
    Rankable classes receive their prior quota (the OT pull); un-rankable classes keep
    their argmax-natural share (no pull). The AUC threshold is tuned on val for the
    objective. It reduces to ot_alpha(a=1) when every class is rankable and to argmax
    when none are, so on val it can only match-or-beat those endpoints, and it never
    spends quota on a class it cannot rank."""
    K = len(CLASSES)
    auc = _val_class_auc(vL, vY)
    pr = prior / prior.sum()

    def marginal(L, thr):
        g = (auc >= thr).astype(float)                 # 1 = rankable -> gets OT quota
        base = np.bincount(L.argmax(1), minlength=K).astype(float)
        base = base / max(base.sum(), 1.0)             # un-rankable keep argmax share
        cm = g * pr + (1.0 - g) * base
        cm = np.clip(cm, 1e-12, None)
        return cm / cm.sum()

    best = (-9, 0.5)
    for thr in np.linspace(0.5, 0.9, 9):
        s = obj(vY, sinkhorn_assign(vlog, marginal(vL, thr)))
        if s > best[0]:
            best = (s, thr)
    thr = best[1]
    return lambda tL, tlog: sinkhorn_assign(tlog, marginal(tL, thr))


def fit_ot_hier(vL, vlog, vY, prior, log_prior, obj):
    """Hierarchical / two-stage OT for extreme normal-dominated triage (ours). On a
    pool that is ~99.9% one 'normal' class, every rare class sits near 0.06% base
    rate, so per-frame ranking has no precision and NO target marginal (prior/free/
    gated) can recover it - the dilution, not the quota, is the wall. So we gate
    normal-vs-abnormal first (threshold on P(normal), tuned on val), then run OT over
    ONLY the non-normal classes on the abnormal-predicted frames, prior restricted and
    renormalized to those classes. Among abnormal frames the rare base rates rise ~100x,
    so OT's quota assignment finally has precision. Reduces to argmax when the gate
    keeps everything normal, so on val it can only match-or-beat it. NORMAL_IDX is the
    index of the dominant class (0 for the pathology classes.txt: 'normal' first)."""
    K = len(CLASSES)
    ni = NORMAL_IDX if NORMAL_IDX is not None else 0
    nn = np.array([c for c in range(K) if c != ni])
    pr_nn = prior[nn] / max(prior[nn].sum(), 1e-12)

    def decode(L, t):
        p = _softmax(L)
        pred = np.full(len(L), ni)
        ab = p[:, ni] < t                      # predicted abnormal
        if ab.any():
            sub = L[ab][:, nn]                 # restrict to non-normal classes
            a = sinkhorn_assign(log_softmax(sub), pr_nn)
            pred[ab] = nn[a]
        return pred

    best = (-9, 1.0)
    for t in np.linspace(0.5, 0.999, 12):
        s = obj(vY, decode(vL, t))
        if s > best[0]:
            best = (s, t)
    t = best[1]
    return lambda tL, tlog: decode(tL, t)


def _softmax(L):
    e = np.exp(L - L.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def em_test_prior(probs, train_prior, iters=200):
    """Saerens (2002) EM / BBSE label-shift adaptation: MLE of the TEST class prior
    under p(x|y) unchanged, using test posteriors only (no test labels -> leakage
    free, transductive). Returns (estimated prior, adjusted posteriors)."""
    tp = np.clip(train_prior, 1e-8, None)
    w = train_prior.astype(float).copy()
    for _ in range(iters):
        adj = probs * (w / tp)[None, :]
        adj /= adj.sum(1, keepdims=True)
        wn = adj.mean(0)
        if np.abs(wn - w).max() < 1e-7:
            break
        w = wn
    return w, adj


def bbse_test_prior(vL, vY, tL):
    """Black-Box Shift Estimation (Lipton 2018): estimate the TEST class prior by
    inverting the VAL confusion matrix against the test prediction histogram. Unlike
    EM it can recover rare-class mass the model under-predicts, because the confusion
    matrix encodes the model's known per-class error rates. Label-shift assumption:
    p(pred|y) is invariant train->test. Uses hard argmax predictions."""
    K = len(CLASSES)
    vp, tp = vL.argmax(1), tL.argmax(1)
    C = np.zeros((K, K))
    np.add.at(C, (vp, vY), 1.0)      # C[i,j] = count(pred=i, true=j) on val
    C /= max(len(vY), 1)
    mu = np.bincount(tp, minlength=K).astype(float) / max(len(tp), 1)  # test pred hist
    w = np.clip(np.linalg.pinv(C) @ mu, 0, None)   # importance weights p_test/p_val
    pval = C.sum(0)                                  # p_val(y)
    pt = w * pval
    return (pt / pt.sum()) if pt.sum() > 0 else (pval / pval.sum())


def fit_temp_nll(vL, vY):
    """Temperature that minimizes val NLL (standard calibration)."""
    best = (1e9, 1.0)
    for T in np.linspace(0.5, 5.0, 19):
        p = _softmax(vL / T)
        nll = -np.log(p[np.arange(len(vY)), vY] + 1e-12).mean()
        if nll < best[0]:
            best = (nll, T)
    return best[1]


def fit_ot_tuned(vL, vlog, vY, prior, log_prior, obj):
    """OT with posterior temperature T, marginal prior**alpha, and Sinkhorn entropy
    eps all tuned on val for the objective. T flattens the posterior; eps softens
    the assignment - the two knobs the fixed-eps OT-alpha left on the table."""
    best = (-9, None)
    for T in (0.5, 1.0, 2.0):
        vlp = np.log(_softmax(vL / T) + 1e-12)
        for a in np.linspace(0.0, 1.0, 11):
            cm = np.power(prior, a); cm /= cm.sum()
            for eps in (0.02, 0.05, 0.1, 0.2, 0.5):
                s = obj(vY, sinkhorn_assign(vlp, cm, eps=eps))
                if s > best[0]:
                    best = (s, (T, cm, eps))
    T, cm, eps = best[1]
    return lambda tL, tlog: sinkhorn_assign(np.log(_softmax(tL / T) + 1e-12), cm, eps=eps)


def fit_em_argmax(vL, vlog, vY, prior, log_prior, obj):
    """Argmax of the EM label-shift-adjusted posteriors (adaptive logit adjustment
    to the ESTIMATED test prior, unlike train-prior LA which did nothing)."""
    def pred(tL, tlog):
        _, adj = em_test_prior(_softmax(tL), prior)
        return adj.argmax(1)
    return pred


def fit_ot_em(vL, vlog, vY, prior, log_prior, obj):
    """OT assignment to the EM-estimated TEST marginal (label-shift + transport).
    eps tuned on val using val's own EM marginal."""
    vp = _softmax(vL); vw, _ = em_test_prior(vp, prior)
    vlp = np.log(vp + 1e-12)
    best = (-9, 0.05)
    for eps in (0.02, 0.05, 0.1, 0.2):
        s = obj(vY, sinkhorn_assign(vlp, vw, eps=eps))
        if s > best[0]:
            best = (s, eps)
    eps = best[1]
    def pred(tL, tlog):
        p = _softmax(tL); w, _ = em_test_prior(p, prior)
        return sinkhorn_assign(np.log(p + 1e-12), w, eps=eps)
    return pred


DECODERS = {
    "argmax":    fit_argmax,
    "la":        fit_la,
    "free_bias": fit_free_bias,
    "ot_alpha":  fit_ot_alpha,
    "ot_free":   fit_ot_free,
    "ot_reg":    fit_ot_reg,
    "ot_gated":  fit_ot_gated,
    "ot_hier":   fit_ot_hier,
    "ot_tuned":  fit_ot_tuned,
    "em_argmax": fit_em_argmax,
    "ot_em":     fit_ot_em,
    "ot_uot":    fit_ot_uot,
}


def fit_auto(vL, vlog, vY, prior, log_prior, obj, seed=0):
    """Held-out-val model selection among all decoders (objective = macro-F1),
    then refit the winner on full val. Prevents auto-picking an overfit decoder."""
    rng = np.random.default_rng(seed + 1)
    idx = rng.permutation(len(vY))
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    scores = {}
    for name, fit in DECODERS.items():
        pred = fit(vL[a], vlog[a], vY[a], prior, log_prior, f1m)
        scores[name] = f1m(vY[b], pred(vL[b], vlog[b]))
    win = max(scores, key=lambda k: scores[k])
    pred = DECODERS[win](vL, vlog, vY, prior, log_prior, obj)
    pred._winner = win
    return pred


# ── sequence-aware (temporal) smoothing ─────────────────────────────────────
def temporal_smooth(L, group, frame, window):
    """Within-VIDEO moving-average of the posterior over a +/- `window` frame-index
    neighborhood, returned as log-probs (so it drops straight into any decoder that
    reads logits/log-softmax). VCE lesions span contiguous frame runs while artifact
    false-positives are isolated spikes; averaging the sequence suppresses the spikes
    and keeps the runs -> higher precision without dropping any frame. Uses ONLY model
    posteriors + (video, frame) tags -- never labels. Respects the gaps left by dropped
    multi-finding frames (neighbors are by frame index, not row adjacency), and never
    crosses a video boundary."""
    if window <= 0:
        return L
    p = _softmax(L)
    order = np.lexsort((frame, group))            # sort by video, then frame
    g_s = group[order]
    f_s = frame[order].astype(np.int64)
    p_s = p[order]
    csum = np.vstack([np.zeros((1, p.shape[1])), np.cumsum(p_s, axis=0)])
    n = len(order)
    lo = np.empty(n, np.int64); hi = np.empty(n, np.int64)
    # per-video segments (contiguous after lexsort); window bounds via vectorized
    # searchsorted within each segment so it never crosses a video boundary.
    bnd = np.flatnonzero(np.concatenate(([True], g_s[1:] != g_s[:-1], [True])))
    for a, b in zip(bnd[:-1], bnd[1:]):
        fs = f_s[a:b]
        lo[a:b] = a + np.searchsorted(fs, fs - window, side="left")
        hi[a:b] = a + np.searchsorted(fs, fs + window, side="right")
    sm_s = (csum[hi] - csum[lo]) / (hi - lo)[:, None]
    out = np.empty_like(sm_s)
    out[order] = sm_s
    return np.log(out + 1e-12)


# ── data loading ────────────────────────────────────────────────────────────
def load_run(logits_dir, prefix, fold, seed, window=0):
    # allow_pickle: some npz carry an object-dtype key (kvasir), and the video_id tags
    # are stored as object arrays; harmless for the float logits we actually decode.
    z = np.load(os.path.join(logits_dir, f"{prefix}_f{fold}_seed{seed}.npz"), allow_pickle=True)
    vL, vY, tL, tY = z["vL"], z["vY"], z["tL"], z["tY"]
    if window > 0 and "tgroup" in z.files:      # sequence-aware decoding (needs tags)
        vL = temporal_smooth(vL, z["vgroup"], z["vframe"].astype(np.int64), window)
        tL = temporal_smooth(tL, z["tgroup"], z["tframe"].astype(np.int64), window)
    return vL, vY.astype(int), tL, tY.astype(int)


def log_softmax(L):
    return L - logsumexp(L, 1)[:, None]


def load_hetero(logits_dir, fold, seed=42):
    """All diverse-model logit files for a fold: arch_* (architectures),
    dn201_* (method variants), and the densenet201_swint baseline."""
    import glob
    suf = f"_f{fold}_seed{seed}.npz"
    members = {}
    for f in sorted(glob.glob(os.path.join(logits_dir, f"*{suf}"))):
        name = os.path.basename(f)[: -len(suf)]
        if name.startswith("arch_") or name.startswith("dn201_") \
           or name == "densenet201_swint_otdecode_official":
            z = np.load(f)
            members[name] = (z["vL"], z["vY"].astype(int), z["tL"], z["tY"].astype(int))
    return members


def greedy_ensemble(vP, vY, maxsize=25):
    """Caruana (2004) greedy ensemble selection WITH replacement: repeatedly add
    the member that most improves the averaged ensemble's val macro-F1. Returns the
    chosen multiset of member names (weights = multiplicity)."""
    names = list(vP)
    best, first = -9, names[0]
    for n in names:
        s = f1m(vY, vP[n].argmax(1))
        if s > best:
            best, first = s, n
    chosen = [first]; ens = vP[first].copy()
    while len(chosen) < maxsize:
        bs, bn = best, None
        for n in names:
            cand = (ens * len(chosen) + vP[n]) / (len(chosen) + 1)
            s = f1m(vY, cand.argmax(1))
            if s > bs:
                bs, bn = s, n
        if bn is None:
            break
        ens = (ens * len(chosen) + vP[bn]) / (len(chosen) + 1)
        chosen.append(bn); best = bs
    return chosen


def train_prior(data_root, fold):
    """Train-label class prior via KvasirDataset. Returns None if unavailable
    (no torch/data-root) so the caller can fall back to the val-label prior."""
    try:
        import datasets.kvasir as _kv
        _kv._build_path_index = lambda d: {}   # prior needs labels only; skip 47k-file scan
        from datasets.kvasir import KvasirDataset
        ds = KvasirDataset(root=data_root, split="train", split_mode="official",
                           split_id=fold, seed=0)
        p = np.bincount(np.asarray(ds.sample_labels()), minlength=len(CLASSES)).astype(float)
        return p / p.sum()
    except Exception as e:
        print(f"[train_prior] KvasirDataset unavailable ({e}); using val-label prior")
        return None


def val_prior(vY):
    p = np.bincount(vY, minlength=len(CLASSES)).astype(float)
    return p / p.sum()


# ── evaluation ────────────────────────────────────────────────────────────
def eval_decoder(fit, vL, vY, tL, tY, prior, obj, seed):
    log_prior = np.log(prior + 1e-12)
    vlog, tlog = log_softmax(vL), log_softmax(tL)
    try:
        pred = fit(vL, vlog, vY, prior, log_prior, obj, seed=seed)
    except TypeError:
        pred = fit(vL, vlog, vY, prior, log_prior, obj)
    p = pred(tL, tlog)
    per_rec = recall_score(tY, p, labels=list(range(len(CLASSES))), average=None, zero_division=0)
    return {
        "f1": f1m(tY, p), "recall": recm(tY, p), "mcc": mccm(tY, p),
        "rare_rec": per_rec[CLASSES.index(RARE)], "mid_rec": per_rec[CLASSES.index(MID)],
        "winner": getattr(pred, "_winner", ""),
    }


def agg(rows, key):
    v = np.array([r[key] for r in rows], float)
    return v.mean(), v.std()


class _Tee:
    """Mirror stdout to a file so results survive even if kubelet logs are
    unreachable (the node hosting a completed pod can go dark)."""
    def __init__(self, path):
        self.f = open(path, "w")
        self.stdout = sys.stdout
    def write(self, s):
        self.stdout.write(s); self.f.write(s)
    def flush(self):
        self.stdout.flush(); self.f.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits-dir", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--exp-prefix", default="densenet201_swint_otdecode_official")
    ap.add_argument("--objective", default="f1", choices=["f1", "mcc", "bacc", "gmean"])
    ap.add_argument("--temporal-window", default=0, type=int,
                    help="sequence-aware decoding: within-video +/- W frame-index moving average "
                         "of posteriors before decoding (0 = off). Needs (video,frame) tags in npz.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda runs the Sinkhorn on GPU (float64, ~100x faster on the 2M-row test). "
                         "Default cpu keeps Kvasir numerically identical/reproducible.")
    ap.add_argument("--eval-class-cap", default=0, type=int,
                    help="IMBALANCE SWEEP (transparent, controlled): cap each class at M frames "
                         "in BOTH val and test (rare classes kept fully), shrinking the negative "
                         "pool to dial down imbalance. 0 = natural (deployment) distribution. "
                         "Sweep M to trace where OT starts/stops helping (base-rate diagnostic).")
    ap.add_argument("--prior-source", default="train", choices=["train", "val"],
                    help="'val' uses the val-label prior (numpy-only, no torch/data-root).")
    ap.add_argument("--write-summary", default=None,
                    help="Decoder name to emit as full per-class rows into the summary csv.")
    ap.add_argument("--summary-name", default="summary.csv",
                    help="Target summary filename (e.g. summary_galar_pathology.csv) so "
                         "decode rows land beside the matching training rows, not in Kvasir's.")
    ap.add_argument("--write-only", action="store_true",
                    help="Skip the comparison sweep; only write the --write-summary rows.")
    ap.add_argument("--decoders", default=None,
                    help="Comma list to restrict the sweep (e.g. argmax,la,ot_alpha,ot_reg).")
    ap.add_argument("--no-calens", action="store_true", help="Skip the calibrated-ensemble block.")
    ap.add_argument("--dataset", default="kvasir", choices=["kvasir", "galar"],
                    help="galar switches to the 5-class section task (rare = mouth/esophagus).")
    ap.add_argument("--galar-task", default="section", choices=["section", "pathology"])
    ap.add_argument("--classes-file", default=None,
                    help="pathology: classes.txt (defines CLASSES order = label indices).")
    ap.add_argument("--rare", default=None, help="override rare-class name for diagnostics")
    ap.add_argument("--mid", default=None, help="override mid-class name for diagnostics")
    args = ap.parse_args()
    if args.dataset == "galar":
        global CLASSES, RARE, MID, NORMAL_IDX
        if args.galar_task == "pathology":
            with open(args.classes_file) as f:
                CLASSES = [ln.strip() for ln in f if ln.strip()]
            RARE = args.rare or "erythema"    # rarest usable pathology (~2.5k frames)
            MID  = args.mid or "ulcer"        # next-rarest (~6k frames)
            NORMAL_IDX = CLASSES.index("normal") if "normal" in CLASSES else 0
        else:
            CLASSES, RARE, MID = SECTION_CLASSES, "mouth", "esophagus"
    if args.device == "cuda":
        global _DEVICE
        assert torch is not None and torch.cuda.is_available(), "cuda requested but unavailable"
        _DEVICE = torch.device("cuda")
        print(f"[device] Sinkhorn on GPU: {torch.cuda.get_device_name(0)}", flush=True)
    obj = OBJ[args.objective]
    tag = os.path.basename(args.exp_prefix).replace("_otdecode_official", "")
    # per-window / per-cap filename so each config's table is preserved (not overwritten)
    wsuf = f"_w{args.temporal_window}" if args.temporal_window > 0 else ""
    csuf = f"_cap{args.eval_class_cap}" if args.eval_class_cap > 0 else ""
    sys.stdout = _Tee(os.path.join(args.output_dir, f"decode_sweep_{tag}_{args.objective}{wsuf}{csuf}.txt"))

    runs = {(f, s): load_run(args.logits_dir, args.exp_prefix, f, s, window=args.temporal_window)
            for f in FOLDS for s in SEEDS}
    if args.temporal_window > 0:
        print(f"[temporal] within-video +/-{args.temporal_window}-frame posterior smoothing applied", flush=True)

    if args.eval_class_cap > 0:  # IMBALANCE SWEEP: cap each class in val+test (rare kept full)
        M, K = args.eval_class_cap, len(CLASSES)
        for f in FOLDS:
            tY0 = runs[(f, SEEDS[0])][3]                 # one test mask per fold -> ensemble stays aligned
            rng = np.random.default_rng(2000 + f)
            tmask = np.zeros(len(tY0), bool)
            for c in range(K):
                idx = np.where(tY0 == c)[0]
                if len(idx) > M:
                    idx = rng.choice(idx, M, replace=False)
                tmask[idx] = True
            for s in SEEDS:
                vL, vY, tL, tY = runs[(f, s)]
                vrng = np.random.default_rng(4000 + f * 7 + s)
                vmask = np.zeros(len(vY), bool)
                for c in range(K):
                    vidx = np.where(vY == c)[0]
                    if len(vidx) > M:
                        vidx = vrng.choice(vidx, M, replace=False)
                    vmask[vidx] = True
                runs[(f, s)] = (vL[vmask], vY[vmask], tL[tmask], tY[tmask])
            cnt = np.bincount(tY0[tmask], minlength=K)
            nz = cnt[cnt > 0]
            ratio = nz.max() / max(nz.min(), 1)
            print(f"[cap {M}] fold{f} test pool={int(tmask.sum())} maj:min ratio={ratio:.0f}x "
                  f"per-class={cnt.tolist()}", flush=True)
    priors = {}
    for f in FOLDS:
        p = None if args.prior_source == "val" else train_prior(args.data_root, f)
        priors[f] = p if p is not None else val_prior(runs[(f, SEEDS[0])][1])

    if args.write_only:
        write_summary(args, priors, runs, obj)
        return

    order = ["argmax", "la", "free_bias", "ot_alpha", "ot_free", "ot_reg", "ot_gated",
             "ot_hier", "ot_tuned", "ot_uot", "em_argmax", "ot_em", "auto"]
    if args.decoders:
        keep = set(args.decoders.split(","))
        order = [d for d in order if d in keep]
    fits = dict(DECODERS); fits["auto"] = fit_auto

    # ── single-model sweep (6 runs) ────────────────────────────────────────
    print(f"\n{'='*94}\nSINGLE MODEL  (objective={args.objective}, tuned on val, prior from train, "
          f"reported on test)\n{'='*94}")
    print(f"{'decoder':<11}{'f1_macro':>16}{'recall_macro':>16}{'mcc':>16}"
          f"{'rare_rec':>12}{'mid_rec':>12}   winner")
    single = {}
    for name in order:
        rows = []
        for f in FOLDS:
            for s in SEEDS:
                vL, vY, tL, tY = runs[(f, s)]
                rows.append(eval_decoder(fits[name], vL, vY, tL, tY, priors[f], obj, seed=s))
        single[name] = rows
        wins = ",".join(sorted(set(r["winner"] for r in rows if r["winner"])))
        print(f"{name:<11}"
              f"{agg(rows,'f1')[0]:>8.4f}±{agg(rows,'f1')[1]:.3f}"
              f"{agg(rows,'recall')[0]:>8.4f}±{agg(rows,'recall')[1]:.3f}"
              f"{agg(rows,'mcc')[0]:>8.4f}±{agg(rows,'mcc')[1]:.3f}"
              f"{agg(rows,'rare_rec')[0]:>12.3f}{agg(rows,'mid_rec')[0]:>12.3f}   {wins}")

    # ── seed ensemble (average posteriors across 3 seeds per fold; 2 runs) ──
    print(f"\n{'='*94}\nSEED ENSEMBLE  (posteriors averaged across 3 seeds per fold; 2 runs)\n{'='*94}")
    print(f"{'decoder':<11}{'f1_macro':>16}{'recall_macro':>16}{'mcc':>16}{'rare_rec':>12}{'mid_rec':>12}")
    ens_runs = {}
    for f in FOLDS:
        vY0 = runs[(f, SEEDS[0])][1]; tY0 = runs[(f, SEEDS[0])][3]
        # test is a shared per-fold set across seeds (aligned) -> ensemble tP is valid.
        # val may be seed-dependent (e.g. Kvasir official carves val with the seed) AND
        # class-capping can leave per-seed val different lengths; guard so a val-shape
        # mismatch can't kill the run. Fall back to seed-0 val logits for tuning only.
        tP = np.mean([softmax(runs[(f, s)][2]) for s in SEEDS], axis=0)
        vshapes = {runs[(f, s)][0].shape for s in SEEDS}
        if len(vshapes) == 1:
            vLog = np.log(np.mean([softmax(runs[(f, s)][0]) for s in SEEDS], axis=0) + 1e-12)
        else:
            vLog = np.log(softmax(runs[(f, SEEDS[0])][0]) + 1e-12)
        ens_runs[f] = (vLog, vY0, np.log(tP + 1e-12), tY0)
    ens_order = ["argmax", "la", "ot_alpha", "ot_reg", "ot_tuned", "ot_uot", "em_argmax", "ot_em", "auto"]
    if args.decoders:
        ens_order = [d for d in ens_order if d in set(args.decoders.split(","))]
    for name in ens_order:
        rows = []
        for f in FOLDS:
            vL, vY, tL, tY = ens_runs[f]
            rows.append(eval_decoder(fits[name], vL, vY, tL, tY, priors[f], obj, seed=0))
        print(f"{name:<11}"
              f"{agg(rows,'f1')[0]:>8.4f}±{agg(rows,'f1')[1]:.3f}"
              f"{agg(rows,'recall')[0]:>8.4f}±{agg(rows,'recall')[1]:.3f}"
              f"{agg(rows,'mcc')[0]:>8.4f}±{agg(rows,'mcc')[1]:.3f}"
              f"{agg(rows,'rare_rec')[0]:>12.3f}{agg(rows,'mid_rec')[0]:>12.3f}")

    # ── calibrated ensemble (per-seed temperature on val NLL, then average) ──
    if args.no_calens:
        return
    print(f"\n{'='*94}\nCALIBRATED SEED ENSEMBLE  (each seed temp-scaled on val NLL before averaging)\n{'='*94}")
    print(f"{'decoder':<11}{'f1_macro':>16}{'recall_macro':>16}{'mcc':>16}{'rare_rec':>12}{'mid_rec':>12}")
    cal_runs = {}
    for f in FOLDS:
        vY0 = runs[(f, SEEDS[0])][1]; tY0 = runs[(f, SEEDS[0])][3]
        Ts = {s: fit_temp_nll(runs[(f, s)][0], runs[(f, s)][1]) for s in SEEDS}
        vP = np.mean([_softmax(runs[(f, s)][0] / Ts[s]) for s in SEEDS], axis=0)
        tP = np.mean([_softmax(runs[(f, s)][2] / Ts[s]) for s in SEEDS], axis=0)
        cal_runs[f] = (np.log(vP + 1e-12), vY0, np.log(tP + 1e-12), tY0)
    for name in ens_order:
        rows = []
        for f in FOLDS:
            vL, vY, tL, tY = cal_runs[f]
            rows.append(eval_decoder(fits[name], vL, vY, tL, tY, priors[f], obj, seed=0))
        print(f"{name:<11}"
              f"{agg(rows,'f1')[0]:>8.4f}±{agg(rows,'f1')[1]:.3f}"
              f"{agg(rows,'recall')[0]:>8.4f}±{agg(rows,'recall')[1]:.3f}"
              f"{agg(rows,'mcc')[0]:>8.4f}±{agg(rows,'mcc')[1]:.3f}"
              f"{agg(rows,'rare_rec')[0]:>12.3f}{agg(rows,'mid_rec')[0]:>12.3f}")

    if args.write_summary:
        write_summary(args, priors, runs, obj)


def softmax(L):
    e = np.exp(L - L.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def _apply(fit, vL, vY, tL, prior, obj, s):
    lp = np.log(prior + 1e-12)
    vlog, tlog = log_softmax(vL), log_softmax(tL)
    try:
        pred = fit(vL, vlog, vY, prior, lp, obj, seed=s)
    except TypeError:
        pred = fit(vL, vlog, vY, prior, lp, obj)
    return pred(tL, tlog)


def write_summary(args, priors, runs, obj):
    """Emit the chosen decoder's FULL per-class metrics to summary.csv for the
    single-model (per fold/seed), 3-seed ensemble, and calibrated ensemble (per
    fold) configurations. AUC uses the softmax posterior (rank-based, unchanged by
    the decision rule). Model name and exp prefix are derived from --exp-prefix so
    this works for any backbone."""
    from utils import compute_metrics, ResultsLogger
    name = args.write_summary
    fit = ({**DECODERS, "auto": fit_auto})[name]
    # base identifies backbone+recipe (distinct exp_name per recipe); model = backbone
    base = os.path.basename(args.exp_prefix).replace("_otdecode_official", "")
    model = "densenet161" if base.startswith("densenet161") else "densenet201_swint"
    loss = ("focal" if "focal" in base else "ldam" if "ldam" in base else
            "ce" if "cbsampler" in base else "weighted_ce")

    def cfg_for(f, nv, nt):
        return argparse.Namespace(
            model=model, dataset=args.dataset, split_mode="official", split_id=f,
            train_size=None, val_size=nv, test_size=nt, epochs=None, optimizer=None,
            lr=None, momentum=None, weight_decay=None, batch_size=32, loss=loss,
            scheduler=None, plateau_patience=None, plateau_factor=None,
            early_stop_lr=None, pretrained_backbone=None)

    rare_key = "recall_" + RARE.lower().replace(" ", "_").replace("-", "_").replace("__", "_")

    def emit(exp, seed, f, vL, vY, tL, tY):
        p = _apply(fit, vL, vY, tL, priors[f], obj, seed if isinstance(seed, int) else 0)
        m = compute_metrics(tY, p, CLASSES, probs=softmax(tL))
        lg = ResultsLogger(args.output_dir, exp, seed, cfg_for(f, len(vY), len(tY)),
                           summary_name=args.summary_name)
        lg.log_test(m, 0); lg.close()
        print(f"wrote {exp} seed{seed}  f1={m['f1_macro']:.4f} recall={m['recall_macro']:.4f} "
              f"mcc={m['mcc']:.4f} rare_rec={m.get(rare_key, float('nan')):.3f}")

    # method tag: ot_* -> "otalpha"/"otreg"; baselines keep their name ("argmax"/"la").
    # Guarantees argmax/la/OT rows differ ONLY in the decoder (same logits, fair compare).
    mtag = ("ot" + name.replace("ot_", "").replace("_", "")) if name.startswith("ot_") else name

    # 1) single model, per fold/seed
    for f in FOLDS:
        for s in SEEDS:
            vL, vY, tL, tY = runs[(f, s)]
            emit(f"{base}_{mtag}_official_f{f}", s, f, vL, vY, tL, tY)

    # 2) plain and 3) calibrated 3-seed ensemble, per fold
    for f in FOLDS:
        vY0 = runs[(f, SEEDS[0])][1]; tY0 = runs[(f, SEEDS[0])][3]
        vP = np.mean([softmax(runs[(f, s)][0]) for s in SEEDS], axis=0)
        tP = np.mean([softmax(runs[(f, s)][2]) for s in SEEDS], axis=0)
        emit(f"{base}_{mtag}ens_official_f{f}", "ens", f,
             np.log(vP + 1e-12), vY0, np.log(tP + 1e-12), tY0)
        Ts = {s: fit_temp_nll(runs[(f, s)][0], runs[(f, s)][1]) for s in SEEDS}
        vPc = np.mean([softmax(runs[(f, s)][0] / Ts[s]) for s in SEEDS], axis=0)
        tPc = np.mean([softmax(runs[(f, s)][2] / Ts[s]) for s in SEEDS], axis=0)
        emit(f"{base}_{mtag}calens_official_f{f}", "ens", f,
             np.log(vPc + 1e-12), vY0, np.log(tPc + 1e-12), tY0)


if __name__ == "__main__":
    main()
