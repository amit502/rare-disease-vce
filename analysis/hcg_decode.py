"""
HCG -- Hierarchical Conditional decoding with Granularity selection.

The reframe.
------------
Every decoder on this project, and the plug-in theorem in analysis/pix_decode.py that explains why
they all failed, treats the class prevalence pi_c as ONE number. For rare disease it is a product of
two very different things:

    pi_frame(c)  =  P(patient has c)  x  P(frame shows c | patient has c)
       1e-4      =       ~0.02        x            ~0.005 - 0.05

A patient with angioectasia has MANY angioectasia frames. The disease is rare ACROSS patients, not
WITHIN one. Collapsing that product into a single frame-level marginal is what puts the problem below
the base-rate wall, and every offset-based decoder then inherits the wall.

Why this escapes the wall (a corollary of the plug-in theorem, not a new assumption).
-------------------------------------------------------------------------------------
The theorem says precision is linear in prevalence as pi -> 0, so the achievability correction
cancels the -log pi boost and no frame-level offset can help. But the frame decision FACTORISES into
two decisions that are both far above the wall:

  * "does this video contain c?"  -- decided at pi ~ 0.02, i.e. ~200x the frame rate, and on an
    aggregated video score whose FPR is much lower because averaging frames suppresses noise;
  * "which frames, given the video contains c?" -- decided at pi ~ 0.05, i.e. ~500x the frame rate.

The product of the two prevalences is conserved; the DIFFICULTY is not, because precision depends on
pi AND on FPR, and both improve under aggregation. The frame-level problem is below the wall; its two
factors are not.

Crucially this leaves the family the theorem constrains. The theorem bounds decoders of the form
argmax_c [log p_c + delta_c] with delta a single global vector -- which is what all 28 previous
decoders were, and why they all traced one Pareto frontier. Here delta_c depends on the PATIENT:

    argmax_c [ log p_c(x) + delta_c(v) ] ,     v = the video x belongs to.

That is why the precision/recall trade that killed everything else need not apply. Uniform
rare-boosting raises recall and destroys precision because it fires in every video. This fires only
where the video-level evidence supports the disease, and actively SUPPRESSES c in videos without it,
so precision is protected exactly where the previous methods bled it.

The model.
----------
Per class c, with everything below fitted on VAL only:

  r_c    = P(frame shows c | video contains c)   -- within-video prevalence given present. Estimated
           from val videos containing c. Each such video contributes MANY frames, so this is far
           better determined than any frame-level rare-class statistic.
  q_c(v) = P(video v contains c | evidence from v's own frames). A presence model applied to a
           top-k aggregate of p_c over the frames of v.

Then the patient-conditional prevalence and offset are

    pi_c(v) = q_c(v) r_c + (1 - q_c(v)) alpha pi_c        (alpha pi_c smooths the absent case)
    delta_c(v) = lam * log( pi_c(v) / pi_c )

with alpha and lam selected on val macro-F1. q_c(v) -> 1 gives a boost of log(r_c/pi_c), which is
large and positive; q_c(v) -> 0 suppresses c in that patient.

Leakage discipline (rule 1).
----------------------------
Fitted on val: r_c, pi_c, the presence model, alpha, lam, the aggregation fraction, the temperature.
Computed per test video from ITS OWN frames only: the aggregate score and hence q_c(v). No statistic
is ever pooled across test videos, so the decoder runs on one patient in isolation, which is the
deployment condition. Within-video aggregation is explicitly permitted; cohort aggregation is not.

The presence model borrows strength across classes. Rare classes appear in very few val videos, so a
per-class fit would repeat exactly the O(1/n_c) variance failure that sank the tail-extrapolation
attempt. Scores are standardised per class using val statistics and ONE pooled logistic is fitted
over all (class, video) pairs; a per-class fit is used only where a class has >= MINVID positive val
videos.

Ablations, each isolating one claim:
  hcg_const  -- q_c(u) replaced by its VAL MEAN, so delta_c is a single global vector again. This is
                the decisive ablation: it removes unit-conditioning while keeping every other moving
                part, dropping the method back inside the family the theorem bounds. If HCG works and
                hcg_const does not, the gain comes from conditioning on the unit.
  hcg_hard   -- q_c(u) thresholded to {0,1} instead of a soft posterior.
  hcg_video  -- granularity FIXED to the whole video, i.e. exactly HPC. The gap to hcg is the value
                of selecting the granularity.
  hcg_L*     -- one row per fixed granularity, so the whole granularity axis is visible at once.
  argmax / la_tau / ot_alpha / mean_temporal / ega_b4 -- external baselines, ega_b4 being our best.

What HPC got wrong, and the generalisation.
-------------------------------------------
HPC fixed the conditioning unit to be the VIDEO. That won Kvasir on every metric but LOST GALAR on
both scopes, and worse, the ablation that is supposed to prove the mechanism (hpc_const, which
freezes q at its val mean and so collapses delta back to a global vector) INVERTED there: 0.1871 >
0.1861 single, 0.1966 > 0.1946 ens. Every GALAR fold also selected the most conservative grid corner
(alpha=1.0, which disables suppression entirely, and lam=0.25), i.e. validation detected the signal
was untrustworthy and switched the method off.

The factorisation never required the unit to be a video. For ANY unit U,

    pi_frame(c) = P(U contains c) x P(frame shows c | U contains c) ,

and the method has leverage exactly when presence VARIES across units. If most GALAR videos contain
most classes then q_c(v) is near-constant, the patient-conditional offset degenerates to a global one
-- which is precisely the observed hpc ~ hpc_const signature. Shortening the unit restores the
variance: a disease present in every video is still absent from most 100-frame segments of it.

So conditioning GRANULARITY is the free variable, and it unifies two things previously treated as
unrelated: mean_temporal (a tiny window, no presence model) and HPC (the whole video) are the two
ends of ONE axis. The optimum lies in between and is selectable on val. This also explains why plain
temporal smoothing was ever competitive on these datasets.

Units are contiguous runs of L frames inside a video (L=None means the whole video), so a test unit
is still built only from the test video's own frames -- rule 1 is unchanged. L is selected on val
macro-F1. A per-class L is reported as an ablation but is NOT the headline: selecting L per class
from a rare class's handful of val frames would reintroduce exactly the O(1/n_c) variance that sank
the tail-extrapolation attempt.

Diagnostics printed per granularity: the per-class presence RATE (fraction of units containing the
class) and its variance. This directly tests the GALAR hypothesis rather than assuming it -- if
presence rate is ~1.0 at video level and drops toward 0 as L shrinks, the diagnosis is confirmed.

  python analysis/hcg_decode.py --dataset kvasir|galar_pathology --device cuda [--ensemble]
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compose_principled as C
import decode_sweep as D
import ega_v5 as E

W_TEMP = 7
MCC_TOL = 0.002       # allowed val-MCC loss vs the temporal baseline (same tolerance as ega_auto)
SCALES = (1.0, 0.5, 0.25)   # shrinkage on the conditional offset, chosen under the MCC constraint
VIDEO_DS = {"kvasir", "galar_pathology"}
TEMPS = (0.5, 1.0, 2.0, 4.0)
TOPK = (0.02, 0.05, 0.10, 0.25)      # fraction of a video's frames used by the presence aggregate
ALPHAS = (0.02, 0.1, 0.3, 1.0)       # residual prevalence weight when the video looks negative
LAMS = (0.25, 0.5, 0.75, 1.0)        # strength of the patient-conditional offset
MINVID = 5                            # positive val units needed before a per-class presence fit
# conditioning granularities: None = whole video, else contiguous runs of L frames within a video
GRANS = (None, 1000, 300, 100, 30, 10, 3)


def prefix_ddir(ds):
    return {
        "galar_pathology": ("densenet201_swint_galar_pathology_focal", "/pvc/results/logits"),
        "kvasir":          ("densenet201_swint_focal_otdecode_official", "/pvc/results/logits_kvtemporal"),
        # class-balanced-sampler ablation: does the HCE/PACT family do even better on a base
        # model whose raw rare recall is already ~2x focal's? Separate dataset key so it writes
        # its own results_hcg.csv rows instead of overwriting the reported "kvasir" (focal) ones.
        "kvasir_cbsampler": ("densenet201_swint_cbsampler_otdecode_official", "/pvc/results/logits_kvtemporal"),
    }[ds]


def load(ds, prefix, ddir, fold, seed, ensemble):
    if not ensemble:
        z = np.load(os.path.join(ddir, f"{prefix}_f{fold}_seed{seed}.npz"), allow_pickle=True)
        base = (z["vL"], z["vY"].astype(int), z["tL"], z["tY"].astype(int))
    else:
        zs = [np.load(os.path.join(ddir, f"{prefix}_f{fold}_seed{s}.npz"), allow_pickle=True)
              for s in C.SEEDS]
        z = zs[0]
        base = (C._log_mean_softmax([x["vL"] for x in zs]), z["vY"].astype(int),
                C._log_mean_softmax([x["tL"] for x in zs]), z["tY"].astype(int))
    return base + (z["vgroup"], z["tgroup"],
                   z["vframe"].astype(np.int64), z["tframe"].astype(np.int64))


def _groups_from_inv(inv):
    """Frame indices grouped by unit id. Sort-and-split is O(N log N); the obvious
    [flatnonzero(inv==i) for i in ...] is O(n_units * N) and costs seconds per call once the finest
    granularity produces ~50k units."""
    n = int(inv.max()) + 1 if len(inv) else 0
    order = np.argsort(inv, kind="stable")
    cuts = np.searchsorted(inv[order], np.arange(1, n))
    return np.split(order, cuts)


def video_index(groups):
    """Map the group/video id array to (unique ids, index array, list of frame-index arrays)."""
    uniq, inv = np.unique(groups, return_inverse=True)
    return uniq, inv, _groups_from_inv(inv)


def unit_index(groups, frames, L):
    """Partition frames into conditioning units. L=None gives one unit per video; otherwise each
    video is cut into contiguous runs of L frames in temporal order. Units never span videos, so a
    unit built at test time uses only the frames of the video being decoded."""
    if L is None:
        return video_index(groups)[1:]
    order = np.lexsort((frames, groups))
    uid = np.empty(len(groups), dtype=np.int64)
    g_sorted = groups[order]
    start, nxt = 0, 0
    for i in range(1, len(order) + 1):
        if i == len(order) or g_sorted[i] != g_sorted[start]:
            n = i - start
            uid[order[start:i]] = nxt + (np.arange(n) // L)
            nxt += int(np.ceil(n / L))
            start = i
    _, inv = np.unique(uid, return_inverse=True)
    return inv, _groups_from_inv(inv)


def presence_labels(Y, idx, K):
    """Boolean [n_units, K]: does this unit contain the class? VAL only."""
    Z = np.zeros((len(idx), K), dtype=bool)
    for i, fr in enumerate(idx):
        if len(fr):
            Z[i, np.unique(Y[fr])] = True
    return Z


def within_rate(Y, idx, Z, K, prior):
    """r_c = P(frame shows c | unit contains c), averaged over the val units that contain c."""
    r = np.zeros(K)
    for c in range(K):
        vals = [float(np.mean(Y[fr] == c)) for i, fr in enumerate(idx) if Z[i, c] and len(fr)]
        r[c] = float(np.mean(vals)) if vals else prior[c]
    return np.clip(r, 1e-6, 1.0)


def topk_aggregate(P, idx, frac):
    """Per (unit, class) evidence: the mean of the top-k posteriors for that class WITHIN the unit.
    A disease occupying a burst of B frames shows up as B high scores, so a top-k mean is far more
    sensitive than the unit mean and far more robust than the max. Computed per unit in isolation."""
    nV, K = len(idx), P.shape[1]
    S = np.zeros((nV, K))
    for i, fr in enumerate(idx):
        if len(fr) == 0:
            continue
        k = max(1, int(round(frac * len(fr))))
        blk = P[fr]                                  # this video's frames only
        if k >= len(fr):
            S[i] = blk.mean(0)
        else:
            part = np.partition(blk, len(fr) - k, axis=0)[len(fr) - k:]
            S[i] = part.mean(0)
    return S


def fast_f1(y, pred, K):
    """Macro-F1 via a bincount confusion matrix. Bit-identical to sklearn's
    f1_score(average='macro', labels=range(K), zero_division=0) but ~140x faster, which matters
    because the selection sweep evaluates it thousands of times. Exactness verified against sklearn;
    all REPORTED metrics still go through D.f1m so they stay comparable with every other result."""
    cm = np.bincount(y * K + pred, minlength=K * K).reshape(K, K).astype(np.float64)
    tp = np.diag(cm); P = cm.sum(0); T = cm.sum(1)
    return float(np.mean(np.where(P + T > 0, 2.0 * tp / np.maximum(P + T, 1e-12), 0.0)))


def roc_auc(score, pos):
    """Rank AUC of `score` against boolean `pos`; nan when a class is all/never present."""
    n1 = int(pos.sum()); n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = np.empty(len(score)); r[np.argsort(score, kind="stable")] = np.arange(len(score))
    return float((r[pos].sum() - n1 * (n1 - 1) / 2.0) / (n0 * n1))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_presence(Sv, Zv, K):
    """Fit P(video contains c | aggregate score). Scores are standardised per class on val, then ONE
    pooled logistic is fitted over all (class, video) pairs so that rare classes -- which appear in
    only a handful of val videos -- borrow the shape from the well-populated ones. A per-class fit
    replaces the pooled one only where the class has >= MINVID positive val videos.
    Returns (mu, sd, pooled_coef, per_class_coef_or_None)."""
    mu = Sv.mean(0)
    sd = Sv.std(0) + 1e-9
    X = (Sv - mu[None, :]) / sd[None, :]

    def _fit(x, y):
        """Two-parameter logistic by Newton steps; returns (w, b)."""
        w, b = 1.0, 0.0
        for _ in range(50):
            p = _sigmoid(w * x + b)
            g = np.array([np.sum((p - y) * x), np.sum(p - y)])
            wgt = p * (1 - p) + 1e-9
            H = np.array([[np.sum(wgt * x * x), np.sum(wgt * x)],
                          [np.sum(wgt * x),     np.sum(wgt)]]) + 1e-6 * np.eye(2)
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            w -= step[0]; b -= step[1]
            if np.abs(step).max() < 1e-8:
                break
        return float(w), float(b)

    pooled = _fit(X.ravel(), Zv.ravel().astype(float))
    per = {}
    for c in range(K):
        if Zv[:, c].sum() >= MINVID and Zv[:, c].sum() < len(Zv):
            per[c] = _fit(X[:, c], Zv[:, c].astype(float))

    # Per-class INTERCEPT calibration. The pooled logistic shares one intercept across classes whose
    # video-level base rates differ by orders of magnitude, so its q is badly miscalibrated per class
    # even though it ranks well. Split the difficulty: the SLOPE is what needs many positives, so it
    # stays pooled; the INTERCEPT only needs a count, which is well determined even for a class
    # present in a handful of val videos. Solve for b_c so the mean predicted presence over val
    # videos matches that class's observed video-level prevalence.
    for c in range(K):
        w, b = per.get(c, pooled)
        target = float(Zv[:, c].mean())
        if target <= 0.0 or target >= 1.0:
            per[c] = (w, b)
            continue
        lo, hi = -40.0, 40.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if _sigmoid(w * X[:, c] + mid).mean() < target:
                lo = mid
            else:
                hi = mid
        per[c] = (w, 0.5 * (lo + hi))
    return mu, sd, pooled, per


def cond_fit(baseL, Y, g, f, K, L, T, frac):
    """Fit the whole conditional model on ONE split of val (prior, within-unit rate, presence model,
    per-class reliability). Everything it returns is derived only from the rows handed to it, so it
    can be used for honest out-of-sample scoring."""
    prior = np.bincount(Y, minlength=K).astype(float)
    prior = prior / max(prior.sum(), 1)
    inv, idx = unit_index(g, f, L)
    Z = presence_labels(Y, idx, K)
    r = within_rate(Y, idx, Z, K, prior)
    valid = (prior > 0) & (Z.sum(0) > 0)
    P = D._softmax(baseL / T)
    S = topk_aggregate(P, idx, frac)
    mu, sd, pooled, per = fit_presence(S, Z, K)
    rel = reliability(crossfit_presence(S, Z, unit_videos(g, idx), K), Z, valid, K)
    return dict(prior=prior, r=r, valid=valid, mu=mu, sd=sd, pooled=pooled, per=per, rel=rel)


def cond_adjust(m, baseL, g, f, K, L, T, frac, alpha, lam, scale):
    """Apply a fitted conditional model to arbitrary rows -> reliability-gated adjusted log-probs."""
    inv, idx = unit_index(g, f, L)
    P = D._softmax(baseL / T)
    logP = np.log(np.clip(P, 1e-12, None))
    Q = presence_prob(topk_aggregate(P, idx, frac), m["mu"], m["sd"], m["pooled"], m["per"], K)
    Dl = hpc_offsets(Q, m["r"], m["prior"], alpha, lam * m["rel"] * scale, m["valid"])
    return logP + Dl[inv]


def strategy_predict(mode, scale, cfgc, bv, K,
                     bL_fit, Y_fit, g_fit, f_fit, bL_app, g_app, f_app):
    """Fit `mode` on the FIT rows and predict the APP rows. Used both for honest 2-fold-by-video
    cross-validation on val (selection) and for the final full-val fit applied to test."""
    L, T, frac, alpha, lam, Te = cfgc
    if mode == "ega":
        Pf = D._softmax(bL_fit / Te); Pa = D._softmax(bL_app / Te)
        st = E.ega_fit(Pf, bL_fit.argmax(1), Y_fit, K, beta=bv)
        return E.ega_apply(Pa, bL_app.argmax(1), st)
    m = cond_fit(bL_fit, Y_fit, g_fit, f_fit, K, L, T, frac)
    adj_a = cond_adjust(m, bL_app, g_app, f_app, K, L, T, frac, alpha, lam, scale)
    if mode == "cond":
        return adj_a.argmax(1)
    adj_f = cond_adjust(m, bL_fit, g_fit, f_fit, K, L, T, frac, alpha, lam, scale)
    st = E.ega_fit(D._softmax(adj_f), adj_f.argmax(1), Y_fit, K, beta=bv)
    return E.ega_apply(D._softmax(adj_a), adj_a.argmax(1), st)


def unit_videos(groups, idx):
    """Which video each unit came from (units never span videos)."""
    return np.array([groups[fr[0]] if len(fr) else -1 for fr in idx])


def crossfit_presence(S, Z, uvid, K, nfold=2, seed=0):
    """Out-of-sample presence posteriors, split BY VIDEO so no unit is scored by a model that saw
    another unit of the same patient.

    Why this is required, not cosmetic: fitting the presence model on the val units and then scoring
    the selection criterion on those same units makes q_c(u) recover the true presence indicator
    in-sample. The offsets then restrict each val unit to exactly the classes it contains and val
    macro-F1 goes to 1.0 -- observed on Kvasir. Granularity is then chosen by whichever L memorises
    val best (coarse L, fewer units), which is not the quantity we want to maximise. Cross-fitting
    makes the criterion out-of-sample, and its by-product -- the held-out AUC of q against true
    presence -- is the DETECTABILITY of the presence signal, the precondition the method actually
    needs and which was never measured.

    The model applied to TEST is still fitted on ALL val units; these folds only score selection."""
    vids = np.unique(uvid)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(vids))
    assign = {v: int(perm[i] % nfold) for i, v in enumerate(vids)}
    ufold = np.array([assign[v] for v in uvid])
    Q = np.full(S.shape, np.nan)
    for f in range(nfold):
        tr, te = ufold != f, ufold == f
        if tr.sum() < 2 or te.sum() == 0 or Z[tr].sum() == 0:
            continue
        mu, sd, pooled, per = fit_presence(S[tr], Z[tr], K)
        Q[te] = presence_prob(S[te], mu, sd, pooled, per, K)
    bad = np.isnan(Q).any(1)
    if bad.any():                      # folds that could not be fitted fall back to the prior rate
        Q[bad] = Z.mean(0)[None, :]
    return Q


def presence_prob(S, mu, sd, pooled, per, K):
    """Apply the val-fitted presence model. Each row of S is one video, scored from its own frames."""
    X = (S - mu[None, :]) / sd[None, :]
    Q = np.zeros_like(X)
    for c in range(K):
        w, b = per.get(c, pooled)
        Q[:, c] = _sigmoid(w * X[:, c] + b)
    return Q


def reliability(Qcv, Z, valid, K):
    """Per-class conditioning reliability from the CROSS-FITTED presence posteriors:

        rel_c = 2 * max(AUC_c - 0.5, 0)      (Somers' D; 0 at chance, 1 at perfect detection)

    Why this is needed and not a knob: the offset magnitude log(r_c/pi_c) is LARGEST for the rarest
    classes, so if their presence signal is noise the decoder applies its biggest boosts on its least
    reliable evidence. That is the observed GALAR signature -- conditioning actively hurting, with
    hcg_const beating hcg four times out of four. Weighting by measured reliability makes the method
    self-limiting: it conditions only as far as the conditioning signal is verifiable, and switches
    itself off entirely for classes whose presence cannot be detected. Held-out, val-only, untuned."""
    rel = np.zeros(K)
    for c in range(K):
        if not valid[c]:
            continue
        a = roc_auc(Qcv[:, c], Z[:, c])
        rel[c] = 0.0 if not np.isfinite(a) else float(np.clip(2.0 * (a - 0.5), 0.0, 1.0))
    return rel


def rank_scores(P, idx):
    """Descending rank of p_c WITHIN each unit, per class (rank 0 = highest scoring frame in that
    unit for that class). Computed per unit in isolation, so it uses only the decoded video."""
    R = np.zeros_like(P)
    rows = None
    for fr in idx:
        if len(fr) == 0:
            continue
        blk = P[fr]
        order = np.argsort(-blk, axis=0, kind="stable")
        rk = np.empty_like(order)
        rows = np.broadcast_to(np.arange(len(fr))[:, None], order.shape)
        np.put_along_axis(rk, order, rows, axis=0)
        R[fr] = rk
    return R


def budget_gate(P, idx, inv, Q, r, tau=0.3):
    """Soft within-unit top-m gate, m = q_c(u) * r_c * n_u = the EXPECTED number of frames in this
    unit that actually show class c.

    This is the second factor of the prevalence factorisation, which the earlier versions collapsed
    into the constant r_c:
        pi_frame(c) = P(unit has c) x P(frame shows c | unit has c)
    Modelling the second factor as a constant means that once q_c(u) ~ 1 EVERY frame of the unit gets
    the identical boost -- including the 85-95% with no lesion, since r_c is typically 0.05-0.15. Those
    are false positives stolen from the dominant normal class, which is exactly what MCC punishes, and
    it is why the global MCC constraint could only recover MCC by shrinking the boost and discarding
    rare recall with it.

    Here the boost is not reduced, it is REDIRECTED: the same log(pi_c(u)/pi_c) mass is spent on the
    top-m frames by within-unit rank instead of spread over all of them. Recall should hold (true
    positives rank high within their own video) while false positives on normal frames collapse, i.e.
    precision at fixed recall -- lifting the curve rather than sliding along it."""
    n_u = np.array([len(fr) for fr in idx], dtype=float)
    m = Q * r[None, :] * n_u[:, None]                      # [nU, K] expected positives per unit
    R = rank_scores(P, idx)                                # [N, K] within-unit ranks
    scale = np.maximum(1.0, tau * m)
    return _sigmoid((m[inv] - R) / scale[inv])


def framewise_offsets(P, idx, inv, Q, r, pi, alpha, lam, valid, tau):
    """Per-FRAME offsets: budgeted boosts, unmodulated suppression. When a unit looks negative the
    offset is negative and must apply to every frame of it, so only the positive side is gated."""
    Du = hpc_offsets(Q, r, pi, alpha, lam, valid)[inv]
    G = budget_gate(P, idx, inv, Q, r, tau)
    return np.where(Du > 0, Du * G, Du)


def hpc_offsets(Q, r, pi, alpha, lam, valid=None, lam_neg=None):
    """Patient-conditional offsets: one row per video.
        pi_c(v)   = q_c(v) r_c + (1 - q_c(v)) alpha pi_c
        delta_c(v)= lam log( pi_c(v) / pi_c )

    `valid` masks classes that have NO val support. Without it such a class has pi_c = 0 and r_c
    floored at 1e-6, and the ratio hands the class a large positive offset -- the biggest boost in
    the vector goes to the one class we have never observed. With no evidence the honest action is
    no adjustment, so those classes fall back to the base decoder (delta = 0)."""
    pv = Q * r[None, :] + (1.0 - Q) * alpha * pi[None, :]
    lam_v = np.asarray(lam, dtype=float)          # scalar, or a per-class reliability weight
    d = lam_v * (np.log(np.clip(pv, 1e-12, None)) - np.log(np.clip(pi, 1e-12, None))[None, :])
    if lam_neg is not None:
        # ASYMMETRIC CONDITIONING. The two halves carry different risk:
        #   boost    (d>0, unit looks positive) -- ADDS predictions, so it creates false positives on
        #            the dominant class, which is what MCC punishes hardest.
        #   suppress (d<0, unit looks negative) -- REMOVES predictions. For a rare class most units
        #            genuinely lack it, so this deletes almost pure false positives and costs no
        #            recall, because there was no true positive there to lose.
        # Three previous attempts (pact's global shrink, hcg_rel's reliability scaling, hceb's
        # within-unit budget) all modulated the BOOST and so gave back recall by construction. This
        # leaves the boost alone and turns up the safe half instead -- precision without a recall
        # cost, which is the only move that lifts the curve rather than sliding along it.
        d = np.where(d < 0, d * (np.asarray(lam_neg, dtype=float) / np.maximum(lam_v, 1e-9)), d)
    if valid is not None:
        d[:, ~valid] = 0.0
    return d


def decode(logP, inv, Delta):
    """argmax_c [ log p_c(x) + delta_c(video(x)) ]."""
    return (logP + Delta[inv]).argmax(1)


def run(ds, ensemble):
    prefix, ddir = prefix_ddir(ds)
    z0 = np.load(os.path.join(ddir, f"{prefix}_f0_seed0.npz"), allow_pickle=True)
    K = int(max(z0["vY"].max(), z0["tY"].max())) + 1
    D.CLASSES = [str(i) for i in range(K)]
    methods = (["argmax", "la_tau", "ot_alpha", "mean_temporal", "ega_b4",
                "hcg", "hcg_const", "hcg_hard", "hcg_video", "hcg_rel", "hcg_rel_vid",
                "hce", "hce_auto", "pact", "pact_uncon",
                "hceb", "hceb_t15", "hceb_t60", "condb",
                "hcea", "hcea_s2", "hcea_nosup", "conda"]
               + [f"hcg_L{'vid' if L is None else L}" for L in GRANS])
    YT = []
    TG = []                 # per-fold TEST video/group ids, for cluster (patient-level) bootstrap
    acc = {m: [] for m in methods}
    picks = []
    for fold, seed in C._runs(ensemble):
        vL, vY, tL, tY, vg, tg, vf, tf = load(ds, prefix, ddir, fold, seed, ensemble)
        prior = np.bincount(vY, minlength=K).astype(float); prior /= prior.sum()
        lp = np.log(prior + 1e-12)
        vlog, tlog = D.log_softmax(vL), D.log_softmax(tL)
        YT.append(tY)
        TG.append(tg)
        acc["argmax"].append(tL.argmax(1))
        acc["ot_alpha"].append(D.fit_ot_alpha(vL, vlog, vY, prior, lp, D.f1m)(tL, tlog))
        bt, btf = 0.0, D.f1m(vY, vlog.argmax(1))
        for tau in np.linspace(0.0, 2.0, 11):
            f = D.f1m(vY, (vlog - tau * lp[None, :]).argmax(1))
            if f > btf:
                btf, bt = f, tau
        acc["la_tau"].append((tlog - bt * lp[None, :]).argmax(1))

        vbaseL = D.temporal_smooth(vL, vg, vf, W_TEMP)
        tbaseL = D.temporal_smooth(tL, tg, tf, W_TEMP)
        acc["mean_temporal"].append(tbaseL.argmax(1))

        # SPLIT AUDIT. Val-based selection has misfired all through this project, and the val scores
        # are implausibly high (Kvasir ~0.99 val vs ~0.25 test) for a baseline that fits NOTHING.
        # If val shares videos with train it is memorised and cannot stand in for test, which would
        # invalidate every hyperparameter choice regardless of how the criterion is cross-fitted.
        # mean_temporal fits no parameters, so a large val-vs-test gap here is a property of the
        # SPLIT, not of any decoder.
        ov = len(set(np.unique(vg).tolist()) & set(np.unique(tg).tolist()))
        print(f"    SPLIT AUDIT: |val videos|={len(np.unique(vg))} |test videos|={len(np.unique(tg))} "
              f"overlap={ov} | mean_temporal valF1={D.f1m(vY, vbaseL.argmax(1)):.4f} "
              f"testF1={D.f1m(tY, tbaseL.argmax(1)):.4f} | argmax valF1="
              f"{D.f1m(vY, vlog.argmax(1)):.4f} testF1={D.f1m(tY, tlog.argmax(1)):.4f}", flush=True)

        # our current best decoder, for reference
        bv = E.beta_vector(prior, 4.0, "tail")
        bT, bf = 1.0, -9.0
        for T in TEMPS:
            vP_ = D._softmax(vbaseL / T)
            st_ = E.ega_fit(vP_, vbaseL.argmax(1), vY, K, beta=bv)
            f = D.f1m(vY, E.ega_apply(vP_, vbaseL.argmax(1), st_))
            if f > bf:
                bf, bT = f, T
        st_ = E.ega_fit(D._softmax(vbaseL / bT), vbaseL.argmax(1), vY, K, beta=bv)
        acc["ega_b4"].append(E.ega_apply(D._softmax(tbaseL / bT), tbaseL.argmax(1), st_))

        # ---- HCG: sweep the conditioning granularity ------------------------------------
        # Selection uses CROSS-FITTED presence posteriors (split by video), so the criterion is
        # out-of-sample. The model used at test time is refitted on all val units.
        diag = []
        per_gran, per_gran_rel = {}, {}
        best = (-9.0, None); best_rel = (-9.0, None); rel_vid = None
        t_stage = {"units": 0.0, "presence": 0.0, "cv": 0.0, "sweep": 0.0}
        for L in GRANS:
            t0 = time.time()
            vinv, vidx = unit_index(vg, vf, L)
            uvid = unit_videos(vg, vidx)
            Zv = presence_labels(vY, vidx, K)
            r = within_rate(vY, vidx, Zv, K, prior)
            valid = (prior > 0) & (Zv.sum(0) > 0)
            t_stage["units"] += time.time() - t0
            prate = Zv.mean(0)
            gbest = (-9.0, None); gauc = np.nan
            rbest = (-9.0, None); rrel = None
            for T in TEMPS:
                vP = D._softmax(vbaseL / T)
                vlogP = np.log(np.clip(vP, 1e-12, None))
                for frac in TOPK:
                    t0 = time.time()
                    Sv = topk_aggregate(vP, vidx, frac)
                    mu, sd, pooled, per = fit_presence(Sv, Zv, K)
                    t_stage["presence"] += time.time() - t0
                    t0 = time.time()
                    Qcv = crossfit_presence(Sv, Zv, uvid, K)
                    t_stage["cv"] += time.time() - t0
                    aucs = [roc_auc(Qcv[:, c], Zv[:, c]) for c in range(K) if valid[c]]
                    a_ = float(np.nanmean(aucs)) if len(aucs) else np.nan
                    rel = reliability(Qcv, Zv, valid, K)
                    t0 = time.time()
                    for alpha in ALPHAS:
                        for lam in LAMS:
                            f = fast_f1(vY, decode(vlogP, vinv,
                                        hpc_offsets(Qcv, r, prior, alpha, lam, valid)), K)
                            if f > gbest[0]:
                                gbest = (f, (T, frac, alpha, lam, mu, sd, pooled, per, r, valid))
                                gauc = a_
                            fr_ = fast_f1(vY, decode(vlogP, vinv,
                                          hpc_offsets(Qcv, r, prior, alpha, lam * rel, valid)), K)
                            if fr_ > rbest[0]:
                                rbest = (fr_, (T, frac, alpha, lam * rel, mu, sd, pooled, per, r, valid))
                                rrel = rel
                    t_stage["sweep"] += time.time() - t0
            per_gran[L] = gbest
            per_gran_rel[L] = rbest
            diag.append((L, len(vidx), float(prate[valid].mean()) if valid.any() else 0.0, gauc))
            if L is None and rrel is not None:
                rel_vid = rrel
            if gbest[0] > best[0]:
                best = (gbest[0], L)
            if rbest[0] > best_rel[0]:
                best_rel = (rbest[0], L)

        def _apply(L, cfg, hard=False, const=False):
            """Decode test frames at granularity L. Test units are built only from the frames of the
            video being decoded, so this stays a single-patient operation."""
            T, frac, alpha, lam, mu, sd, pooled, per, r, valid = cfg
            tinv_, tidx_ = unit_index(tg, tf, L)
            tP_ = D._softmax(tbaseL / T)
            tlogP_ = np.log(np.clip(tP_, 1e-12, None))
            Q = presence_prob(topk_aggregate(tP_, tidx_, frac), mu, sd, pooled, per, K)
            if hard:
                Q = (Q > 0.5).astype(float)
            if const:
                vinv_, vidx_ = unit_index(vg, vf, L)
                vP_ = D._softmax(vbaseL / T)
                qb = presence_prob(topk_aggregate(vP_, vidx_, frac), mu, sd, pooled, per, K).mean(0)
                Q = np.tile(qb[None, :], (len(tidx_), 1))
            return decode(tlogP_, tinv_, hpc_offsets(Q, r, prior, alpha, lam, valid))

        Lb = best[1]
        cfg = per_gran[Lb][1]
        acc["hcg"].append(_apply(Lb, cfg))
        acc["hcg_hard"].append(_apply(Lb, cfg, hard=True))
        acc["hcg_const"].append(_apply(Lb, cfg, const=True))
        acc["hcg_video"].append(_apply(None, per_gran[None][1]))     # = HPC, the fixed-video variant
        acc["hcg_rel"].append(_apply(best_rel[1], per_gran_rel[best_rel[1]][1]))
        acc["hcg_rel_vid"].append(_apply(None, per_gran_rel[None][1]))

        # ---- HCE: compose the patient-conditional prior shift with EGA's global arbitration ----
        # hcg_rel and ega_b4 win on different datasets and act on DIFFERENT axes: EGA is a global
        # plug-in arbitration over the confusion matrix, conditioning is a per-patient prior shift.
        # The gap in hcg_rel is its fallback: when reliability drives lam_c -> 0 the offset vanishes
        # and it degrades to temporal argmax, the weakest option. Composing makes it degrade to EGA
        # instead, which beats temporal on BOTH datasets.
        Lr = best_rel[1]
        Tr, fr_, ar_, lamr, mur, sdr, por, per_r, rr_, valr = per_gran_rel[Lr][1]
        vinv_r, vidx_r = unit_index(vg, vf, Lr)
        vP_r = D._softmax(vbaseL / Tr)
        vlogP_r = np.log(np.clip(vP_r, 1e-12, None))
        Qv_r = presence_prob(topk_aggregate(vP_r, vidx_r, fr_), mur, sdr, por, per_r, K)
        vadj = vlogP_r + hpc_offsets(Qv_r, rr_, prior, ar_, lamr, valr)[vinv_r]

        tinv_r, tidx_r = unit_index(tg, tf, Lr)
        tP_r = D._softmax(tbaseL / Tr)
        tlogP_r = np.log(np.clip(tP_r, 1e-12, None))
        Qt_r = presence_prob(topk_aggregate(tP_r, tidx_r, fr_), mur, sdr, por, per_r, K)
        tadj = tlogP_r + hpc_offsets(Qt_r, rr_, prior, ar_, lamr, valr)[tinv_r]

        # budgeted variants: same conditioning, boost redirected to the top-m frames per unit
        vidxr = unit_index(vg, vf, Lr)[1]
        for nm, tau in (("hceb", 0.3), ("hceb_t15", 0.15), ("hceb_t60", 0.6)):
            vfo = framewise_offsets(vP_r, vidxr, vinv_r, Qv_r, rr_, prior, ar_, lamr, valr, tau)
            tfo = framewise_offsets(tP_r, tidx_r, tinv_r, Qt_r, rr_, prior, ar_, lamr, valr, tau)
            va, ta = vlogP_r + vfo, tlogP_r + tfo
            bvb = E.beta_vector(prior, 4.0, "tail")
            stb = E.ega_fit(D._softmax(va), va.argmax(1), vY, K, beta=bvb)
            acc[nm].append(E.ega_apply(D._softmax(ta), ta.argmax(1), stb))
            if nm == "hceb":
                acc["condb"].append(ta.argmax(1))   # budgeted conditioning WITHOUT the EGA stage

        # ---- HCEA: asymmetric conditioning, suppression forced ON --------------------------
        # alpha is what enables suppression (1.0 = none, small = strong) and it was being chosen on
        # the memorised val: Kvasir-single picked 0.02 and the method worked; GALAR picked 1.0 in
        # EVERY fold and the method failed. So fix alpha rather than let a val that scores 0.99 for
        # a parameter-free baseline decide it, and scale the negative side by lam_neg.
        A_SUP = 0.02
        bva = E.beta_vector(prior, 4.0, "tail")
        for nm, lneg in (("hcea", 1.0), ("hcea_s2", 2.0), ("hcea_nosup", None)):
            al = A_SUP if lneg is not None else ar_
            ln = (lamr * lneg) if lneg is not None else None
            vfa = hpc_offsets(Qv_r, rr_, prior, al, lamr, valr, ln)[vinv_r]
            tfa = hpc_offsets(Qt_r, rr_, prior, al, lamr, valr, ln)[tinv_r]
            va2, ta2 = vlogP_r + vfa, tlogP_r + tfa
            sta = E.ega_fit(D._softmax(va2), va2.argmax(1), vY, K, beta=bva)
            acc[nm].append(E.ega_apply(D._softmax(ta2), ta2.argmax(1), sta))
            if nm == "hcea":
                acc["conda"].append(ta2.argmax(1))    # asymmetric conditioning WITHOUT the EGA stage

        bvc = E.beta_vector(prior, 4.0, "tail")
        vPa, tPa = D._softmax(vadj), D._softmax(tadj)
        vba, tba = vadj.argmax(1), tadj.argmax(1)
        st_c = E.ega_fit(vPa, vba, vY, K, beta=bvc)
        acc["hce"].append(E.ega_apply(tPa, tba, st_c))

        # hce_auto: pick among {pure EGA, pure conditioning, composed} on VAL macro-F1. The choice is
        # the applicability criterion doing its job -- we expect EGA where tail reliability is poor
        # (GALAR) and the conditional/composed rule where it is high (Kvasir).
        cand = {
            "ega":  (fast_f1(vY, E.ega_apply(D._softmax(vbaseL / bT), vbaseL.argmax(1), st_), K),
                     E.ega_apply(D._softmax(tbaseL / bT), tbaseL.argmax(1), st_)),
            "cond": (fast_f1(vY, decode(vlogP_r, vinv_r,
                              hpc_offsets(Qv_r, rr_, prior, ar_, lamr, valr)), K),
                     _apply(Lr, per_gran_rel[Lr][1])),
            "comp": (fast_f1(vY, E.ega_apply(vPa, vba, st_c), K),
                     E.ega_apply(tPa, tba, st_c)),
        }
        pick = max(cand, key=lambda k: cand[k][0])
        acc["hce_auto"].append(cand[pick][1])

        # ---- PACT: honest (cross-fitted) selection under an MCC constraint --------------------
        # Two defects this repairs. (1) Every val score above is IN-SAMPLE: ega_fit does greedy
        # coordinate ascent on val macro-F1 and is then scored on that same val, so all candidates
        # read ~0.99-1.00 and hce_auto's picks are ties. Here each candidate is fitted on one half of
        # the val VIDEOS and scored on the other, so the comparison is out-of-sample and an MCC
        # constraint on it is meaningful rather than vacuous. (2) hce buys rare recall partly with
        # precision: MCC fell below mean_temporal in 3 of 4 blocks. So select
        #     argmax CV-val macro-F1   s.t.   CV-val MCC >= CV-val MCC(mean_temporal) - MCC_TOL
        # the same constraint pattern ega_auto used to fix its own MCC dips. If nothing satisfies it,
        # fall back to the highest-MCC candidate rather than the highest-F1 one.
        vv = np.unique(vg)
        half = set(vv[np.random.default_rng(0).permutation(len(vv))[: len(vv) // 2]].tolist())
        hmask = np.array([g in half for g in vg])
        cfgc = (Lr, Tr, fr_, ar_, lamr, bT)
        cands = [("ega", 1.0)] + [(m_, sc) for m_ in ("cond", "comp") for sc in SCALES]

        def _cv(mode, scale):
            pr = np.empty(len(vY), dtype=np.int64)
            for hm in (hmask, ~hmask):
                fitm = ~hm
                if fitm.sum() == 0 or hm.sum() == 0:
                    return -9.0, -9.0
                pr[hm] = strategy_predict(mode, scale, cfgc, bvc, K,
                                          vbaseL[fitm], vY[fitm], vg[fitm], vf[fitm],
                                          vbaseL[hm], vg[hm], vf[hm])
            return fast_f1(vY, pr, K), D.mccm(vY, pr)

        mcc_floor = D.mccm(vY, vbaseL.argmax(1)) - MCC_TOL      # temporal baseline, nothing fitted
        scored = {c: _cv(*c) for c in cands}
        ok = [c for c in cands if scored[c][1] >= mcc_floor]
        sel = (max(ok, key=lambda c: scored[c][0]) if ok
               else max(cands, key=lambda c: scored[c][1]))
        unc = max(cands, key=lambda c: scored[c][0])
        for nm, ch in (("pact", sel), ("pact_uncon", unc)):
            acc[nm].append(strategy_predict(ch[0], ch[1], cfgc, bvc, K,
                                            vbaseL, vY, vg, vf, tbaseL, tg, tf))
        print(f"    PACT sel={sel} uncon={unc} mcc_floor={mcc_floor:.4f} | "
              + " ".join(f"{c[0]}{c[1]:g}=({scored[c][0]:.4f},{scored[c][1]:.4f})" for c in cands),
              flush=True)
        print(f"    hce_auto picked '{pick}'  valF1: "
              + " ".join(f"{k}={cand[k][0]:.4f}" for k in cand), flush=True)
        for L in GRANS:
            acc[f"hcg_L{'vid' if L is None else L}"].append(_apply(L, per_gran[L][1]))
        T, frac, alpha, lam = cfg[0], cfg[1], cfg[2], cfg[3]
        picks.append((Lb, T, frac, alpha, lam))
        print("    presence diag (L, n_units, rate, HELD-OUT AUC): "
              + " | ".join(f"{'vid' if L is None else L}:{n},{m:.2f},auc={a:.3f}"
                           for L, n, m, a in diag), flush=True)
        if rel_vid is not None:
            ordr = np.argsort(prior)
            print("    per-class presence reliability at VIDEO level (rarest first): "
                  + " ".join(f"c{c}(pi={prior[c]:.4f}):{rel_vid[c]:.2f}" for c in ordr), flush=True)
        print("    timing(s): " + " ".join(f"{k}={v:.1f}" for k, v in t_stage.items()), flush=True)
        print(f"  fold{fold} seed{seed}: L*={'vid' if Lb is None else Lb} T={T} topk={frac} "
              f"alpha={alpha} lam={lam}  valF1={best[0]:.4f}", flush=True)

    lab = list(range(K))
    support = np.bincount(np.concatenate(YT), minlength=K).astype(float)
    prev = support / max(support.sum(), 1)
    rare = sorted(np.argsort(prev)[:max(1, K // 3)].tolist())

    # companion dump (hce/pact predictions + labels/video-groups per fold) so a downstream script
    # can bootstrap-compare two DIFFERENT configs (e.g. cbsampler vs focal) head-to-head on the
    # same videos -- results_hcg.csv only has aggregated means, this is what's needed instead.
    cmp_dir = "/pvc/results/experimental"
    os.makedirs(cmp_dir, exist_ok=True)
    tag_ = "ens" if ensemble else "single"
    np.savez(os.path.join(cmp_dir, f"cmp_preds_{ds}_{tag_}.npz"),
              YT=np.array(YT, dtype=object), TG=np.array(TG, dtype=object),
              hce=np.array(acc.get("hce", []), dtype=object),
              pact=np.array(acc.get("pact", []), dtype=object),
              K=K, lab=np.array(lab), rare=np.array(rare))

    macro, rareR, rareP, rareF = {}, {}, {}, {}
    for m in methods:
        f1s, recs, mccs, rr, rp, rf = [], [], [], [], [], []
        for i, pr in enumerate(acc[m]):
            f1s.append(D.f1m(YT[i], pr)); recs.append(D.recm(YT[i], pr)); mccs.append(D.mccm(YT[i], pr))
            P_, R_, F_, _ = precision_recall_fscore_support(YT[i], pr, labels=lab, zero_division=0)
            rr.append(R_[rare].mean()); rp.append(P_[rare].mean()); rf.append(F_[rare].mean())
        macro[m] = (np.mean(f1s), np.std(f1s), np.mean(recs), np.std(recs), np.mean(mccs), np.std(mccs))
        rareR[m] = (np.mean(rr), np.std(rr))
        rareP[m] = (np.mean(rp), np.std(rp))
        rareF[m] = (np.mean(rf), np.std(rf))

    # pooled (summed over all fold/seed runs) confusion matrix per headline method, for the
    # qualitative rescue figure (which classes argmax/OT/LA confuse vs. which HCE/PACT fix).
    CM_METHODS = ["argmax", "la_tau", "ot_alpha", "mean_temporal", "ega_b4", "hce", "pact"]
    cms = {}
    for m in CM_METHODS:
        if m not in acc:
            continue
        cm = np.zeros((K, K), dtype=np.int64)
        for i, pr in enumerate(acc[m]):
            cm += np.bincount(YT[i] * K + np.asarray(pr), minlength=K * K).reshape(K, K)
        cms[m] = cm
    np.savez(os.path.join("/pvc/results/experimental", f"confmat_{ds}_{tag_}.npz"),
              **{m: cm for m, cm in cms.items()}, K=K, lab=np.array(lab), rare=np.array(rare))

    tag = "ens" if ensemble else "single"
    os.makedirs("/pvc/results/experimental", exist_ok=True)

    # Per-class P/R/F1, SAME schema as per_class_report.py's summary_{dataset}.csv (class column is
    # the numeric index here, not resolved to disease names -- join with the name list downstream,
    # same fallback the per_class_report.py convention already uses for datasets without an easy
    # class-name source).
    pc_outp = f"/pvc/results/experimental/per_class_{ds}_{tag}_hcg.csv"
    with open(pc_outp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "method", "class", "support", "prevalence", "rare",
                    "P_mean", "R_mean", "R_sd", "F1_mean", "F1_sd"])
        for m in methods:
            Ps = np.stack([precision_recall_fscore_support(YT[i], pr, labels=lab, zero_division=0)[0]
                           for i, pr in enumerate(acc[m])])
            Rs = np.stack([precision_recall_fscore_support(YT[i], pr, labels=lab, zero_division=0)[1]
                           for i, pr in enumerate(acc[m])])
            Fs = np.stack([precision_recall_fscore_support(YT[i], pr, labels=lab, zero_division=0)[2]
                           for i, pr in enumerate(acc[m])])
            Pm, Rm, Rsd, Fm, Fsd = Ps.mean(0), Rs.mean(0), Rs.std(0), Fs.mean(0), Fs.std(0)
            for i in range(K):
                w.writerow([ds, m, i, int(support[i]), f"{prev[i]:.6f}", int(i in rare),
                            f"{Pm[i]:.4f}", f"{Rm[i]:.4f}", f"{Rsd[i]:.4f}", f"{Fm[i]:.4f}", f"{Fsd[i]:.4f}"])

    # kvasir_cbsampler writes to its own file: it may run concurrently with a kvasir/galar_pathology
    # job, and append-mode writes from two separate pods to the same CephFS-backed file are not
    # guaranteed atomic across clients, so give it a private output instead of racing the shared one.
    suffix = f"_{ds}" if ds not in ("kvasir", "galar_pathology") else ""
    outp = f"/pvc/results/experimental/results_hcg{suffix}.csv"
    hdr = not os.path.exists(outp)
    with open(outp, "a", newline="") as f:
        w = csv.writer(f)
        if hdr:
            w.writerow(["dataset", "scope", "method", "f1_mean", "f1_sd", "recall_mean", "recall_sd",
                        "mcc_mean", "mcc_sd", "rareRec_mean", "rareRec_sd", "rarePrec_mean", "rarePrec_sd",
                        "rareF1_mean", "rareF1_sd"])
        for m in methods:
            a, b, c, csd, d, e = macro[m]; rrm, rrsd = rareR[m]; rpm, rpsd = rareP[m]; rfm, rfsd = rareF[m]
            w.writerow([ds, tag, m, f"{a:.4f}", f"{b:.4f}", f"{c:.4f}", f"{csd:.4f}", f"{d:.4f}", f"{e:.4f}",
                        f"{rrm:.4f}", f"{rrsd:.4f}", f"{rpm:.4f}", f"{rpsd:.4f}",
                        f"{rfm:.4f}", f"{rfsd:.4f}"])
    amf = macro["argmax"][0]; basef = macro["mean_temporal"][0]; amrr = rareR["argmax"][0]
    print(f"=== {ds} [{tag}] K={K} runs={len(YT)} rare={rare} ===", flush=True)
    for m in methods:
        a, b, c, csd, d, e = macro[m]; rrm, rrsd = rareR[m]; rpm, _ = rareP[m]; rfm, _ = rareF[m]
        print(f"  {m:<14} F1 {a:.4f}+-{b:.4f} (dArg {a-amf:+.4f}, dBase {a-basef:+.4f}) "
              f"rec {c:.4f}+-{csd:.4f} mcc {d:.4f} | rareRec {rrm:.4f} (dArg {rrm-amrr:+.4f}) "
              f"rarePrec {rpm:.4f} rareF1 {rfm:.4f}", flush=True)

    # Paired significance on the ACTUAL swept PACT (not a fixed-recipe approximation), vs
    # mean_temporal, over the matched per-run predictions already collected above.
    def _paired(a, b):
        d = np.asarray(a, float) - np.asarray(b, float)
        n = len(d); md = float(d.mean())
        if n < 2 or np.allclose(d, 0):
            return md, float("nan")
        se = d.std(ddof=1) / np.sqrt(n)
        try:
            from scipy import stats as st
            return md, float(2 * st.t.sf(abs(md / max(se, 1e-12)), n - 1))
        except Exception:
            return md, float("nan")

    def _metrics_from_cm(cm, K, rare):
        """F1/MCC/rarePrec/rareRec/rareF1 from ONE confusion matrix, no sklearn calls. Bit-identical
        to D.f1m/D.mccm (verified: fast_f1 above is the same trick, already checked against sklearn
        to machine precision) -- this just also derives MCC and per-class P/R/F1 from the SAME cm
        instead of recomputing the confusion matrix three more times via separate sklearn functions."""
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

    def _boot_pvalue(y, pa, pb, grp, K, lab, rare, nboot=800, seed=0):
        """Per-PATIENT (video) cluster bootstrap for one fold: resample VIDEO IDS with replacement,
        take all frames of the resampled videos, recompute F1/MCC/rareRec/rarePrec/rareF1 for BOTH
        methods on the SAME resample, take the difference. Frames within a video are correlated, so a
        naive per-frame bootstrap overstates independence (understates variance); resampling at the
        video level is the correct unit here and is still far more powered than the 6-fold-level test
        (tens of videos per fold vs. 6 folds). Two-sided p from the fraction of bootstrap diffs
        crossing zero.

        Vectorised via ONE bincount confusion matrix per method per iteration (see _metrics_from_cm),
        not 4 separate sklearn calls (f1_score, matthews_corrcoef, precision_recall_fscore_support x2)
        -- at GALAR's scale (~200k frames/fold) the sklearn version measured 3+ HOURS and was still
        running on GALAR-single alone; this version is the same ~140x speedup fast_f1 already uses
        elsewhere in this file, extended to also derive MCC and per-class P/R/F1 from the same matrix."""
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
        try:
            from scipy import stats as st
            chi2 = -2 * sum(np.log(pvals))
            return float(st.chi2.sf(chi2, 2 * len(pvals)))
        except Exception:
            return float("nan")

    sig_rows = []
    if "pact" in methods:
        print(f"--- pact/hce vs mean_temporal, paired ({len(YT)} runs, FULL swept config) ---", flush=True)
        for pm in ("pact", "pact_uncon", "hce"):
            f1r = [D.f1m(YT[i], p) for i, p in enumerate(acc[pm])]
            mccr = [D.mccm(YT[i], p) for i, p in enumerate(acc[pm])]
            rrr, rpr, rfr = [], [], []
            for i, p in enumerate(acc[pm]):
                P_, R_, F_, _ = precision_recall_fscore_support(YT[i], p, labels=lab, zero_division=0)
                rrr.append(float(R_[rare].mean())); rpr.append(float(P_[rare].mean()))
                rfr.append(float(F_[rare].mean()))
            f1b = [D.f1m(YT[i], p) for i, p in enumerate(acc["mean_temporal"])]
            mccb = [D.mccm(YT[i], p) for i, p in enumerate(acc["mean_temporal"])]
            rrb, rpb, rfb = [], [], []
            for i, p in enumerate(acc["mean_temporal"]):
                P_, R_, F_, _ = precision_recall_fscore_support(YT[i], p, labels=lab, zero_division=0)
                rrb.append(float(R_[rare].mean())); rpb.append(float(P_[rare].mean()))
                rfb.append(float(F_[rare].mean()))
            for nm, x, y in (("F1", f1r, f1b), ("MCC", mccr, mccb), ("rareRec", rrr, rrb),
                             ("rarePrec", rpr, rpb), ("rareF1", rfr, rfb)):
                md, pv = _paired(x, y)
                star = "*" if (pv == pv and pv < 0.05) else " "
                print(f"  {pm:<12} {nm:<8} fold-level d={md:+.4f} p={pv:.4f} {star}", flush=True)
                sig_rows.append([ds, tag, pm, "fold_ttest", nm, f"{md:.4f}", f"{pv:.4f}", "", ""])

            # per-VIDEO cluster bootstrap, one per run, combined across runs via Fisher's method
            per_run_boot = []
            for i, p in enumerate(acc[pm]):
                per_run_boot.append(_boot_pvalue(YT[i], np.asarray(p), np.asarray(acc["mean_temporal"][i]),
                                                 TG[i], K, lab, rare, seed=i))
            for nm in ("F1", "MCC", "rareRec", "rarePrec", "rareF1"):
                ds_ = [r[nm][0] for r in per_run_boot]
                ps_ = [r[nm][1] for r in per_run_boot]
                cilo_ = [r[nm][2] for r in per_run_boot]
                cihi_ = [r[nm][3] for r in per_run_boot]
                combined_p = _fisher_combine(ps_)
                star = "*" if (combined_p == combined_p and combined_p < 0.05) else " "
                per_fold = " | ".join(f"fold{i}:d={r[nm][0]:+.4f},p={r[nm][1]:.4g}"
                                      for i, r in enumerate(per_run_boot))
                print(f"  {pm:<12} {nm:<8} per-video-boot combined_p={combined_p:.4g} {star}  "
                      f"CI=[{np.mean(cilo_):+.4f},{np.mean(cihi_):+.4f}]  [{per_fold}]", flush=True)
                # CI bounds are the across-run MEAN of each run's own 2.5/97.5 percentile bootstrap
                # bound (same simple averaging already used for the point delta above) -- not a
                # single re-pooled bootstrap, so treat as an approximate/illustrative interval.
                sig_rows.append([ds, tag, pm, "video_cluster_bootstrap", nm,
                                 f"{np.mean(ds_):.4f}", f"{combined_p:.4g}",
                                 f"{np.mean(cilo_):.4f}", f"{np.mean(cihi_):.4f}"])

    sig_outp = f"/pvc/results/experimental/results_hcg_significance{suffix}.csv"
    sig_hdr = not os.path.exists(sig_outp)
    with open(sig_outp, "a", newline="") as f:
        w = csv.writer(f)
        if sig_hdr:
            w.writerow(["dataset", "scope", "method", "test", "metric", "delta", "pvalue", "ci_lo", "ci_hi"])
        for row in sig_rows:
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                     choices=["galar_pathology", "kvasir", "kvasir_cbsampler"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--ensemble", action="store_true")
    a = ap.parse_args()
    if a.device == "cuda":
        assert torch is not None and torch.cuda.is_available()
        D._DEVICE = torch.device("cuda")
    run(a.dataset, a.ensemble)


if __name__ == "__main__":
    main()
