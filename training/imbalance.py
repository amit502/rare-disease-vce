"""Imbalance-oriented losses and sampler, shared by train.py (baseline) and
train_consistency.py (method) so both use IDENTICAL recipe code - required for a
fair baseline-vs-method comparison.

Recipes are (loss, sampler) pairs:
  loss    : ce | weighted_ce | focal | la_loss | ldam
  sampler : random | class_balanced
LDAM uses deferred reweighting (DRW): it starts unweighted; the training loop calls
criterion.reweight(class_weights) at the DRW epoch.
"""
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


class LogitAdjust(nn.Module):
    """Menon et al. 2021: add tau*log(pi_c) to logits before CE/argmax (LA-loss)."""
    def __init__(self, class_counts, tau=1.0):
        super().__init__()
        n = float(sum(class_counts))
        lp = torch.tensor([math.log(c / n) for c in class_counts], dtype=torch.float32)
        self.register_buffer("log_prior", lp)
        self.tau = tau

    def forward(self, logits):
        return logits + self.tau * self.log_prior


class FocalLoss(nn.Module):
    """Lin et al. 2017 focal loss over CE: down-weights easy (high-pt) examples."""
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class LDAMLoss(nn.Module):
    """Cao et al. 2019 label-distribution-aware margin. Class-dependent margins
    ~ 1/n_c^{1/4}. Deferred reweighting: start weight=None, call reweight() at DRW."""
    def __init__(self, class_counts, max_m=0.5, s=30.0, weight=None):
        super().__init__()
        m = 1.0 / np.sqrt(np.sqrt(np.asarray(class_counts, dtype=np.float64)))
        m = m * (max_m / m.max())
        self.register_buffer("m_list", torch.tensor(m, dtype=torch.float32))
        self.s = s
        self.weight = weight

    def reweight(self, weight):
        self.weight = weight

    def forward(self, logits, target):
        idx = torch.zeros_like(logits, dtype=torch.bool)
        idx.scatter_(1, target.view(-1, 1), True)
        batch_m = self.m_list.to(logits.device)[target].view(-1, 1)     # [B,1]
        out = torch.where(idx, logits - batch_m, logits)
        return F.cross_entropy(self.s * out, target, weight=self.weight)


def build_criterion(loss_name, class_counts, class_weights, focal_gamma, device):
    """Returns (criterion, logit_adjust). logit_adjust is None except for la_loss.
    For 'ldam' the criterion starts UNWEIGHTED (apply DRW via criterion.reweight)."""
    if loss_name == "ce":
        return nn.CrossEntropyLoss(), None
    if loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights.to(device)), None
    if loss_name == "focal":
        return FocalLoss(gamma=focal_gamma).to(device), None
    if loss_name == "la_loss":
        return nn.CrossEntropyLoss(), LogitAdjust(class_counts).to(device)
    if loss_name == "ldam":
        return LDAMLoss(class_counts).to(device), None
    raise ValueError(f"unknown loss {loss_name!r}")


def build_sampler(sampler_name, labels, seed=0):
    """labels: per-sample int class, aligned to the dataset's __getitem__ order.
    Returns a WeightedRandomSampler drawing classes ~uniformly (with a seeded
    generator for reproducibility), or None for random."""
    if sampler_name != "class_balanced":
        return None
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[labels]
    g = torch.Generator()
    g.manual_seed(int(seed) + 12345)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(labels), replacement=True, generator=g)


def seed_worker(worker_id):
    """DataLoader worker_init_fn: re-seed numpy/random per worker from the worker's
    torch seed (which PyTorch derives deterministically from the loader generator),
    so augmentations are reproducible across runs of the same seed."""
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)
