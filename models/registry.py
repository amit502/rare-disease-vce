"""
Model registry.

All baselines use a unified training recipe for fair comparison:
  AdamW, lr=1e-4, weight_decay=0.01, cosine scheduler, 50 epochs, batch_size=32,
  weighted cross-entropy loss (inverse-frequency weights).

AdamW + cosine works well for both CNN and transformer architectures and avoids
attributing performance differences to optimizer choice. ViT-L uses batch_size=16
due to GPU memory, not as a hyperparameter choice.

Original paper settings are preserved as comments for reference.
"""
import os
import sys
import torch
import torch.nn as nn
import timm

# Unified training recipe — same for all models
_UNIFIED = {
    "optimizer":    "adamw",
    "lr":           1e-4,
    "weight_decay": 0.01,
    "scheduler":    "cosine",
    "epochs":       100,
    "batch_size":   32,
    "loss":         "weighted_ce",
}

MODEL_HPARAMS = {
    "densenet161": {
        # Original paper: SGD lr=0.001, momentum=0.9, plateau, up to 2000 epochs
        **_UNIFIED,
    },
    "efficientnet_b2": {
        **_UNIFIED,
    },
    "efficientnet_b4": {
        **_UNIFIED,
    },
    "focalconvnet": {
        # Original paper: SGD lr=0.001, momentum=0.9, plateau, up to 2000 epochs
        **_UNIFIED,
    },
    # ImageNet-pretrained ViTs — fair comparison baseline (patch16, supervised ImageNet-21k)
    "vit_small": {
        **_UNIFIED,
    },
    "vit_large": {
        # batch_size=16 is a GPU memory constraint, not a hyperparameter choice
        **_UNIFIED,
        "batch_size": 16,
    },
    # LVD-142M pretrained DINOv2 — kept as separate strong upper-bound baselines
    "dinov2_vitl": {
        **_UNIFIED,
        "batch_size": 16,
    },
    "dinov2_vits": {
        **_UNIFIED,
    },
    "densenet201_swint": {
        # Original paper: AdamW lr=1e-4, cosine, 100 epochs
        **_UNIFIED,
    },
    "densenet_swint_attn": {
        # Lesion-attentive dual stream: DenseNet201+Swin-T + attention-pooled
        # local pathway. Same recipe as densenet201_swint for a clean comparison.
        **_UNIFIED,
    },
    "densenet_swint_devattn": {
        # Deviation-guided attention: attention biased toward within-frame
        # anomalies (lesions), fixing the plain-attention failure. Same recipe.
        **_UNIFIED,
    },
    "densenet_swint_boxattn": {
        # Box-supervised, deviation-guided, positionless attention pooling added
        # to the dual stream. The attention map is supervised by the lesion bbox
        # at train time (fixes plain attention's drift onto Normal), stays
        # content-based (1x1 conv, no position) so it follows the lesion wherever
        # it appears, and needs no box at inference. Trained with train_boxattn.py.
        **_UNIFIED,
    },
    "densenet_swint_mixstyle": {
        # Cross-patient dual stream: densenet201_swint + MixStyle (patient-style
        # mixing) in early DenseNet layers. Targets cross-patient generalization.
        **_UNIFIED,
    },
    "resnet152": {
        **_UNIFIED,
    },
    "densenet161_spectral": {
        # Same recipe as densenet161 baseline — only architecture changes
        **_UNIFIED,
    },
    "densenet161_spectral_frozen": {
        # Frozen FFT centroid branch — same training recipe, no spectral params to tune
        **_UNIFIED,
    },
    "densenet161_rarity_rotation": {
        # DenseNet161 + RoPE-inspired rarity-rotated prototype head
        **_UNIFIED,
    },
}


def build_model(name: str, num_classes: int, focalconvnet_dir: str = None,
                class_counts: list = None) -> nn.Module:
    """
    Build a model by name, returning an nn.Module with the correct output size.

    focalconvnet_dir: path to the cloned FocalConvNet repo
                      (required when name == "focalconvnet")
    """
    if name == "densenet161":
        model = timm.create_model(
            "densenet161", pretrained=True, num_classes=num_classes
        )

    elif name == "efficientnet_b2":
        model = timm.create_model(
            "efficientnet_b2", pretrained=True, num_classes=num_classes
        )

    elif name == "efficientnet_b4":
        model = timm.create_model(
            "efficientnet_b4", pretrained=True, num_classes=num_classes
        )

    elif name == "focalconvnet":
        model = _load_focalconvnet(num_classes, focalconvnet_dir)

    elif name == "vit_small":
        model = timm.create_model(
            "vit_small_patch16_224.augreg_in21k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        )

    elif name == "vit_large":
        model = timm.create_model(
            "vit_large_patch16_224.augreg_in21k_ft_in1k",
            pretrained=True,
            num_classes=num_classes,
        )

    elif name == "dinov2_vitl":
        model = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=num_classes,
            img_size=224,
        )

    elif name == "dinov2_vits":
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=num_classes,
            img_size=224,
        )

    elif name == "densenet201_swint":
        model = _build_densenet201_swint(num_classes)

    elif name == "densenet_swint_attn":
        model = _build_densenet_swint_attn(num_classes)

    elif name == "densenet_swint_devattn":
        model = _build_densenet_swint_devattn(num_classes)
    elif name == "densenet_swint_boxattn":
        model = _build_densenet_swint_boxattn(num_classes)

    elif name == "densenet_swint_mixstyle":
        model = _build_densenet_swint_mixstyle(num_classes)

    elif name == "resnet152":
        model = timm.create_model("resnet152", pretrained=True, num_classes=num_classes)

    elif name == "densenet161_spectral":
        from .dual_domain import DualDomainNet
        model = DualDomainNet(num_classes=num_classes)

    elif name == "densenet161_spectral_frozen":
        from .frozen_fft import DenseNetFrozenFFT
        model = DenseNetFrozenFFT(num_classes=num_classes)

    elif name == "densenet161_rarity_rotation":
        if class_counts is None:
            raise ValueError("densenet161_rarity_rotation requires class_counts")
        from .rarity_rotation import DenseNetRarityRotation
        model = DenseNetRarityRotation(num_classes=num_classes, class_counts=class_counts)

    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Available: {list(MODEL_HPARAMS.keys())}"
        )

    return model


def _build_densenet201_swint(num_classes: int) -> nn.Module:
    """
    Dual-branch hybrid from Subedi et al. 2024 (arXiv:2408.10733).
    DenseNet201 (local features) + Swin-T (global features) → concat → classifier.
    """
    class DenseNet201SwinT(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.densenet = timm.create_model("densenet201", pretrained=True, num_classes=0)
            self.swint    = timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
            )
            densenet_dim = self.densenet.num_features  # 1920
            swint_dim    = self.swint.num_features     # 768
            self.fusion = nn.Sequential(
                nn.Linear(densenet_dim + swint_dim, 256),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
            )
            self.classifier = nn.Linear(256, num_classes)

        def forward_features(self, x):
            """Pooled 256-dim fused representation, before the final classifier.
            Used for feature extraction (e.g. retrieval) on a fine-tuned model."""
            d = self.densenet(x)
            s = self.swint(x)
            return self.fusion(torch.cat([d, s], dim=1))

        def forward(self, x):
            return self.classifier(self.forward_features(x))

    return DenseNet201SwinT(num_classes)


class _MultiHeadAttentionPool(nn.Module):
    """
    Multi-head spatial attention pooling. Replaces global average pooling on a
    CNN feature map so small, subtle lesions (rare vascular/erosion classes) are
    preserved instead of averaged into the background.

    K learned attention maps each softmax over the H*W positions; each head pools
    the feature map at its salient region → [B, K, C]; heads are concatenated and
    projected to out_dim.
    """
    def __init__(self, in_ch: int, n_heads: int, out_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.Conv2d(in_ch, n_heads, kernel_size=1)
        self.proj = nn.Linear(in_ch * n_heads, out_dim)

    def forward(self, feat):                       # feat: [B, C, H, W]
        B, C, H, W = feat.shape
        a = self.attn(feat).view(B, self.n_heads, H * W)
        a = torch.softmax(a, dim=2)                # attention over spatial positions
        f = feat.view(B, C, H * W)
        pooled = torch.einsum("bkn,bcn->bkc", a, f)  # [B, K, C]
        pooled = pooled.reshape(B, self.n_heads * C)
        return self.proj(pooled)                   # [B, out_dim]


def _build_densenet_swint_attn(num_classes: int) -> nn.Module:
    """
    Lesion-attentive dual stream (extends densenet201_swint).

    Streams:
      - Swin-T          : global context (structural classes)     -> 768
      - DenseNet201 GAP : global CNN features (kept, no loss)      -> 1920
      - DenseNet201 attn: multi-head attention pooling of the SAME
                          DenseNet spatial map, preserving small
                          lesions global pooling would average away -> 512
    Fusion: concat(all three) -> MLP -> classifier.

    The attention pathway is strictly ADDED to the proven dual stream, so it can
    only help: the model learns to route small-lesion evidence through attention
    while keeping the global streams intact for structural classes and MCC.
    """
    class DenseNetSwinTAttn(nn.Module):
        def __init__(self, num_classes, n_heads=4, attn_dim=512):
            super().__init__()
            self.densenet = timm.create_model("densenet201", pretrained=True, num_classes=0)
            self.swint    = timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
            )
            d_dim = self.densenet.num_features   # 1920
            s_dim = self.swint.num_features       # 768
            self.attn_pool = _MultiHeadAttentionPool(d_dim, n_heads, attn_dim)
            self.fusion = nn.Sequential(
                nn.Linear(d_dim + s_dim + attn_dim, 256),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
            )
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            feat = torch.relu(self.densenet.forward_features(x))  # [B, 1920, H, W]
            d_global = feat.mean(dim=(2, 3))                       # GAP  -> [B, 1920]
            d_attn   = self.attn_pool(feat)                        # attn -> [B, 512]
            s = self.swint(x)                                      # global -> [B, 768]
            fused = self.fusion(torch.cat([s, d_global, d_attn], dim=1))
            return self.classifier(fused)

    return DenseNetSwinTAttn(num_classes)


class _DeviationGuidedAttentionPool(nn.Module):
    """
    Attention pooling biased toward within-frame DEVIATIONS (candidate lesions).

    Learning from the failure of plain learned attention: trained end-to-end on
    imbalanced labels, learned attention drifts onto Normal regions (the majority
    gradient), so small lesions are still lost. Fix: add a per-position deviation
    prior = distance of each position from the frame's own spatial-mean (background)
    feature. A lesion is a local region that deviates from the mostly-normal
    background of its frame; normal mucosa does not deviate. This prior is a
    WITHIN-IMAGE geometric signal that the majority class cannot dominate, so it
    pulls attention onto lesion-like regions regardless of class imbalance.

    attention = softmax(learned_attn + beta * deviation), beta learnable.
    """
    def __init__(self, in_ch: int, n_heads: int, out_dim: int, beta_init: float = 2.0):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.Conv2d(in_ch, n_heads, kernel_size=1)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.proj = nn.Linear(in_ch * n_heads, out_dim)

    def forward(self, feat):                        # feat: [B, C, H, W]
        B, C, H, W = feat.shape
        f = feat.view(B, C, H * W)
        bg  = f.mean(dim=2, keepdim=True)           # frame background [B, C, 1]
        dev = (f - bg).norm(dim=1)                  # per-position deviation [B, HW]
        dev = dev / (dev.mean(dim=1, keepdim=True) + 1e-6)   # scale-normalize
        learned = self.attn(feat).view(B, self.n_heads, H * W)
        logits  = learned + self.beta * dev.unsqueeze(1)     # deviation prior per head
        a = torch.softmax(logits, dim=2)
        pooled = torch.einsum("bkn,bcn->bkc", a, f).reshape(B, self.n_heads * C)
        return self.proj(pooled)


def _build_densenet_swint_devattn(num_classes: int) -> nn.Module:
    """
    Deviation-guided lesion-attentive dual stream. Same as densenet_swint_attn but
    the DenseNet attention pathway uses _DeviationGuidedAttentionPool, so attention
    is pulled toward within-frame anomalies (lesions) rather than learned purely
    from the majority-dominated labels — the fix for the attention model's failure.
    """
    class DenseNetSwinTDevAttn(nn.Module):
        def __init__(self, num_classes, n_heads=4, attn_dim=512):
            super().__init__()
            self.densenet = timm.create_model("densenet201", pretrained=True, num_classes=0)
            self.swint    = timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
            )
            d_dim = self.densenet.num_features   # 1920
            s_dim = self.swint.num_features       # 768
            self.attn_pool = _DeviationGuidedAttentionPool(d_dim, n_heads, attn_dim)
            self.fusion = nn.Sequential(
                nn.Linear(d_dim + s_dim + attn_dim, 256),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
            )
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            feat = torch.relu(self.densenet.forward_features(x))
            d_global = feat.mean(dim=(2, 3))
            d_attn   = self.attn_pool(feat)
            s = self.swint(x)
            fused = self.fusion(torch.cat([s, d_global, d_attn], dim=1))
            return self.classifier(fused)

    return DenseNetSwinTDevAttn(num_classes)


class _BoxDeviationAttentionPool(nn.Module):
    """
    Positionless multi-head attention pooling with a within-frame deviation prior,
    exposing the attention map so it can be SUPERVISED by the lesion bounding box
    at train time (see training/train_boxattn.py).

    Content-based by construction: the 1x1 conv scores every location from its own
    feature vector alone, with no coordinates or positional encoding, so it can
    only learn "attend to lesion-like features," not "attend to a fixed location."
    A CNN map + 1x1 conv is translation-equivariant, so the attention follows the
    lesion wherever it appears at inference. The deviation prior (distance from the
    frame's mean feature) adds an unsupervised anomaly bias that still works at
    inference where no box is available.
    """
    def __init__(self, in_ch: int, n_heads: int, out_dim: int, beta_init: float = 2.0):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.Conv2d(in_ch, n_heads, kernel_size=1)
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))
        self.proj = nn.Linear(in_ch * n_heads, out_dim)

    def forward(self, feat):                        # feat: [B, C, H, W]
        B, C, H, W = feat.shape
        f = feat.view(B, C, H * W)
        bg  = f.mean(dim=2, keepdim=True)
        dev = (f - bg).norm(dim=1)
        dev = dev / (dev.mean(dim=1, keepdim=True) + 1e-6)
        learned = self.attn(feat).view(B, self.n_heads, H * W)
        logits  = learned + self.beta * dev.unsqueeze(1)
        a = torch.softmax(logits, dim=2)            # [B, n_heads, HW], each sums to 1
        pooled = torch.einsum("bkn,bcn->bkc", a, f).reshape(B, self.n_heads * C)
        pooled = self.proj(pooled)
        attn_map = a.mean(dim=1).view(B, H, W)      # mean over heads, sums to 1 -> [B,H,W]
        return pooled, attn_map


def _build_densenet_swint_boxattn(num_classes: int) -> nn.Module:
    """
    Box-supervised, deviation-guided attention dual stream. The attention pathway
    is ADDED alongside GAP (as in densenet_swint_attn), so structural classes and
    MCC cannot regress; the box supervision (applied in train_boxattn.py via
    forward_attn) pins that attention onto the lesion, fixing the drift-to-Normal
    failure of plain learned attention. No box is needed at inference: forward(x)
    returns logits exactly like the baseline.
    """
    class DenseNetSwinTBoxAttn(nn.Module):
        def __init__(self, num_classes, n_heads=4, attn_dim=512):
            super().__init__()
            self.densenet = timm.create_model("densenet201", pretrained=True, num_classes=0)
            self.swint    = timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
            )
            d_dim = self.densenet.num_features   # 1920
            s_dim = self.swint.num_features       # 768
            self.attn_pool = _BoxDeviationAttentionPool(d_dim, n_heads, attn_dim)
            self.fusion = nn.Sequential(
                nn.Linear(d_dim + s_dim + attn_dim, 256),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
            )
            self.classifier = nn.Linear(256, num_classes)

        def features_attn(self, x):
            """Returns (fused_256d, attn_map [B,H,W]). The fused feature is the
            pre-classifier representation used for optional crop-consistency."""
            feat = torch.relu(self.densenet.forward_features(x))  # [B, 1920, H, W]
            d_global = feat.mean(dim=(2, 3))                       # GAP  -> [B, 1920]
            d_attn, attn_map = self.attn_pool(feat)               # attn -> [B, 512], [B,H,W]
            s = self.swint(x)                                      # -> [B, 768]
            fused = self.fusion(torch.cat([s, d_global, d_attn], dim=1))
            return fused, attn_map

        def forward(self, x):
            fused, _ = self.features_attn(x)
            return self.classifier(fused)

        def forward_attn(self, x):
            """Returns (logits, attn_map [B,H,W]) for box-supervised training."""
            fused, attn_map = self.features_attn(x)
            return self.classifier(fused), attn_map

    return DenseNetSwinTBoxAttn(num_classes)


class _MixStyle(nn.Module):
    """
    MixStyle (Zhou et al. 2021): domain generalization by mixing feature-statistic
    "style" across instances. Per-instance channel mean/std encode appearance
    (here: patient-specific mucosa tone, lighting, capsule response). Mixing a
    sample's style with another patient's during training synthesizes NEW apparent
    patients, forcing patient-invariant content features -> better cross-patient
    transfer for the learnable-tail classes. Identity at eval time.
    """
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):                            # [B, C, H, W]
        if not self.training or torch.rand(1).item() > self.p or x.size(0) < 2:
            return x
        B = x.size(0)
        mu  = x.mean(dim=[2, 3], keepdim=True)
        sig = (x.var(dim=[2, 3], keepdim=True) + self.eps).sqrt()
        x_norm = (x - mu) / sig
        lam  = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu_mix  = lam * mu  + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix


def _build_densenet_swint_mixstyle(num_classes: int) -> nn.Module:
    """
    Cross-patient dual stream: densenet201_swint with MixStyle inserted into the
    DenseNet stream after transition1 (early layers, where patient appearance
    lives). Targets the PROVEN bottleneck — cross-patient generalization — rather
    than lesion visibility (which devattn ruled out). Swin stream unchanged.
    """
    class DenseNetSwinTMixStyle(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            dn = timm.create_model("densenet201", pretrained=True, num_classes=0)
            self.swint = timm.create_model(
                "swin_tiny_patch4_window7_224", pretrained=True, num_classes=0
            )
            # split DenseNet features after transition1 to insert MixStyle
            children = list(dn.features.named_children())
            split = next(i for i, (n, _) in enumerate(children) if n == "transition1") + 1
            self.dn_early = nn.Sequential(*[m for _, m in children[:split]])
            self.dn_late  = nn.Sequential(*[m for _, m in children[split:]])
            self.mixstyle = _MixStyle(p=0.5, alpha=0.1)
            d_dim = dn.num_features   # 1920
            s_dim = self.swint.num_features   # 768
            self.fusion = nn.Sequential(
                nn.Linear(d_dim + s_dim, 256),
                nn.LeakyReLU(0.1),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
            )
            self.classifier = nn.Linear(256, num_classes)

        def forward(self, x):
            f = self.dn_early(x)
            f = self.mixstyle(f)                       # patient-style mixing (train only)
            f = torch.relu(self.dn_late(f))            # [B, 1920, H, W]
            d = f.mean(dim=(2, 3))                      # GAP -> [B, 1920]
            s = self.swint(x)                           # [B, 768]
            return self.classifier(self.fusion(torch.cat([d, s], dim=1)))

    return DenseNetSwinTMixStyle(num_classes)


def _patch_focalconvnet_syntax(repo_dir: str):
    """
    Fix two issues in the upstream focalconv.py before importing:
    1. Missing Mlp import — prepend it at the top of the file.
    2. assert statements with the message on the next line without a backslash
       continuation — a SyntaxError in Python 3.
    """
    import re
    path = os.path.join(repo_dir, "focalconv.py")
    src  = open(path).read()
    changed = False

    # 1. Prepend Mlp import if not already imported
    mlp_imported = any(
        "Mlp" in line and "import" in line
        for line in src.splitlines()
    )
    if not mlp_imported:
        mlp_import = (
            "try:\n"
            "    from timm.layers import Mlp\n"
            "except ImportError:\n"
            "    from timm.models.layers import Mlp\n"
        )
        src = mlp_import + src
        changed = True

    # 2. Guard module-level test/demo code that executes on import.
    #    Look for the start of the test block (bare model instantiation or
    #    pytorch_total_params) and wrap everything from there to EOF.
    m = re.search(
        r'^(?:v\s*=|pytorch_total_params\s*=|print\()',
        src,
        re.MULTILINE,
    )
    if m:
        guard_pos = src.rfind('\n', 0, m.start()) + 1
        head = src[:guard_pos]
        tail_lines = src[guard_pos:].splitlines(keepends=True)
        indented = ['    ' + l if l.strip() else l for l in tail_lines]
        src = head + 'if __name__ == "__main__":\n' + "".join(indented)
        changed = True

    # 3. Fix assert lines whose message is on the next line (no backslash)
    lines = src.splitlines(keepends=True)
    out   = []
    i     = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r'^\s*assert\b', line) and re.search(r',\s*$', line):
            next_stripped = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if next_stripped.startswith(('"', "'")):
                out.append(line.rstrip().rstrip(",") + ", " + next_stripped + "\n")
                i += 2
            else:
                out.append(re.sub(r',\s*$', '\n', line))
                i += 1
            changed = True
        else:
            out.append(line)
            i += 1

    if changed:
        open(path, "w").write("".join(out))
        print(f"[registry] patched focalconv.py ({path})")


def _load_focalconvnet(num_classes: int, repo_dir: str) -> nn.Module:
    """
    Import FocalConvNet from the cloned repo at repo_dir, then patch
    the classifier head to output num_classes.
    The repo's focalconv.py contains the FocalConvNet class.
    """
    if repo_dir is None:
        raise ValueError(
            "--focalconvnet-dir is required when model=focalconvnet. "
            "Clone https://github.com/NoviceMAn-prog/FocalConvNet first."
        )
    repo_dir = os.path.abspath(repo_dir)
    _patch_focalconvnet_syntax(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from focalconv import FocalConvNet  # noqa: E402

    # Architecture config matching the FocalConvNet-Small used in the paper.
    # dim and depth are required keyword-only args with no defaults.
    # The paper trains from random initialisation (no pretrained weights).
    model = FocalConvNet(
        num_classes=num_classes,
        dim=96,
        depth=(2, 2, 6, 2),
    )

    return model
