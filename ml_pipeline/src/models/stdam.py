"""
ST-DAM: Shape-Texture Decoupled Attention Module (Novelty 1).

A lesion is described by two different things: its outline, and the texture
inside it. A single attention map has to serve both. ST-DAM gives each its own
branch -- edges through fixed Sobel filters, texture through a fixed Gabor bank
-- lets them exchange information by cross-attention, and adds the result back
to the backbone's features.

Inserted into Swin-V2-T, which is where this differs from STA-Net (Qiu et al.,
2025), the CNN work it takes the idea from. Three further departures, each
deliberate:

  Fixed filters, not learned. STA-Net learns its shape branch (deformable
    convolution) and its Gabor weights. Here both are fixed, so the branches
    cannot drift into computing the same thing, the module adds no parameters
    for them, and what each branch responds to stays interpretable. The
    `learned` variant makes this an ablation row rather than an assumption.
  Cross-attention fusion, not concatenation. STA-Net concatenates the two maps
    and convolves. Here shape features query texture features, so an edge can
    ask what texture lies along it. The `sum` variant ablates this away.
  Stages 2 and 3, not the last stage. docs/project_breakdown_v3_final.md puts
    ST-DAM on stage 4's 8x8 map, but STA-Net's own analysis (its section 5.2)
    argues the opposite and places STAM at 28x28 and 14x14: by the last stage
    "the original visual forms and textural details are essentially
    obliterated". At 256px Swin-V2-T's stages are 64/32/16/8, so 32x32 and
    16x16 are the equivalent, and that is where these go.

Why the branches use magnitudes. The Sobel branch takes sqrt(gx^2 + gy^2),
which responds to an edge of any orientation, and the Gabor branch pairs each
filter with its 90-degrees-shifted twin and takes the magnitude of the pair, so
a texture reads the same whether a stripe is light-on-dark or dark-on-light
under it. Training rotates images by up to 30 degrees and flips them both ways,
so a branch that keyed on raw signed responses would be fighting the
augmentation.

The residual is zero-initialised (`gamma` starts at 0), so a model built here
*is* pretrained Swin-V2-T on the first step, to the bit. Whatever the ablation
measures afterwards is what ST-DAM added, not what a disturbed initialisation
cost. The module is registered with timm, so training it needs no new code:

    python ml_pipeline/src/training/train.py --config ml_pipeline/configs/ablations/stdam_full.yaml

Running this file prints what each variant costs and checks the claims above:

    python ml_pipeline/src/models/stdam.py
"""

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model

BACKBONE = "swinv2_tiny_window8_256"
# (stage index, kernel size, Gabor wavelengths in pixels of that stage's map).
# Stage 2 is 32x32 and stage 3 is 16x16 at 256px input; the wavelengths are a
# little shorter on the smaller map so a filter still spans a few pixels.
PLACEMENT = ((1, 7, (3.0, 5.0)), (2, 5, (2.5, 4.0)))
REDUCTION = 4          # branch width is dim // REDUCTION
HEADS = 2              # cross-attention heads
N_ORIENTATIONS = 4     # Gabor orientations, each as an (even, odd) quadrature pair

SOBEL = torch.tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]) / 8.0


def gabor_bank(ksize, wavelengths, n_orientations=N_ORIENTATIONS, aspect=0.5):
    """Quadrature Gabor pairs: (2 * len(wavelengths) * n_orientations, k, k).

    Each orientation contributes a cosine filter and its sine twin, kept
    adjacent so pair_magnitude() can combine them. Every filter is made
    zero-mean, so flat regions give no response, and normalised to unit L1, so
    one wavelength cannot dominate another by scale alone.
    """
    half = ksize // 2
    yy, xx = torch.meshgrid(torch.arange(-half, half + 1.0),
                            torch.arange(-half, half + 1.0), indexing="ij")
    kernels = []
    for wavelength in wavelengths:
        sigma = 0.56 * wavelength                      # ~1 octave bandwidth
        for i in range(n_orientations):
            theta = math.pi * i / n_orientations
            x_r = xx * math.cos(theta) + yy * math.sin(theta)
            y_r = -xx * math.sin(theta) + yy * math.cos(theta)
            envelope = torch.exp(-(x_r ** 2 + (aspect * y_r) ** 2) / (2 * sigma ** 2))
            for phase in (torch.cos, torch.sin):
                kernel = envelope * phase(2 * math.pi * x_r / wavelength)
                kernel = kernel - kernel.mean()
                kernels.append(kernel / kernel.abs().sum().clamp_min(1e-8))
    return torch.stack(kernels)


class FilterBank(nn.Module):
    """Applies one bank of k x k filters to every channel independently.

    A depthwise convolution whose weights are the same bank repeated per
    channel. Fixed by default (a buffer, not saved in checkpoints since it is
    reproducible); `learned` makes the same shape trainable, which is the
    ablation that asks whether the fixed prior is doing the work.
    """

    def __init__(self, channels, bank, learned=False):
        super().__init__()
        n, k, _ = bank.shape
        self.channels, self.n, self.pad = channels, n, k // 2
        weight = bank.repeat(channels, 1, 1).unsqueeze(1)        # (C*n, 1, k, k)
        if learned:
            self.weight = nn.Parameter(weight.clone())           # starts from the prior
        else:
            self.register_buffer("weight", weight, persistent=False)

    def forward(self, x):
        return F.conv2d(x, self.weight.to(x.dtype), padding=self.pad, groups=self.channels)


def pair_magnitude(y, channels, n):
    """(B, C*n, H, W) of adjacent (even, odd) pairs -> (B, C*n/2, H, W).

    sqrt(even^2 + odd^2) in float32: under fp16 autocast the squares of small
    responses would flush to zero and the gradient with them.
    """
    b, _, h, w = y.shape
    pairs = y.float().view(b, channels, n // 2, 2, h, w)
    return pairs.pow(2).sum(3).add(1e-6).sqrt().view(b, channels * n // 2, h, w).to(y.dtype)


class SqueezeExcite(nn.Module):
    """Channel attention: which texture channels matter for this image."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
                                nn.Conv2d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(x.mean((2, 3), keepdim=True))


class STDAM(nn.Module):
    """Shape-Texture Decoupled Attention. Takes and returns (B, H, W, C)."""

    def __init__(self, dim, ksize, wavelengths, reduction=REDUCTION, heads=HEADS,
                 shape=True, texture=True, fusion="cross", learned=False):
        super().__init__()
        if not (shape or texture):
            raise ValueError("ST-DAM needs at least one branch")
        if fusion not in ("cross", "sum"):
            raise ValueError(f"fusion must be 'cross' or 'sum', got {fusion!r}")
        width = max(8, dim // reduction)
        self.width, self.heads, self.shape, self.texture = width, heads, shape, texture
        # cross-attention needs both branches to have something to exchange
        self.fusion = fusion if (shape and texture) else "sum"

        self.reduce = nn.Sequential(nn.Conv2d(dim, width, 1, bias=False),
                                    nn.BatchNorm2d(width), nn.GELU())
        if shape:
            self.sobel = FilterBank(width, SOBEL, learned)
            self.shape_proj = nn.Sequential(nn.Conv2d(width, width, 1, bias=False),
                                            nn.BatchNorm2d(width), nn.ReLU(inplace=True))
            self.spatial_gate = nn.Conv2d(width, 1, 7, padding=3)
        if texture:
            bank = gabor_bank(ksize, wavelengths)
            self.gabor = FilterBank(width, bank, learned)
            self.n_gabor = bank.shape[0]
            self.texture_proj = nn.Sequential(
                nn.Conv2d(width * self.n_gabor // 2, width, 1, bias=False),
                nn.BatchNorm2d(width), nn.ReLU(inplace=True))
            self.se = SqueezeExcite(width)
        if self.fusion == "cross":
            self.q = nn.Linear(width, width, bias=False)
            self.kv = nn.Linear(width, 2 * width, bias=False)
            self.out = nn.Linear(width, width)
        self.expand = nn.Conv2d(width, dim, 1)
        # zero-init, per channel: the module starts as the identity, so the
        # backbone it is inserted into begins exactly as its pretrained self
        self.gamma = nn.Parameter(torch.zeros(dim))

    def branches(self, z):
        shape_feat = texture_feat = None
        if self.shape:
            edges = pair_magnitude(self.sobel(z), self.width, 2)       # |grad|, any orientation
            shape_feat = self.shape_proj(edges)
            shape_feat = shape_feat * torch.sigmoid(self.spatial_gate(shape_feat))
        if self.texture:
            energy = pair_magnitude(self.gabor(z), self.width, self.n_gabor)
            texture_feat = self.se(self.texture_proj(energy))
        return shape_feat, texture_feat

    def cross_attend(self, shape_feat, texture_feat):
        """Shape queries texture: for each location, what texture lies on this edge."""
        b, d, h, w = shape_feat.shape
        q = self.q(shape_feat.flatten(2).transpose(1, 2))
        q = q.view(b, h * w, self.heads, -1).transpose(1, 2)
        k, v = self.kv(texture_feat.flatten(2).transpose(1, 2)) \
            .view(b, h * w, 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, k, v)
        attended = attended.transpose(1, 2).reshape(b, h * w, d)
        return self.out(attended).transpose(1, 2).view(b, d, h, w)

    def forward(self, x):
        z = self.reduce(x.permute(0, 3, 1, 2))                 # NHWC -> NCHW
        shape_feat, texture_feat = self.branches(z)
        fused = sum(f for f in (shape_feat, texture_feat) if f is not None)
        if self.fusion == "cross":
            fused = fused + self.cross_attend(shape_feat, texture_feat)
        return x + self.gamma * self.expand(fused).permute(0, 2, 3, 1)


def insert_stdam(model, **kwargs):
    """Append an ST-DAM block to the end of each stage in PLACEMENT.

    A stage runs its blocks in order, so appending puts ST-DAM after that
    stage's Swin blocks have finished -- the "most complete and mature" point
    STA-Net argues for, and before the next stage halves the resolution.

    The new layers draw their initial weights from a fork of the random number
    stream, which is then put back as it was. train.py seeds first and builds
    the model second, so without this every draw after ST-DAM's would shift:
    the classifier head would start from different weights than the plain
    Swin-V2-T control's at the same seed, and each epoch would visit the
    images in a different order with different augmentations. "Seed 1 against
    seed 1" would then share the number and nothing else. With the fork, the
    two runs start from the same head and see the same images in the same
    order, and since ST-DAM's forward pass draws nothing random, they stay in
    step for the whole run -- so a paired comparison pairs runs that differ in
    ST-DAM and in nothing else.
    """
    added = []
    with torch.random.fork_rng(devices=[]):     # built on the CPU: only its stream moves
        for index, ksize, wavelengths in PLACEMENT:
            stage = model.layers[index]
            dim = stage.blocks[-1].dim
            stage.blocks.append(STDAM(dim, ksize, wavelengths, **kwargs))
            added.append((index, dim))
    return added


def _variant(pretrained, shape=True, texture=True, fusion="cross", learned=False, **kwargs):
    """Build Swin-V2-T from timm, then insert ST-DAM. Returning timm's own model
    keeps its pretrained_cfg, so the training loop resolves input size and
    normalisation exactly as it does for the plain backbone."""
    model = timm.create_model(BACKBONE, pretrained=pretrained, **kwargs)
    insert_stdam(model, shape=shape, texture=texture, fusion=fusion, learned=learned)
    return model


@register_model
def swinv2_tiny_stdam(pretrained=False, **kwargs):
    """Both branches, cross-attention fusion, fixed filters. The full module."""
    return _variant(pretrained, **kwargs)


@register_model
def swinv2_tiny_stdam_shape(pretrained=False, **kwargs):
    """Shape branch only -- what the Sobel half contributes on its own."""
    return _variant(pretrained, texture=False, **kwargs)


@register_model
def swinv2_tiny_stdam_texture(pretrained=False, **kwargs):
    """Texture branch only -- what the Gabor half contributes on its own."""
    return _variant(pretrained, shape=False, **kwargs)


@register_model
def swinv2_tiny_stdam_sum(pretrained=False, **kwargs):
    """Both branches added together: fusion ablated away, branches kept."""
    return _variant(pretrained, fusion="sum", **kwargs)


@register_model
def swinv2_tiny_stdam_learned(pretrained=False, **kwargs):
    """Sobel and Gabor weights trainable, starting from the fixed prior:
    does fixing them help, or merely cost?"""
    return _variant(pretrained, learned=True, **kwargs)


VARIANTS = ["swinv2_tiny_stdam", "swinv2_tiny_stdam_shape", "swinv2_tiny_stdam_texture",
            "swinv2_tiny_stdam_sum", "swinv2_tiny_stdam_learned"]


if __name__ == "__main__":
    import os
    import sys
    import time

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from classifier import Classifier  # noqa: E402

    torch.manual_seed(0)
    print("Gabor bank: zero-mean and unit-L1 filters")
    bank = gabor_bank(7, (3.0, 5.0))
    print(f"  {bank.shape[0]} filters of {bank.shape[1]}x{bank.shape[2]}, "
          f"max |mean| {bank.mean((1, 2)).abs().max():.2e}, "
          f"L1 in [{bank.abs().sum((1, 2)).min():.3f}, {bank.abs().sum((1, 2)).max():.3f}]")
    assert bank.mean((1, 2)).abs().max() < 1e-6, "a Gabor filter is not zero-mean"

    print("\nQuadrature magnitude is phase invariant")
    size, k = 32, 7
    single = gabor_bank(k, (4.0,), n_orientations=1)
    grid = torch.arange(size).float()
    for shift in (0.0, 0.25, 0.5):
        wave = torch.sin(2 * math.pi * (grid / 4.0 + shift)).expand(size, size)[None, None]
        response = F.conv2d(wave, single[:, None], padding=k // 2)
        magnitude = pair_magnitude(response, 1, 2)
        print(f"  grating shifted by {shift:.2f} wavelengths -> mean energy "
              f"{magnitude.mean():.4f}")

    print("\nIdentity at initialisation, and parameter cost")
    vanilla = Classifier(BACKBONE, 38, pretrained=True)
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        reference = vanilla.eval()(x)
    base_params = sum(p.numel() for p in vanilla.parameters())
    print(f"  {'variant':<30}{'params':>10}{'added':>9}{'max |logit diff|':>19}")
    for name in VARIANTS:
        torch.manual_seed(0)
        model = Classifier(name, 38, pretrained=True)
        model.head.load_state_dict(vanilla.head.state_dict())
        with torch.no_grad():
            diff = (model.eval()(x) - reference).abs().max().item()
        n = sum(p.numel() for p in model.parameters())
        print(f"  {name:<30}{n / 1e6:>9.2f}M{(n - base_params) / 1e3:>8.0f}k{diff:>19.2e}")
        assert diff < 1e-4, f"{name} does not start as plain Swin-V2-T"
        del model

    print("\nWhere inserted, and how the zero-initialised residual trains")
    model = Classifier("swinv2_tiny_stdam", 38, pretrained=False)
    for index, _, _ in PLACEMENT:
        block = model.backbone.layers[index].blocks[-1]
        print(f"  stage {index + 1}: ST-DAM on {block.expand.out_channels} channels, "
              f"branch width {block.width}")

    # gamma = 0 makes the module the identity, so on the very first step only
    # gamma itself has a gradient: every branch weight is multiplied by it.
    # That is how ReZero (Bachlechner et al., 2020) and LayerScale behave. What
    # has to be true is that gamma moves, and that the branches train from then
    # on -- a module whose branches never received gradient would be dead.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    stdam = model.backbone.layers[1].blocks[-1]
    watched = (("gamma", lambda m: m.gamma), ("shape gate", lambda m: m.spatial_gate.weight),
               ("texture proj", lambda m: m.texture_proj[0].weight),
               ("cross-attn q", lambda m: m.q.weight), ("expand", lambda m: m.expand.weight))
    norms = []
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        model(torch.randn(2, 3, 256, 256)).sum().backward()
        norms.append({label: get(stdam).grad.norm().item() for label, get in watched})
        opt.step()
    print(f"  {'parameter':<16}{'grad at step 1':>16}{'grad at step 2':>16}")
    for label, _ in watched:
        print(f"  {label:<16}{norms[0][label]:>16.3e}{norms[1][label]:>16.3e}")
    assert norms[0]["gamma"] > 0, "gamma gets no gradient, so ST-DAM could never switch on"
    for label, _ in watched:
        assert norms[1][label] > 0, f"{label} still has no gradient after a step -- branch is dead"
    assert stdam.gamma.abs().max().item() > 0, "gamma did not move off zero"

    if torch.cuda.is_available():
        print("\nTraining step, batch 32, 256px, fp16 (this GPU)")
        for name in [BACKBONE] + VARIANTS:
            model = Classifier(name, 38, pretrained=False).cuda()
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            images = torch.randn(32, 3, 256, 256, device="cuda")
            labels = torch.randint(0, 38, (32,), device="cuda")
            torch.cuda.reset_peak_memory_stats()
            for i in range(12):
                if i == 4:
                    torch.cuda.synchronize()
                    start = time.time()
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = F.cross_entropy(model(images), labels)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            torch.cuda.synchronize()
            ms = (time.time() - start) / 8 * 1000
            print(f"  {name:<30}{ms:>7.0f} ms/step  peak "
                  f"{torch.cuda.max_memory_allocated() / 1024 ** 3:>5.2f} GB")
            del model, opt
            torch.cuda.empty_cache()

    print("\nOK: ST-DAM starts as plain Swin-V2-T, both branches train, every variant builds.")
