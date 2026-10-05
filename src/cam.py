"""Integrated Gradients attribution over the detector's input field.

QKV-STEER ROLE
--------------
This is the LOCALIZE half of the pipeline. `attribution_field` is the reusable
entry point: it returns per-token attribution addressed by the same
(token, layer, segment) coordinates the feature field uses, which is what the
steering stage needs in order to know *where* to intervene. `run_cam` is the
thin rendering wrapper over it, kept for eyeballing single examples.

WHY INTEGRATED GRADIENTS, NOT GRAD-CAM
---------------------------------------
Grad-CAM needs a late-stage activation tensor with spatial extent (h, w) that
still maps back to (L, M) -- it hooks a convolutional block's output. FlatMLP
(the main backbone as of the structure-preservation ablation) has no such
tensor: it flattens (L, M) to a vector BEFORE the first learned weight, so
there is nothing conv-shaped to hook. Integrated Gradients instead
differentiates straight through the whole model back to the RAW (L, M, 3)
input -- it needs nothing but a forward() that is differentiable end to end,
which both FlatMLP and ScratchCNN already are (see classifier.py's
`encode_tokens` docstring). This also means the exact same attribution method
now applies to every backbone, so a diffuse-vs-concentrated comparison across
backbones is not confounded by using two different explanation techniques.

Axiomatically, IG attributes the change in the model's output, relative to a
BASELINE input, to each input coordinate -- here, each (layer, segment,
projection) cell of one token's field. The baseline is all-zeros, which is a
meaningful "absence of signal" point because the field reaching the model has
already been standardised (see src/data/dataset.normalize): zero is each
projection's own training-set mean, not an arbitrary origin.

Note what this does and does not give you. It is signed (can be read as "this
cell pushed the prediction toward/away from hallucinated"); `attribution_field`
returns its absolute value by default, matching Grad-CAM's old non-negative
convention ("where does the detector look", not "which way"), since every
current caller (the CAM figure, attribution_heatmap.py) treats it as a
[0, 1]-normalisable heatmap. Pass `signed=True` for the raw signed map.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.data.dataset import collate
from src.models.classifier import build_model
from src.utils.logger import get_logger
from src.utils.seed import pick_device, seed_everything

logger = get_logger(__name__)

#: Riemann-sum steps along the straight-line path from baseline to input.
#: 32 is well above the ~20-30 where IG's standard convergence check
#: (completeness: sum(attributions) == f(input) - f(baseline)) typically
#: stabilises for a model this size; see `integrated_gradients`'s docstring.
#: Cost is exactly n_steps forward+backward passes through the WHOLE model
#: (temporal encoder included) -- on CPU this is seconds per step even for a
#: small scratch_cnn, so a figure script sweeping many examples/checkpoints
#: (attribution_heatmap.py) should run on a GPU, or lower n_steps and confirm
#: with `check_completeness` that accuracy hasn't meaningfully degraded.
DEFAULT_N_STEPS = 32


def load_detector(cfg: Config, checkpoint: str | Path, device) -> tuple[torch.nn.Module, dict]:
    """Load a trained detector and its checkpoint dict.

    Shared by the CAM path here and by the steering stage, so both instantiate
    the detector identically.
    """
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_model(
        cfg, field_shape=ckpt.get("field_shape"), in_ch=ckpt.get("in_ch")
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _load_example(cfg: Config, ckpt: dict, dataset_name: str, idx: int):
    """One example, collated into a batch of size 1 (so masking matches training)."""
    from src.train import load_source

    source = load_source(cfg, dataset_name, cfg.llm.alias, field_source=cfg.extract.source)
    source.stats = ckpt["stats"]
    return collate([source[idx]]), source


def integrated_gradients(
    model: torch.nn.Module,
    images: torch.Tensor,
    mask: torch.Tensor,
    n_steps: int = DEFAULT_N_STEPS,
    baseline: torch.Tensor | float = 0.0,
) -> tuple[torch.Tensor, float]:
    """IG attribution of the model's logit w.r.t. every (T, 3, L, M) input cell.

    Args:
        model:    a detector in eval mode. BatchNorm1d in TemporalEncoder's
                  conv stack needs a batch of size > 1 in train() mode but is
                  fine in eval() mode with a batch of 1, which is what this
                  always runs with -- no special-casing needed (contrast the
                  old Grad-CAM path's `_set_eval_but_cudnn_rnn_trainable`,
                  which does not apply here: IG never needs the LSTM's own
                  backward in train mode, only one scalar's gradient back to
                  the INPUT, which eval-mode autograd gives for free).
        images:   (1, T, 3, L, M) -- one example, already normalised.
        mask:     (1, T) bool, True at real tokens.
        n_steps:  interpolation points on the straight-line path from
                  `baseline` to `images` (Riemann-sum approximation of the
                  path integral). Completeness (see module docstring) is the
                  standard way to sanity-check this is high enough: raising
                  n_steps should leave the attribution sum roughly unchanged.
        baseline: the "absence of signal" input IG attributes relative to.
                  0.0 (default) is well-defined here because the field is
                  already standardised per projection at this point in the
                  pipeline -- zero IS each projection's training-set mean.

    Returns:
        (A, p) where A is (T, 3, L, M) SIGNED attribution (same shape as the
        input), and p is the predicted hallucination probability. Summing A
        over one example should approximately equal
        `model_logit(images) - model_logit(baseline)` (completeness); this is
        not asserted at runtime (it is a sanity property to spot-check while
        developing, not a per-call invariant worth paying for), see
        `check_completeness`.
    """
    images = images.detach()
    if isinstance(baseline, torch.Tensor):
        base = baseline.detach().to(images.device, images.dtype)
    else:
        base = torch.full_like(images, float(baseline))

    diff = images - base  # (1, T, 3, L, M)

    # alphas in (0, 1], not including 0: the gradient AT the baseline itself is
    # not part of the path integral's Riemann sum (the standard left-endpoint
    # convention used by the reference IG implementation).
    alphas = torch.linspace(1.0 / n_steps, 1.0, n_steps, device=images.device)

    grad_sum = torch.zeros_like(images)
    prob = None
    for alpha in alphas:
        interpolated = (base + alpha * diff).clone().requires_grad_(True)
        logit = model(interpolated, mask)  # (1,)
        if prob is None:
            with torch.no_grad():
                prob = torch.sigmoid(model(images, mask)).item()
        model.zero_grad(set_to_none=True)
        logit.sum().backward()
        grad_sum += interpolated.grad.detach()

    avg_grad = grad_sum / n_steps
    attribution = diff * avg_grad  # (1, T, 3, L, M)
    return attribution.squeeze(0), prob  # (T, 3, L, M), float


def check_completeness(
    model: torch.nn.Module, images: torch.Tensor, mask: torch.Tensor, attribution: torch.Tensor
) -> tuple[float, float]:
    """(attribution_sum, logit_delta) -- IG's own correctness check.

    Not called by default (extra forward passes); use while developing or
    when raising/lowering DEFAULT_N_STEPS, to confirm n_steps is high enough
    that the two numbers agree to within a few percent.
    """
    with torch.no_grad():
        logit_input = model(images, mask).item()
        logit_base = model(torch.zeros_like(images), mask).item()
    return float(attribution.sum().item()), logit_input - logit_base


def attribution_field(
    model: torch.nn.Module,
    images: torch.Tensor,
    mask: torch.Tensor,
    n_steps: int = DEFAULT_N_STEPS,
    signed: bool = False,
    per_projection: bool = False,
) -> tuple[torch.Tensor, float]:
    """Per-token Integrated Gradients attribution, in the field's own
    (layer, segment[, projection]) coordinates, plus the detector's probability.

    Args:
        model:  a detector in eval mode.
        images: (1, T, 3, L, M) -- one example, already normalised.
        mask:   (1, T) bool, True at real tokens.
        signed: False (default) returns |attribution|, matching every current
                caller's [0, 1]-normalisable-heatmap expectation (the old
                Grad-CAM convention). True returns the raw signed map, where
                a positive cell pushed the logit toward "hallucinated".
        per_projection: False (default) sums the channel axis away, returning
                (T, L, M) -- a drop-in match for the old Grad-CAM-pooled shape
                every current caller (run_cam, attribution_heatmap.py)
                expects. True keeps the channel axis, returning (T, L, M, 3)
                -- unlike Grad-CAM (which pools channels because the CNN's
                first conv already mixed them), IG differentiates the raw
                INPUT, so a genuine per-(Q, K, V) map is available if wanted.

    Returns:
        (A, p): A is (T_real, L, M) or (T_real, L, M, 3) depending on
        `per_projection`; p is the predicted hallucination probability.
    """
    attribution, prob = integrated_gradients(model, images, mask, n_steps=n_steps)
    # encode_tokens folds (B*T) into the batch, and B == 1 here, so this axis
    # IS the token axis, in generation order.
    real_tokens = int(mask[0].sum().item())
    attribution = attribution[:real_tokens]          # (T_real, 3, L, M)

    if not signed:
        attribution = attribution.abs()

    # (T, 3, L, M) -> (T, L, M, 3): channel axis last, matching the field's
    # own (and every figure script's) convention.
    attribution = attribution.permute(0, 2, 3, 1)     # (T_real, L, M, 3)
    if not per_projection:
        attribution = attribution.sum(dim=-1)         # (T_real, L, M)

    return attribution.detach(), prob


def run_cam(
    cfg: Config,
    checkpoint: str | Path,
    dataset_name: str | None = None,
    idx: int = 0,
    method: str = "ig",
    out: str | None = None,
    max_tokens_shown: int | None = 20,
) -> Path:
    """Render an Integrated-Gradients attribution figure for one example to a PNG.

    `method` is accepted (and must be "ig") only to keep detector.py's existing
    `cam` subcommand signature stable; there is only one attribution method now.

    One column per GENERATED TOKEN of the response -- every token gets its own
    heatmap, nothing is averaged away. `max_tokens_shown` caps how many token
    columns get rendered (keeps the figure legible on long responses); pass None
    to always render every real token.
    """
    if method != "ig":
        raise ValueError(f"only method='ig' is supported now, got {method!r}")

    seed_everything(cfg.train.seed)
    device = pick_device()

    ckpt_path = Path(checkpoint)
    model, ckpt = load_detector(cfg, ckpt_path, device)

    name = dataset_name or cfg.dataset.name
    batch, source = _load_example(cfg, ckpt, name, idx)
    images, labels, mask, origins = batch
    images = images.to(device)          # (1, T, 3, L, M)
    mask = mask.to(device)              # (1, T)

    attribution, prob = attribution_field(model, images, mask)

    real_tokens = attribution.shape[0]
    n_shown = real_tokens if max_tokens_shown is None else min(real_tokens, max_tokens_shown)
    if n_shown < real_tokens:
        logger.warning(
            "response has %d tokens; showing the first %d (pass max_tokens_shown=None for all)",
            real_tokens, n_shown,
        )

    dest = Path(out) if out else ckpt_path.parent / f"cam_ig_{name}_{idx:05d}.png"
    if dest.suffix.lower() != ".png":
        # _render always writes via fm.save, which is PNG(+PDF)-only -- fail
        # loudly on a mismatched --out rather than silently write a .png next
        # to the .ext the caller asked for and then claim the wrong path below.
        raise ValueError(f"--out must end in .png, got {dest.name!r}")
    _render(
        attribution=attribution[:n_shown].cpu().numpy(),
        images=images[0, :n_shown].cpu().numpy(),   # (T_shown, 3, L, M)
        label=int(labels.item()),
        prob=prob,
        origin=origins[0],
        idx=idx,
        dest=dest,
    )
    logger.info("wrote %s (label=%d, p(hallucinated)=%.3f)", dest, int(labels.item()), prob)
    return dest


#: Row labels for the rendered grid: the field's three channels.
_CHANNEL_NAMES = ("Q", "K", "V")


def _render(
    attribution: np.ndarray,
    images: np.ndarray,
    label: int,
    prob: float,
    origin: str,
    idx: int,
    dest: Path,
) -> None:
    """Grid layout: one ROW per projection (Q/K/V), one COLUMN per token.

    attribution: (T, L, M) -- the same attribution overlaid on each row (this
                 is the channel-summed map; see attribution_field's
                 `per_projection` for a genuinely per-channel one).
    images:      (T, 3, L, M) -- the actual model input.
    """
    import matplotlib.pyplot as plt

    from scripts.figures import style_modern as fm

    fm.apply()
    n_tokens = attribution.shape[0]
    n_rows = len(_CHANNEL_NAMES)

    fig, axes = plt.subplots(
        n_rows, n_tokens,
        figsize=(1.6 * n_tokens, 1.6 * n_rows),
        squeeze=False,
    )
    for c in range(n_rows):
        for t in range(n_tokens):
            ax = axes[c][t]
            ax.imshow(images[t, c], cmap="gray", aspect="auto")
            # jet is deliberately never used (see style.py's docstring --
            # perceptually non-uniform, illegible in greyscale, collapses
            # under CVD); the attribution overlay here is a NON-NEGATIVE
            # magnitude (unless run_cam is changed to request signed=True),
            # so SEQUENTIAL is the correct validated map, not DIVERGING.
            ax.imshow(attribution[t], cmap=fm.SEQUENTIAL, alpha=0.55, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            fm.strip_frame(ax, keep_bottom=False)
            if c == 0:
                ax.set_title(f"t{t}", fontsize=8, color=fm.INK)
            if t == 0:
                ax.set_ylabel(_CHANNEL_NAMES[c], fontsize=9, color=fm.INK_SOFT)

    fm.title(
        fig,
        f"Integrated Gradients  —  {origin} example {idx}",
        f"label={label} (1=hallucinated), p(hallucinated)={prob:.3f}; "
        f"rows: Q/K/V channel of the field, columns: generated tokens",
    )
    # fm.save always writes <out_dir>/<name>.png (+ .pdf); `run_cam` always
    # builds `dest` with a .png suffix (its own default, or whatever --out the
    # caller passed), so dest.stem is exactly the name fm.save needs, and its
    # .png output lands at `dest` itself -- no extra move/rename required.
    fm.save(fig, dest.parent, dest.stem)
