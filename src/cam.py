# Integrated Gradients attribution over the detector's input field.
#
# IG (not Grad-CAM) because FlatMLP has no conv-shaped activation to hook --
# it flattens (L, M) before the first learned weight. IG differentiates
# straight through to the raw (L, M, 3) input instead, so the same method
# applies to every backbone. Baseline is all-zeros, which is meaningful
# because the field is standardised (src/data/dataset.normalize): zero is
# each projection's training-set mean.

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

# Riemann-sum steps along the baseline->input path. Cost is n_steps
# forward+backward passes through the whole model.
DEFAULT_N_STEPS = 32


def load_detector(cfg: Config, checkpoint: str | Path, device) -> tuple[torch.nn.Module, dict]:
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
    # images: (1, T, 3, L, M), mask: (1, T) bool
    # returns (A, p): A is (T, 3, L, M) signed attribution, p is P(hallucinated)
    images = images.detach()
    if isinstance(baseline, torch.Tensor):
        base = baseline.detach().to(images.device, images.dtype)
    else:
        base = torch.full_like(images, float(baseline))

    diff = images - base  # (1, T, 3, L, M)

    # alphas in (0, 1], excluding 0 (left-endpoint Riemann sum convention)
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
    # (attribution_sum, logit_delta) -- should roughly agree; not called by
    # default, use while tuning DEFAULT_N_STEPS.
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
    # Per-token IG attribution in the field's own (layer, segment[, projection])
    # coordinates. signed=False (default) returns |attribution|; per_projection=False
    # (default) sums the channel axis away to (T, L, M).
    attribution, prob = integrated_gradients(model, images, mask, n_steps=n_steps)
    real_tokens = int(mask[0].sum().item())
    attribution = attribution[:real_tokens]          # (T_real, 3, L, M)

    if not signed:
        attribution = attribution.abs()

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
    # attribution: (T, L, M); images: (T, 3, L, M). One row per Q/K/V, one
    # column per token.
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
    fm.save(fig, dest.parent, dest.stem)
