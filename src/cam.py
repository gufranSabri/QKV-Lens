"""Grad-CAM / Eigen-CAM over the detector's CNN backbone.

Each generated token is one 3-channel (Q, K, V) image of shape (L, M). This
module produces one heatmap per token, showing where that token's field the
backbone's last conv block is looking when the detector makes its prediction.

QKV-STEER ROLE
--------------
This is the LOCALIZE half of the pipeline. `attribution_field` is the reusable
entry point: it returns the per-token attribution as a tensor addressed by the
same (token, layer, segment) coordinates the feature field uses, which is what
the steering stage needs in order to know *where* to intervene. `run_cam` is the
thin rendering wrapper over it, kept for eyeballing single examples.

Note what Grad-CAM does and does not give you. It is non-negative (ReLU'd) and
channel-pooled, so it says WHERE the detector looks, not WHICH WAY to move a
value. The steering direction is a separate factor -- see the plan's D1/D2/D3.

Grad-CAM needs a backward pass (weights = gradient of the target scalar w.r.t.
each channel, averaged spatially); Eigen-CAM needs only a forward pass (the
first principal component of the activation map).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.config import Config
from src.data.dataset import collate
from src.models.classifier import build_model
from src.utils.logger import get_logger
from src.utils.seed import pick_device, seed_everything

logger = get_logger(__name__)

CAM_METHODS = ("gradcam", "eigencam")


def _last_conv_block(backbone: torch.nn.Module) -> torch.nn.Module:
    """The layer whose output we hook: last residual block before pooling."""
    name = backbone.__class__.__name__
    if name == "ScratchCNN":
        return backbone.blocks[-1]
    if name == "ResNet18Adapted":
        return backbone.net.layer4
    raise ValueError(f"no known last-conv-block for backbone {name!r}")


class _Recorder:
    """Grabs a layer's forward activations and (if needed) its gradient."""

    def __init__(self, layer: torch.nn.Module, need_grad: bool):
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._need_grad = need_grad
        self._fwd = layer.register_forward_hook(self._on_forward)
        self._bwd = layer.register_full_backward_hook(self._on_backward) if need_grad else None

    def _on_forward(self, module, inp, out):
        self.activations = out
        if self._need_grad:
            out.retain_grad()

    def _on_backward(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self):
        self._fwd.remove()
        if self._bwd is not None:
            self._bwd.remove()


def _cam_from_activations(
    activations: torch.Tensor, gradients: torch.Tensor | None
) -> torch.Tensor:
    """(N, E, h, w) activations [+ gradients] -> (N, h, w) heatmaps in [0, 1].

    Grad-CAM (gradients given): channel weights = spatially-averaged gradient of
    the target w.r.t. that channel; the map is the ReLU'd weighted sum.
    Eigen-CAM (gradients None): each activation's projection onto the first
    principal component of the channel dimension, computed once over all N token
    maps together so every token is expressed in the same basis.
    """
    if gradients is not None:
        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (N, E, 1, 1)
        cam = (weights * activations).sum(dim=1)             # (N, h, w)
        cam = F.relu(cam)
    else:
        n, e, h, w = activations.shape
        flat = activations.permute(0, 2, 3, 1).reshape(-1, e)    # (N*h*w, E)
        flat = flat - flat.mean(dim=0, keepdim=True)
        # Top right-singular vector = first principal component direction.
        _, _, v = torch.linalg.svd(flat, full_matrices=False)
        pc1 = v[0]                                               # (E,)
        cam = (flat @ pc1).reshape(n, h, w)
        cam = cam.abs()

    # Per-token min-max normalisation so every token's map uses the full range.
    flat = cam.reshape(cam.shape[0], -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo).clamp(min=1e-8)
    return flat.reshape(cam.shape)


def _set_eval_but_cudnn_rnn_trainable(model: torch.nn.Module) -> None:
    """cuDNN refuses to run an LSTM's backward pass in eval mode ("cudnn RNN
    backward can only be called in training mode") -- it only keeps the buffers
    backward needs when training=True. Grad-CAM needs that backward pass, so the
    LSTM has to be flipped to train(); everything else that would make
    training-mode nondeterministic (Dropout, BatchNorm running stats) is forced
    back to eval-style behaviour explicitly.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.LSTM):
            module.train()


def load_detector(cfg: Config, checkpoint: str | Path, device) -> tuple[torch.nn.Module, dict]:
    """Load a trained detector and its checkpoint dict.

    Shared by the CAM path here and by the steering stage, so both instantiate
    the detector identically.
    """
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = build_model(cfg, field_shape=ckpt.get("field_shape")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _load_example(cfg: Config, ckpt: dict, dataset_name: str, idx: int):
    """One example, collated into a batch of size 1 (so masking matches training)."""
    from src.train import load_source

    source = load_source(cfg, dataset_name, cfg.llm.alias)
    source.stats = ckpt["stats"]
    return collate([source[idx]]), source


def attribution_field(
    model: torch.nn.Module,
    images: torch.Tensor,
    mask: torch.Tensor,
    method: str = "gradcam",
    upsample_to_field: bool = True,
) -> tuple[torch.Tensor, float]:
    """Per-token attribution over the QKV field, plus the detector's probability.

    Args:
        model:  a detector in eval mode (Grad-CAM flips its LSTM; see
                `_set_eval_but_cudnn_rnn_trainable`).
        images: (1, T, 3, L, M) -- one example, already normalised.
        mask:   (1, T) bool, True at real tokens.
        upsample_to_field: bilinearly resize each token's heatmap from the
                conv block's (h, w) back to the field's (L, M). Leave True for
                steering, which needs attribution addressed by the SAME
                (layer, segment) coordinates the field uses.

    Returns:
        (A, p) where A is (T_real, L, M) non-negative attribution in [0, 1] per
        token, and p is the predicted hallucination probability.

    A is NOT per-projection: Grad-CAM pools over the conv channel axis, which
    the first conv layer already mixed Q/K/V into. Broadcasting one (l, m) mask
    across all three projections is the honest reading, and the Q/K/V-selective
    arms in the plan's ablation are a separate choice applied on top.
    """
    if method not in CAM_METHODS:
        raise ValueError(f"method must be one of {CAM_METHODS}, got {method!r}")

    need_grad = method == "gradcam"
    if need_grad:
        _set_eval_but_cudnn_rnn_trainable(model)

    recorder = _Recorder(_last_conv_block(model.backbone), need_grad)
    try:
        with torch.set_grad_enabled(need_grad):
            logit = model(images, mask)          # (1,)
            prob = torch.sigmoid(logit).item()
            if need_grad:
                model.zero_grad(set_to_none=True)
                logit.sum().backward()

        acts = recorder.activations              # (T, E, h, w)
        grads = recorder.gradients if need_grad else None
        cam = _cam_from_activations(acts, grads)  # (T, h, w)
    finally:
        recorder.remove()

    # encode_tokens folds (B*T) into the batch, and B == 1 here, so this axis IS
    # the token axis, in generation order.
    real_tokens = int(mask[0].sum().item())
    cam = cam[:real_tokens].detach()

    if upsample_to_field:
        field_hw = images.shape[-2:]             # (L, M)
        if cam.shape[-2:] != field_hw:
            cam = F.interpolate(
                cam.unsqueeze(1).float(), size=field_hw,
                mode="bilinear", align_corners=False,
            ).squeeze(1)

    return cam, prob


def run_cam(
    cfg: Config,
    checkpoint: str | Path,
    dataset_name: str | None = None,
    idx: int = 0,
    method: str = "gradcam",
    out: str | None = None,
    max_tokens_shown: int | None = 20,
) -> Path:
    """Render a Grad-CAM/Eigen-CAM figure for one example to a PNG.

    One column per GENERATED TOKEN of the response -- every token gets its own
    heatmap, nothing is averaged away. `max_tokens_shown` caps how many token
    columns get rendered (keeps the figure legible on long responses); pass None
    to always render every real token.
    """
    seed_everything(cfg.train.seed)
    device = pick_device()

    ckpt_path = Path(checkpoint)
    model, ckpt = load_detector(cfg, ckpt_path, device)

    name = dataset_name or cfg.dataset.name
    batch, source = _load_example(cfg, ckpt, name, idx)
    images, labels, mask, origins = batch
    images = images.to(device)          # (1, T, 3, L, M)
    mask = mask.to(device)              # (1, T)

    cam, prob = attribution_field(model, images, mask, method=method)

    real_tokens = cam.shape[0]
    n_shown = real_tokens if max_tokens_shown is None else min(real_tokens, max_tokens_shown)
    if n_shown < real_tokens:
        logger.warning(
            "response has %d tokens; showing the first %d (pass max_tokens_shown=None for all)",
            real_tokens, n_shown,
        )

    dest = Path(out) if out else ckpt_path.parent / f"cam_{method}_{name}_{idx:05d}.png"
    _render(
        cam=cam[:n_shown].cpu().numpy(),
        images=images[0, :n_shown].cpu().numpy(),   # (T_shown, 3, L, M)
        method=method,
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
    cam: np.ndarray,
    images: np.ndarray,
    method: str,
    label: int,
    prob: float,
    origin: str,
    idx: int,
    dest: Path,
) -> None:
    """Grid layout: one ROW per projection (Q/K/V), one COLUMN per token.

    cam:    (T, L, M) -- the same attribution overlaid on each row, since
            Grad-CAM pools over the conv channel axis and is not per-projection.
    images: (T, 3, L, M) -- the actual model input.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_tokens = cam.shape[0]
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
            ax.imshow(cam[t], cmap="jet", alpha=0.5, aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(f"t{t}", fontsize=8)
            if t == 0:
                ax.set_ylabel(_CHANNEL_NAMES[c], fontsize=9)

    fig.suptitle(
        f"{method}  |  {origin} example {idx}  |  "
        f"label={label} (1=hallucinated)  p(hallucinated)={prob:.3f}  |  "
        f"rows: Q/K/V channel of the field, cols: generated tokens",
        fontsize=11,
    )
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130)
    plt.close(fig)
