# Renders token feature fields to PNG and reports how different Q/K/V
# actually are (the method assumes they're meaningfully different channels;
# the correlation matrix below is the quantitative check).

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.config import Config
from src.extract.run_extraction import parse_meta
from src.extract.tensor_ops import PROJECTIONS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def inspect(cfg: Config, idx: int = 0, n_tokens: int = 4, out: str | None = None) -> None:
    root = cfg.example_dir()
    ex_dir = root / f"{idx:05d}"
    tokens_path = ex_dir / "tokens.npy"
    if not tokens_path.exists():
        raise FileNotFoundError(f"no extracted example at {ex_dir}")

    geometry_path = root / "geometry.json"
    geometry = json.loads(geometry_path.read_text()) if geometry_path.exists() else {}

    arr = np.load(tokens_path)
    field = arr.astype(np.float32)                # (T, L, M, 3)
    meta = parse_meta(ex_dir / "meta.txt")

    print(f"\nexample {idx}  ({ex_dir})")
    print(f"  response: {meta.get('response', '')[:120]}")
    print(f"  gold:     {str(meta.get('gold', ''))[:120]}")
    print(f"  label:    {meta.get('label')}  (1 = hallucinated)")
    print(f"  field:    {field.shape}  (T, layers, segments, channels)")
    print(f"  channels: {list(PROJECTIONS)}")
    print(f"  geometry: L={geometry.get('n_rows', '?')} M={geometry.get('n_segments', '?')}\n")

    print("  magnitude by projection (mean |x|):")
    for c, name in enumerate(PROJECTIONS):
        print(f"    {name}: {np.abs(field[:, :, :, c]).mean():9.4f}")

    print("\n  cross-projection correlation (flattened):")
    flat = [field[:, :, :, c].ravel() for c in range(len(PROJECTIONS))]
    print("        " + "  ".join(f"{n:>6}" for n in PROJECTIONS))
    for i, pi in enumerate(PROJECTIONS):
        row = []
        for j in range(len(PROJECTIONS)):
            r = np.corrcoef(flat[i], flat[j])[0, 1]
            row.append(f"{r:6.3f}")
        print(f"    {pi:>4}  " + "  ".join(row))
    print(
        "\n    (off-diagonal near 1.0 would mean Q/K/V are redundant and the\n"
        "     three-channel field is not buying anything over a single one.)\n"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping PNG render")
        return

    n_tokens = min(n_tokens, field.shape[0])
    n_rows = len(PROJECTIONS)
    fig, axes = plt.subplots(
        n_rows, n_tokens,
        figsize=(2.2 * n_tokens, 2.0 * n_rows),
        squeeze=False,
    )

    for t in range(n_tokens):
        for c, cname in enumerate(PROJECTIONS):
            ax = axes[c][t]
            ax.imshow(field[t, :, :, c], aspect="auto", cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(cname, fontsize=9)
            if c == 0:
                ax.set_title(f"token {t}", fontsize=9)

    fig.suptitle(
        f"{cfg.dataset.name}/{cfg.llm.alias} example {idx} "
        f"(label={meta.get('label')})  |  rows: Q/K/V, y: layers, x: segments",
        fontsize=10,
    )
    fig.tight_layout()

    dest = Path(out or (ex_dir / "preview.png"))
    fig.savefig(dest, dpi=110)
    print(f"  rendered -> {dest}\n")
