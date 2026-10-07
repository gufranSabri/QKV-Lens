# Config loading: YAML -> validated dataclasses, with deep-merge over defaults.
#
# The feature field layout (T, L, M, 3), single-image detector input, and
# greedy decoding are fixed in code, not config keys -- the steering stage
# depends on attribution addressing real (layer, segment, projection)
# coordinates, so changing channel semantics would change what a coordinate
# means. `extract.pool` is the one ablation axis kept as an opt-in knob; a
# non-default pool writes to its own data_root subtree (see
# Config.dataset_dir_for) so it can never shadow the canonical corpus.

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# layer_cnn (mixes across L only) is the MAIN backbone; flat_mlp and
# grid_cnn are the structure-preservation ablation's comparison arms.
VALID_BACKBONES = ("flat_mlp", "layer_cnn", "grid_cnn")
VALID_SCHEMES = ("exact_match", "bleurt")
VALID_SOURCES = ("qkv", "hidden-states")
VALID_COLLAPSE_AXES = ("M", "L", "channels")

# Per-model n_segments baked into that model's own config for a structural
# reason (not an ablation override) -- lets dataset_dir_for tell "this
# model's one true n_segments" apart from a pooling-ablation sweep's M.
CANONICAL_N_SEGMENTS: dict[str, int] = {
    "qwen2.5_7b": 32,
}


@dataclass
class LLMConfig:
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dtype: str = "bfloat16"
    alias: str = "llama3_8b"   # short alias used in output paths


@dataclass
class DatasetConfig:
    name: str = "triviaqa"
    n_samples: int = 10000
    max_new_tokens: int = 64
    prompt_template: str = "Answer the question concisely. Q: {question} A:"


@dataclass
class ExtractConfig:
    dtype: str = "float16"
    max_tokens: int = 100
    batch_size: int = 8
    # M: pooled feature segments per layer. null -> the model's layer count
    # (square field). Paper uses M=32.
    n_segments: int | None = None
    # Pool the LAYER axis to this many rows. null -> keep the model's L.
    # Must stay null for steering: the layer remap is not invertible.
    l_eff: int | None = None
    # Segment-pooling strategy (tensor_ops.POOL_MODES). "mean" is the paper's
    # fixed setting and the only one steering may run against; max/strided
    # exist only to reproduce the pooling ablation.
    pool: str = "mean"
    # Which field train/test/cam/forecasting read: "qkv" or "hidden-states".
    # Not read by extract itself -- see extract-hidden-states (detector.py).
    source: str = "qkv"


@dataclass
class LabelingConfig:
    scheme: str = "bleurt"
    bleurt_threshold: float = 0.5
    bleurt_checkpoint: str = "models/BLEURT-20-D12"


@dataclass
class ModelConfig:
    backbone: str = "layer_cnn"
    embed_dim: int = 2048       # E: backbone output per token
    conv1d_layers: int = 2
    lstm_hidden: int = 2048
    lstm_layers: int = 1
    dropout: float = 0.3
    # Which of the field's 3 fixed channels (Q, K, V) reach the detector.
    # null (default) keeps all three; the field on disk is unchanged either
    # way, this only zeroes channels at data-loading time.
    keep_channels: list[str] | None = None
    # Compress the TOKEN axis to this many buckets by mean-pooling contiguous
    # runs, applied after extract.max_tokens cropping. null keeps every token.
    token_buckets: int | None = None
    # Structure-preservation control: permute the LAYER axis with a fixed
    # seeded permutation, applied to every example. null keeps true order.
    layer_permute_seed: int | None = None
    # Feature-pooling ablation: collapse one field axis to size 1 by
    # averaging, at data-loading time. One of "M", "L", "channels". null
    # collapses nothing. Mutually exclusive with layer_permute_seed.
    collapse_axis: str | None = None


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    # A pretrained backbone needs a gentler LR than the randomly-initialised
    # temporal/head. Its LR is lr * backbone_lr_scale.
    backbone_lr_scale: float = 0.1
    # Hold LR flat for lr_decay_start epochs, then linear decay to
    # lr_final_scale x initial LR by `epochs`.
    lr_decay_start: int = 5
    lr_final_scale: float = 0.0
    epochs: int = 20
    patience: int = 8
    seed: int = 42
    num_workers: int = 4
    val_fraction: float = 0.25
    test_fraction: float = 0.25
    balance_classes: bool = True


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    extract: ExtractConfig = field(default_factory=ExtractConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data_root: str = "/scratch/ahmedubc/QKV-Steer-data"
    runs_root: str = "runs"

    def example_dir(self, root: str | None = None) -> Path:
        return self.dataset_dir_for(self.dataset.name, self.llm.alias, root=root)

    def dataset_dir_for(self, dataset_name: str, llm_alias: str, root: str | None = None) -> Path:
        # {data_root}/{dataset}/{llm_alias}/                                  canonical
        # {data_root}/pool_{pool}[_ncols{n_segments}]/{dataset}/{llm_alias}/  ablation cell
        #
        # Checked against the MODEL's own registered canonical n_segments
        # (not just "is n_segments None"): Qwen2.5-7B's n_segments=32 is
        # baked into its config and must resolve to the bare canonical path,
        # while the other models' non-32 M cells in a pooling-ablation sweep
        # must NOT collapse onto that same canonical path.
        base = Path(root or self.data_root)
        e = self.extract
        canonical_n_segments = CANONICAL_N_SEGMENTS.get(llm_alias)
        is_canonical = e.pool == "mean" and e.n_segments == canonical_n_segments
        if not is_canonical:
            suffix = f"pool_{e.pool}"
            if e.n_segments is not None:
                suffix += f"_ncols{e.n_segments}"
            base = base / suffix
        return base / dataset_name / llm_alias

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        e, m, la, t = self.extract, self.model, self.labeling, self.train

        if e.max_tokens < 1:
            raise ValueError("extract.max_tokens must be >= 1")
        if e.batch_size < 1:
            raise ValueError("extract.batch_size must be >= 1")
        if e.n_segments is not None and e.n_segments < 1:
            raise ValueError("extract.n_segments must be >= 1 or null")
        if e.l_eff is not None and e.l_eff < 1:
            raise ValueError("extract.l_eff must be >= 1 or null")
        from src.extract.tensor_ops import POOL_MODES

        if e.pool not in POOL_MODES:
            raise ValueError(f"extract.pool must be one of {POOL_MODES}, got {e.pool!r}")
        if e.source not in VALID_SOURCES:
            raise ValueError(f"extract.source must be one of {VALID_SOURCES}, got {e.source!r}")

        if m.backbone not in VALID_BACKBONES:
            raise ValueError(
                f"model.backbone must be one of {VALID_BACKBONES}, got {m.backbone!r}"
            )
        if m.embed_dim < 1:
            raise ValueError("model.embed_dim must be >= 1")
        if m.lstm_layers < 1:
            raise ValueError("model.lstm_layers must be >= 1")
        if m.conv1d_layers < 0:
            raise ValueError("model.conv1d_layers must be >= 0")
        if not 0.0 <= m.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if m.keep_channels is not None:
            from src.extract.tensor_ops import PROJECTIONS

            if not m.keep_channels:
                raise ValueError("model.keep_channels must not be empty")
            if len(set(m.keep_channels)) != len(m.keep_channels):
                raise ValueError(f"model.keep_channels has duplicates: {m.keep_channels}")
            bad = [c for c in m.keep_channels if c not in PROJECTIONS]
            if bad:
                raise ValueError(
                    f"model.keep_channels: unknown {bad}, valid are {list(PROJECTIONS)}"
                )
        if m.token_buckets is not None and m.token_buckets < 1:
            raise ValueError("model.token_buckets must be >= 1 or null")
        if m.layer_permute_seed is not None and m.layer_permute_seed < 0:
            raise ValueError("model.layer_permute_seed must be >= 0 or null")
        if m.collapse_axis is not None and m.collapse_axis not in VALID_COLLAPSE_AXES:
            raise ValueError(
                f"model.collapse_axis must be one of {VALID_COLLAPSE_AXES} or null, "
                f"got {m.collapse_axis!r}"
            )
        if m.collapse_axis == "L" and m.layer_permute_seed is not None:
            raise ValueError(
                "model.collapse_axis='L' and model.layer_permute_seed are "
                "mutually exclusive: collapsing the layer axis to size 1 "
                "leaves nothing for a layer-order permutation to reorder."
            )
        if m.collapse_axis == "channels" and m.keep_channels is not None:
            raise ValueError(
                "model.collapse_axis='channels' and model.keep_channels are "
                "mutually exclusive: averaging Q/K/V into one channel leaves "
                "nothing for a per-projection keep-list to select."
            )

        if la.scheme not in VALID_SCHEMES:
            raise ValueError(f"labeling.scheme must be one of {VALID_SCHEMES}")

        if not 0.0 < t.val_fraction < 1.0:
            raise ValueError("train.val_fraction must be in (0, 1)")
        if not 0.0 <= t.test_fraction < 1.0:
            raise ValueError("train.test_fraction must be in [0, 1)")
        if t.val_fraction + t.test_fraction >= 1.0:
            raise ValueError("train.val_fraction + train.test_fraction must be < 1")

        if t.lr_decay_start < 0:
            raise ValueError("train.lr_decay_start must be >= 0")
        if t.lr_decay_start >= t.epochs:
            raise ValueError(
                f"train.lr_decay_start ({t.lr_decay_start}) must be < train.epochs "
                f"({t.epochs}), or the LR never starts decaying"
            )
        if not 0.0 <= t.lr_final_scale < 1.0:
            raise ValueError("train.lr_final_scale must be in [0, 1)")
        if t.backbone_lr_scale <= 0.0:
            raise ValueError("train.backbone_lr_scale must be > 0")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


_SECTIONS = {
    "llm": LLMConfig,
    "dataset": DatasetConfig,
    "extract": ExtractConfig,
    "labeling": LabelingConfig,
    "model": ModelConfig,
    "train": TrainConfig,
}

# Keys that existed in QKV-Lens and are deliberately gone. Named explicitly
# so a leftover config fails loudly instead of silently running under
# settings the user believes are in effect.
_REMOVED_KEYS: dict[str, str] = {
    "extract.extraction_type": "delta/transform channels competed for the channel axis, which now holds Q/K/V",
    "extract.views": "Q, K and V are always all three, as the field's channel axis",
    "extract.boundary_mode": "only meaningful for the removed delta channels",
    "extract.n_cols": "renamed to extract.n_segments (the paper's M)",
    "model.channels": "each token is one 3-channel image; there is nothing to regroup",
    "model.include": "there is only one image stream to keep",
    "model.fusion": "a single stream never had anything to fuse (build_fusion always returned identity)",
    "model.share_backbone": "there is only one backbone",
    "model.fused_dim": "no fusion stage; the CNN's embed_dim feeds the temporal encoder directly",
}


def _check_removed(raw: dict) -> None:
    found = []
    for dotted, why in _REMOVED_KEYS.items():
        section, key = dotted.split(".")
        if isinstance(raw.get(section), dict) and key in raw[section]:
            found.append(f"  {dotted}: {why}")
    if found:
        raise ValueError(
            "config uses option(s) removed in QKV-Steer:\n"
            + "\n".join(found)
            + "\n\nThese were QKV-Lens ablation axes. See the top of src/config.py "
              "for what is fixed and why."
        )


def _build(raw: dict) -> Config:
    _check_removed(raw)

    kwargs: dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        if not isinstance(section, dict):
            raise ValueError(f"config section '{name}' must be a mapping")
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(section) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in '{name}': {sorted(unknown)}. Valid: {sorted(known)}"
            )
        kwargs[name] = cls(**dict(section))

    for top in ("data_root", "runs_root"):
        if top in raw:
            kwargs[top] = raw[top]

    unknown_top = set(raw) - set(_SECTIONS) - {"data_root", "runs_root"}
    if unknown_top:
        raise ValueError(f"unknown top-level config key(s): {sorted(unknown_top)}")

    cfg = Config(**kwargs)
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    raw: dict = {}
    if DEFAULT_CONFIG.exists() and path.resolve() != DEFAULT_CONFIG.resolve():
        with open(DEFAULT_CONFIG) as f:
            raw = yaml.safe_load(f) or {}

    with open(path) as f:
        raw = _deep_merge(raw, yaml.safe_load(f) or {})

    if overrides:
        raw = _deep_merge(raw, overrides)

    return _build(raw)
