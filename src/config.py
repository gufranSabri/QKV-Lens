# Config loading: YAML -> validated dataclasses, with deep-merge over defaults.
#
# The feature field layout (T, L, M, 3), single-image detector input, and
# greedy decoding are fixed in code, not config keys -- the steering stage
# depends on attribution addressing real (layer, segment, projection)
# coordinates, so changing channel semantics would change what a coordinate
# means. Each (extract.source, extract.pool, extract.n_segments) combination
# is its own variant directory under {data_root}/{dataset}/{llm_alias}/ (see
# Config.variant_name) so it can never shadow the default corpus.

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

VALID_SCHEMES = ("exact_match", "bleurt", "llm_judge")
VALID_SOURCES = ("qkv", "hidden-states", "hidden-states-normed")
VALID_COLLAPSE_AXES = ("M", "L", "channels")
VALID_TOKEN_ROWS = ("generated", "prompt", "all")
DEFAULT_VARIANT = "default"
SOURCE_TAGS = {"qkv": "qkv", "hidden-states": "hs", "hidden-states-normed": "hsn"}
# What `extract --variants all` saves on top of the config's own variant.
ALL_VARIANTS = (
    "qkv_M8", "qkv_M16", "qkv_M64", "qkv_M128",
    "hs", "hs_M8", "hs_M16", "hs_M64", "hs_M128",
    "hsn", "hsn_M8", "hsn_M16", "hsn_M64", "hsn_M128",
    "qkv_max", "qkv_min",
)

# Per-model n_segments baked into that model's own config for a structural
# reason (not an ablation override) -- lets dataset_dir_for tell "this
# model's one true n_segments" apart from an ablation variant's M.
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
    # fixed setting and the only one steering may run against.
    pool: str = "mean"
    # Field type, one of VALID_SOURCES; with pool and n_segments it picks the variant dir.
    source: str = "qkv"


@dataclass
class LabelingConfig:
    scheme: str = "llm_judge"
    bleurt_threshold: float = 0.5
    bleurt_checkpoint: str = "models/BLEURT-20-D12"
    llm_judge_model: str = "deepseek/deepseek-v4-pro"  # Novita AI model id


@dataclass
class ModelConfig:
    embed_dim: int = 2048       # E: backbone output per token
    conv1d_layers: int = 2
    lstm_hidden: int = 2048
    lstm_layers: int = 1
    dropout: float = 0.3
    # Component ablation: False -> flatten only / identity / masked mean (conv1d_layers=0 for the conv).
    use_proj: bool = True
    use_lstm: bool = True
    use_attn_pool: bool = True
    # Channels the detector sees, in Q, K, V order; the rest are dropped, not zeroed. null keeps all 3.
    keep_channels: list[str] | None = field(default_factory=lambda: ["Q", "K"])
    # Which rows the detector reads: generated tokens, the prompt's last position (row 0 on disk), or both.
    token_rows: str = "generated"
    # Compress the TOKEN axis to this many buckets by mean-pooling contiguous
    # runs, applied after extract.max_tokens cropping. null keeps every token.
    token_buckets: int | None = None
    # Structure-preservation control: permute the LAYER axis with a fixed
    # seeded permutation, applied to every example. null keeps true order.
    layer_permute_seed: int | None = None
    # Same as layer_permute_seed but for the SEGMENT (M) axis; the two are independent and may combine.
    segment_permute_seed: int | None = None
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
    val_fraction: float = 0.1   # of the non-test examples
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
    data_root: str = "/scratch/ahmedubc/QKV-Lens-data"
    runs_root: str = "runs"

    def example_dir(self, root: str | None = None) -> Path:
        return self.dataset_dir_for(self.dataset.name, self.llm.alias, root=root)

    def pair_dir(self, dataset_name: str, llm_alias: str, root: str | None = None) -> Path:
        return Path(root or self.data_root) / dataset_name / llm_alias

    def dataset_dir_for(self, dataset_name: str, llm_alias: str, root: str | None = None) -> Path:
        return self.pair_dir(dataset_name, llm_alias, root=root) / self.variant_name(llm_alias)

    def variant_name(self, llm_alias: str | None = None) -> str:
        # Checked against the model's own canonical n_segments, so Qwen2.5-7B's baked-in M=32 is "default".
        e = self.extract
        parts = [SOURCE_TAGS[e.source]]
        if e.pool != "mean":
            parts.append(e.pool)
        if e.n_segments is not None and e.n_segments != CANONICAL_N_SEGMENTS.get(llm_alias or self.llm.alias):
            parts.append(f"M{e.n_segments}")
        name = "_".join(parts)
        return DEFAULT_VARIANT if name == "qkv" else name

    def for_variant(self, name: str) -> "Config":
        from src.extract.tensor_ops import POOL_MODES

        tags = {tag: source for source, tag in SOURCE_TAGS.items()}
        source, pool, n_segments = "qkv", "mean", CANONICAL_N_SEGMENTS.get(self.llm.alias)
        tokens = [] if name == DEFAULT_VARIANT else name.split("_")
        if tokens:
            if tokens[0] not in tags:
                raise ValueError(f"variant {name!r} must start with one of {sorted(tags)}")
            source = tags[tokens[0]]
            for tok in tokens[1:]:
                if tok in POOL_MODES:
                    pool = tok
                elif tok[:1] == "M" and tok[1:].isdigit() and int(tok[1:]) > 0:
                    n_segments = int(tok[1:])
                else:
                    raise ValueError(
                        f"variant {name!r}: bad part {tok!r} (pool {POOL_MODES} or M<segments>)"
                    )

        out = copy.deepcopy(self)
        out.extract.source, out.extract.pool, out.extract.n_segments = source, pool, n_segments
        if out.variant_name() != name:
            raise ValueError(f"variant {name!r} is spelled {out.variant_name()!r}")
        return out

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
        if m.token_rows not in VALID_TOKEN_ROWS:
            raise ValueError(f"model.token_rows must be one of {VALID_TOKEN_ROWS}, got {m.token_rows!r}")
        if m.token_buckets is not None and m.token_buckets < 1:
            raise ValueError("model.token_buckets must be >= 1 or null")
        if m.layer_permute_seed is not None and m.layer_permute_seed < 0:
            raise ValueError("model.layer_permute_seed must be >= 0 or null")
        if m.segment_permute_seed is not None and m.segment_permute_seed < 0:
            raise ValueError("model.segment_permute_seed must be >= 0 or null")
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
        if m.collapse_axis == "M" and m.segment_permute_seed is not None:
            raise ValueError(
                "model.collapse_axis='M' and model.segment_permute_seed are "
                "mutually exclusive: collapsing the segment axis to size 1 "
                "leaves nothing for a segment-order permutation to reorder."
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

def _build(raw: dict) -> Config:
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
