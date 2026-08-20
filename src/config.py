"""Config loading: YAML -> validated dataclasses, with deep-merge over defaults.

WHAT IS AND IS NOT CONFIGURABLE
-------------------------------
This is the QKV-Steer codebase. It inherits the QKV-Lens detector, but not the
QKV-Lens ablation surface: every option that only existed to produce an ablation
row in that paper has been removed, and the method's own settings are now fixed
in code rather than re-selected per run.

Fixed, no longer a config key (QKV-Lens paper §5.3 and Tables 3-4):

    feature field    (T, L, M, 3), trailing axis = (Q, K, V)   [Alg. 1]
    pooling          mean over M contiguous segments           [Table 3]
    detector input   ONE image per token, 3 channels           [Alg. 1]
    decoding         greedy                                    [§5.3]

Those are not knobs because QKV-Steer's whole premise is that the detector's
attribution addresses real (layer, segment, projection) coordinates. Changing
the pooling rule or the channel semantics would change what a coordinate means,
and the steering stage would be writing into a different space than the one the
detector looked at.

What remains configurable is what genuinely varies across runs: which LLM, which
dataset, how many segments M, how many tokens, and the detector's own capacity
and optimisation settings.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

VALID_BACKBONES = ("scratch_cnn", "resnet18")
VALID_SCHEMES = ("exact_match", "bleurt")


@dataclass
class LLMConfig:
    name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    dtype: str = "bfloat16"
    # Short alias used in output paths (avoids '/' in directory names).
    alias: str = "llama3_8b"


@dataclass
class DatasetConfig:
    name: str = "triviaqa"
    n_samples: int = 10000
    max_new_tokens: int = 64
    prompt_template: str = "Answer the question concisely. Q: {question} A:"


@dataclass
class ExtractConfig:
    """How the (T, L, M, 3) feature field is built and stored.

    There is exactly one thing to extract now -- the paper's QKV field. The old
    `source` (qkv|hs) and `extraction_type` (delta|transforms) selectors are
    gone: hidden states were a QKV-Lens ablation baseline (Table 4's HS column,
    which lost to QKV on all four models), and the delta/DWT channel variants
    competed for the same channel axis that the paper reserves for Q/K/V.
    """

    dtype: str = "float16"
    max_tokens: int = 100
    # Examples generated together per batch. >1 left-pads prompts to the
    # batch's longest one and tracks per-sequence EOS -- purely a throughput
    # knob, produces the same tensors as batch_size=1 (see qkv_hooks.py).
    batch_size: int = 8
    # M: number of pooled feature segments per layer. `null` -> use the model's
    # layer count, which makes the field square. The paper uses M=32 (§5.3),
    # which is also what `null` yields for every 32-layer model here.
    n_segments: int | None = None
    # Pool the LAYER axis to this many rows. `null` -> keep the model's L.
    # Only needed for cross-LLM work, where Llama (32) and Qwen (28) differ.
    # Must stay null for steering: the layer remap is not invertible.
    l_eff: int | None = None


@dataclass
class LabelingConfig:
    scheme: str = "bleurt"
    bleurt_threshold: float = 0.5
    bleurt_checkpoint: str = "models/BLEURT-20-D12"


@dataclass
class ModelConfig:
    """The detector f_theta.

    `channels`, `include`, `fusion` and `share_backbone` are gone. Under the
    paper's layout each token is ONE 3-channel image, so there was never more
    than one CNN stream to regroup, subset, fuse, or share -- `build_fusion`
    returned IdentityFusion unconditionally and the fusion modules were dead
    code. See src/models/ for what the detector is now.
    """

    backbone: str = "scratch_cnn"
    embed_dim: int = 2048       # E: CNN output per token
    conv1d_layers: int = 2
    lstm_hidden: int = 2048
    lstm_layers: int = 1
    dropout: float = 0.3
    pretrained_backbone: bool = True   # only meaningful for resnet18


@dataclass
class TrainConfig:
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    # A pretrained backbone needs a gentler LR than the randomly-initialised
    # temporal/head, or the early high-LR steps wreck its ImageNet features
    # before the head stabilises. Its LR is `lr * backbone_lr_scale`.
    # 1.0 = single LR for everything (correct for scratch / random-init).
    backbone_lr_scale: float = 0.1
    # Linear LR decay, matching the paper (§5.3): hold the LR flat for the
    # first `lr_decay_start` epochs, then ramp linearly down to
    # `lr_final_scale` x the initial LR by `epochs`.
    lr_decay_start: int = 5
    lr_final_scale: float = 0.0
    epochs: int = 20
    patience: int = 8
    seed: int = 42
    num_workers: int = 4
    # The paper's 75/25 stratified split, with the held-out slice doubling as
    # the early-stopping signal (see src/utils/splits.py).
    val_fraction: float = 0.25
    test_fraction: float = 0.25
    # Weight the positive (hallucination) class to counter imbalance.
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

    # ---- derived paths -------------------------------------------------
    def example_dir(self, root: str | None = None) -> Path:
        """Where this (dataset, LLM)'s feature fields live:

            {data_root}/{dataset}/{llm_alias}/

        The old {source}/{extraction_type} path levels are gone with the
        options that produced them -- there is one field per (dataset, LLM) now.
        """
        return (
            Path(root or self.data_root)
            / self.dataset.name
            / self.llm.alias
        )

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
        # Decay would never begin: the run ends before the flat phase does.
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
    """Recursively merge `override` into `base`, returning a new dict."""
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

#: Keys that existed in QKV-Lens and are deliberately gone. Naming them in the
#: error beats a bare "unknown key": every one of these appears in an inherited
#: config or shell script, and silently ignoring them would let a run proceed
#: under settings the user believes are in effect.
_REMOVED_KEYS: dict[str, str] = {
    "extract.source": "hidden states were a QKV-Lens ablation baseline; only the QKV field remains",
    "extract.extraction_type": "delta/transform channels competed for the channel axis, which now holds Q/K/V",
    "extract.views": "Q, K and V are always all three, as the field's channel axis",
    "extract.pool": "mean pooling is fixed (QKV-Lens Table 3)",
    "extract.boundary_mode": "only meaningful for the removed delta channels",
    "extract.n_cols": "renamed to extract.n_segments (the paper's M)",
    "model.channels": "each token is one 3-channel image; there is nothing to regroup",
    "model.include": "there is only one image stream to keep",
    "model.fusion": "a single stream never had anything to fuse (build_fusion always returned identity)",
    "model.share_backbone": "there is only one backbone",
    "model.fused_dim": "no fusion stage; the CNN's embed_dim feeds the temporal encoder directly",
}


def _check_removed(raw: dict) -> None:
    """Fail loudly on any QKV-Lens-era key, with the reason it is gone."""
    found = []
    for dotted, why in _REMOVED_KEYS.items():
        section, key = dotted.split(".")
        if isinstance(raw.get(section), dict) and key in raw[section]:
            found.append(f"  {dotted}: {why}")
    if found:
        raise ValueError(
            "config uses option(s) removed in QKV-Steer:\n"
            + "\n".join(found)
            + "\n\nThese were QKV-Lens ablation axes. See src/config.py's module "
              "docstring for what is fixed and why."
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
    """Load `path`, layered over configs/default.yaml, then apply `overrides`."""
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
