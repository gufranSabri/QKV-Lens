"""Config loading: YAML -> validated dataclasses, with deep-merge over defaults.

WHAT IS AND IS NOT CONFIGURABLE
-------------------------------
This is the QKV-Steer codebase. It inherits the QKV-Lens detector, but not the
QKV-Lens ablation surface: every option that only existed to produce an ablation
row in that paper has been removed, and the method's own settings are now fixed
in code rather than re-selected per run.

Fixed, no longer a config key (QKV-Lens paper §5.3 and Table 4):

    feature field    (T, L, M, 3), trailing axis = (Q, K, V)   [Alg. 1]
    detector input   ONE image per token, 3 channels           [Alg. 1]
    decoding         greedy                                    [§5.3]

Those are not knobs because QKV-Steer's whole premise is that the detector's
attribution addresses real (layer, segment, projection) coordinates. Changing
the channel semantics would change what a coordinate means, and the steering
stage would be writing into a different space than the one the detector
looked at.

`extract.pool` (mean/max/strided, QKV-Lens Table 3) is the one exception:
`mean` is still THE fixed, canonical setting -- the only one steering may run
against -- but the pooling ablation itself was re-added as an opt-in, since
choosing a different pool changes what each segment covers (a value, not a
coordinate), not which coordinates exist. A non-default pool writes to its own
`data_root` subtree (see Config.dataset_dir_for) precisely so it can never be
mistaken for, or silently shadow, the canonical corpus.

What remains configurable is what genuinely varies across runs: which LLM, which
dataset, how many segments M, how many tokens, the pooling ablation, and the
detector's own capacity and optimisation settings.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

#: "flat_mlp" is the main approach; "scratch_cnn" (BAFE, the original QKV-Lens
#: CNN) is the structure-preservation ablation's "with spatial structure" arm
#: -- see src/models/backbones/flat_mlp.py's docstring.
VALID_BACKBONES = ("flat_mlp", "scratch_cnn")
VALID_SCHEMES = ("exact_match", "bleurt")
#: extract.source: which field train/test/cam/forecasting load -- "qkv" (the
#: paper's (T,L,M,3) field) or "hidden-states" ((T,L,M,1), the representation
#: ablation's QKV alternative). See ExtractConfig.source's docstring.
VALID_SOURCES = ("qkv", "hidden-states")
#: model.collapse_axis: which field axis the feature-pooling ablation
#: collapses by averaging -- see ModelConfig.collapse_axis's docstring.
VALID_COLLAPSE_AXES = ("M", "L", "channels")

#: Per-model n_segments that is CANONICAL for that model -- i.e. baked into
#: its own configs/{dataset}/{alias}.yaml for a structural reason, not an
#: ablation override. Only Qwen2.5-7B needs an entry: its L=28 does not
#: divide its GQA D_kv=512, so `configs/*/qwen2.5_7b.yaml` sets
#: extract.n_segments=32 permanently (see that file's own comment). Every
#: other model's canonical value is `null` (-> the model's own layer count,
#: resolved at extraction time -- see run_extraction.py), so it never appears
#: here. `dataset_dir_for` consults this to tell "this model's one true
#: n_segments" apart from "a pooling-ablation sweep's non-canonical M" --
#: without it, Qwen2.5-7B's real, only-ever-used corpus would misclassify as
#: an ablation cell and route to a pool_mean_ncols32/ subtree nothing ever
#: extracts into. Add an entry here, not a special case in dataset_dir_for,
#: if a future model config needs the same treatment.
CANONICAL_N_SEGMENTS: dict[str, int] = {
    "qwen2.5_7b": 32,
}


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
    """How the feature field is built and stored.

    `extraction_type` (delta|transforms) stays gone -- those DWT/delta channel
    variants competed for the same channel axis Q/K/V occupies, with no way to
    coexist with it. `source` is back (QKV-Lens removed it as a settled
    question: hidden states lost to QKV in that paper's Table 4), reopened by
    the structure-preservation pivot -- see `source`'s own docstring below.
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
    # Segment-pooling strategy: one of tensor_ops.POOL_MODES. `mean` (default)
    # is the paper's fixed setting and the only mode steering may run against
    # (see src/extract/tensor_ops.py POOL_MODES). `max`/`strided` exist ONLY
    # to reproduce the pooling ablation (QKV-Lens Table 3) -- a non-default
    # value writes to its OWN data_root subtree (see Config.example_dir), so
    # it can never silently overwrite the canonical mean-pooled corpus a
    # trained checkpoint or steering run depends on.
    pool: str = "mean"
    # Which field the REPRESENTATION-ablation TRAIN/TEST side reads: "qkv"
    # (default, the paper's (T,L,M,3) field) or "hidden-states" (a (T,L,M,1)
    # field of decoder hidden states -- see
    # src/extract/run_extraction.run_hidden_states_extraction). NOT read by
    # `extract` itself -- extraction always writes the QKV field; a
    # hidden-states corpus is produced by `extract-hidden-states`
    # (detector.py), a SEPARATE command, into its own data_root subtree, and
    # never overwrites or is read in place of the QKV corpus. This key only
    # tells `train`/`test`/`cam`/forecasting which of the two ALREADY-WRITTEN
    # trees to load from (see src/train.py load_source's `source` argument).
    source: str = "qkv"


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

    backbone: str = "flat_mlp"
    embed_dim: int = 2048       # E: backbone output per token
    conv1d_layers: int = 2
    lstm_hidden: int = 2048
    lstm_layers: int = 1
    dropout: float = 0.3
    # Which of the field's 3 fixed channels (Q, K, V) actually reach the
    # detector, e.g. ["Q"] or ["Q", "V"]. `null` (default) keeps all three.
    # The FIELD is unchanged either way -- (T, L, M, 3) is still what gets
    # extracted and stored -- this only zeroes the channels NOT listed at
    # data-loading time, so the CNN's input shape (and every steering
    # coordinate) stays fixed at 3 channels. This is an ablation over the
    # detector's input, not a revival of the removed `model.channels`/
    # `include` machinery: there is still exactly one image stream, one CNN.
    keep_channels: list[str] | None = None
    # Compress the TOKEN axis to this many buckets by mean-pooling contiguous
    # runs of generated tokens, applied AFTER extract.max_tokens cropping (see
    # QKVFieldDataset._load_raw). `null` (default) keeps every token as its
    # own step. Ablation-only, like `keep_channels` -- does not change what a
    # (layer, segment, projection) coordinate means, only how many token
    # positions the temporal encoder sees.
    token_buckets: int | None = None
    # Structure-preservation ablation (control #2): permute the LAYER axis with
    # a FIXED pseudo-random permutation, seeded by this value, applied to every
    # example (train and test alike) and computed once per dataset load -- not
    # re-drawn per example, or the model could not learn any layer-order
    # regularity at all, defeating the point of the control. `null` (default)
    # keeps the true layer order. If structured (correctly-ordered) beats this
    # at otherwise-identical capacity, cross-layer adjacency/order itself is
    # informative, not just "the detector sees every layer". Ablation-only, like
    # `keep_channels`/`token_buckets`: ANY non-null value breaks the
    # (layer, segment, projection) coordinate correspondence the steering stage
    # depends on, so this must stay null outside this one ablation.
    layer_permute_seed: int | None = None
    # Feature-pooling ablation: collapse ONE field axis to size 1 by averaging
    # over it, at data-loading time -- "vector aggregation", the compressed
    # alternative the paper's related work argues the 2D/3-channel field
    # avoids. One of "M" (segments -> (T,L,1,3): keep depth + projection,
    # lose feature resolution), "L" (layers -> (T,1,M,3): keep feature
    # resolution + projection, lose depth -- this is literally "pool the
    # whole field to one vector per projection", the paper's own phrase for
    # the alternative it argues against), "channels" (Q/K/V -> (T,L,M,1):
    # keep depth + feature resolution, lose projection identity -- the
    # single-channel case this shares machinery with hidden-states over, see
    # QKVFieldDataset.n_channels). `null` (default) collapses nothing.
    # Mutually exclusive with layer_permute_seed (collapsing L makes
    # permuting it meaningless) -- see the validator.
    collapse_axis: str | None = None


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
        """`dataset_dir_for` for THIS config's own (dataset, llm) -- see there."""
        return self.dataset_dir_for(self.dataset.name, self.llm.alias, root=root)

    def dataset_dir_for(self, dataset_name: str, llm_alias: str, root: str | None = None) -> Path:
        """Where a (dataset, LLM)'s feature fields live, under THIS config's
        `extract.pool` / `extract.n_segments`:

            {data_root}/{dataset}/{llm_alias}/                              (canonical pool + n_segments)
            {data_root}/pool_{pool}[_ncols{n_segments}]/{dataset}/{llm_alias}/  (an ablation cell)

        Takes an explicit (dataset_name, llm_alias) rather than always reading
        `self.dataset.name`/`self.llm.alias`, because some callers resolve a
        DIFFERENT dataset/LLM than the config's own under the config's pool
        setting -- e.g. a cross-LLM `test` evaluates the checkpoint's dataset
        against another LLM's corpus (see src/train.py load_source).

        The old {source}/{extraction_type} path levels are gone with the
        options that produced them -- there is one field per (dataset, LLM) at
        the canonical path now. The `pool_{mode}[_ncols{M}]/` prefix ONLY
        appears for an actual pooling-ablation cell (QKV-Lens Table 3) -- this
        keeps every existing corpus's path unchanged, and guarantees an
        ablation extraction can never land on top of (or be silently read as)
        the canonical corpus a checkpoint or steering run depends on.

        "Canonical" is `pool == "mean"` AND `n_segments` matching THIS model's
        own canonical value: `null` for every model except the ones listed in
        `CANONICAL_N_SEGMENTS` (currently only Qwen2.5-7B, whose config
        permanently sets n_segments=32 for a structural GQA-divisibility
        reason -- see that constant's docstring). Checking against the
        MODEL's registered value, not merely "is n_segments None", matters
        for two reasons pulling in opposite directions: (1) Qwen2.5-7B's
        n_segments=32 is baked into its own config file, not a sweep
        override, and must resolve to the bare canonical path or every
        Qwen2.5-7B run 404s against a pool_mean_ncols32/ subtree nothing
        extracts into; (2) the OTHER three models' non-32 M cells in a
        pooling-ablation sweep (M=16/64/128 at pool=mean) must NOT collapse
        onto the bare canonical path either, or they silently collide into
        one directory and only the first extraction for that pool mode is
        ever real -- which is the bug this whole scheme exists to prevent.
        """
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
