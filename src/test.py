"""Evaluate a saved detector checkpoint on a dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data.dataset import collate, compute_stats
from src.models.classifier import build_model
from src.train import load_source, run_epoch
from src.utils.logger import get_logger
from src.utils.metrics import format_metrics
from src.utils.seed import pick_device, seed_everything

logger = get_logger(__name__)


def test(
    cfg: Config,
    checkpoint: str | Path,
    dataset_name: str | None = None,
    recompute_stats: bool = False,
    out_name: str | None = None,
) -> dict:
    seed_everything(cfg.train.seed)
    device = pick_device()

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats = ckpt["stats"]

    # Cross-LLM eval: cfg.llm.alias is whatever --config points at, which may
    # differ from the LLM this checkpoint was actually trained on. Surface this
    # loudly and unconditionally -- it silently changes what "in-distribution"
    # even means below (see _resolve_eval_target).
    ckpt_llm_alias = ckpt.get("llm_alias")
    if ckpt_llm_alias is not None and ckpt_llm_alias != cfg.llm.alias:
        logger.warning(
            "cross-LLM eval: checkpoint was trained on llm=%r, evaluating "
            "against llm=%r's data. Normalization stats are %s.",
            ckpt_llm_alias, cfg.llm.alias,
            "recomputed from the target LLM's data" if recompute_stats
            else f"the CHECKPOINT's (llm={ckpt_llm_alias!r}'s), force-applied "
                 f"to llm={cfg.llm.alias!r}'s activations",
        )

    # layer_permute_seed changes what INPUT the model was trained to read --
    # unlike a wrong backbone choice, which fails to load_state_dict, a
    # mismatched (or missing) permutation here loads fine and just silently
    # evaluates the model on a different layer ordering than it was trained
    # on, undermining the whole point of this ablation.
    ckpt_seed = (ckpt.get("config") or {}).get("model", {}).get("layer_permute_seed")
    if ckpt_seed != cfg.model.layer_permute_seed:
        logger.warning(
            "layer_permute_seed mismatch: checkpoint was trained with %r, this "
            "eval config has %r. Pass --set model.layer_permute_seed=%r to "
            "match the checkpoint, or this result is not a faithful eval of it.",
            ckpt_seed, cfg.model.layer_permute_seed, ckpt_seed,
        )

    name = dataset_name or cfg.dataset.name
    name, eval_set = _resolve_eval_target(name, ckpt, cfg.llm.alias)

    source = load_source(cfg, name, cfg.llm.alias)
    if recompute_stats:
        # Explicit opt-in: normalise with statistics computed fresh from the
        # TARGET corpus, instead of the checkpoint's training-LLM statistics.
        # Meaningful only for a genuine cross-LLM eval.
        logger.info(
            "recomputing normalization stats from %s (recompute_stats=True)",
            source.origin,
        )
        stats = compute_stats(source, list(range(len(source))))
    # Normalise with the TRAINING statistics baked into the checkpoint by
    # default, never with statistics silently recomputed on the test set.
    source.stats = stats

    # Restrict to the held-out rows when the model was trained on this corpus.
    # Skipping this is what would make every same-dataset score a train score.
    eval_data = source
    if eval_set is not None:
        if not eval_set:
            raise ValueError(
                f"checkpoint was trained on {name!r} but carries no held-out indices. "
                "Retrain before testing on it."
            )
        eval_data = Subset(source, eval_set)

    logger.info("evaluating on %s (n=%d of %d)", source.origin, len(eval_data), len(source))

    loader = DataLoader(
        eval_data,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(cfg, field_shape=ckpt.get("field_shape")).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    criterion = torch.nn.BCEWithLogitsLoss()
    metrics = run_epoch(model, loader, criterion, device, desc="test")

    logger.info("TEST %s | %s", source.origin, format_metrics(metrics))

    out = {"dataset": name, "checkpoint": str(ckpt_path), "metrics": metrics}

    dest = ckpt_path.parent / f"test_{out_name or name}.json"
    dest.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", dest)

    return out


def _resolve_eval_target(
    name: str, ckpt: dict, llm_alias: str
) -> tuple[str, list[int] | None]:
    """Pick what to actually evaluate on. Returns (corpus_name, row_subset).

    The rule, in order:

    1. We trained on it, on THIS SAME LLM -> evaluate the stratified slice held
       out at train time. (Every dataset mirrors HalluShift, which never
       trains/tests on separate corpora.)
    2. Anything else (a different dataset, OR the same dataset name but a
       DIFFERENT LLM) -> zero-shot; evaluate the full corpus. `heldout_idx` was
       computed against the CHECKPOINT's LLM's manifest -- reusing it against a
       different LLM's manifest would restrict the eval to index positions with
       no real meaning for that LLM's data, so a dataset-name match alone is
       not enough; the LLM must match too.

    A `row_subset` of None means "use the whole corpus".
    """
    trained_on = set(ckpt.get("train_datasets", []))
    ckpt_llm_alias = ckpt.get("llm_alias")
    same_llm = ckpt_llm_alias is None or ckpt_llm_alias == llm_alias

    if name not in trained_on or not same_llm:
        reason = "different LLM" if (name in trained_on and not same_llm) else "unseen dataset"
        logger.info(
            "%s (llm=%s) was not in this checkpoint's training set (%s): zero-shot eval",
            name, llm_alias, reason,
        )
        return name, None

    logger.info("%s has no separate test corpus; evaluating its held-out slice", name)
    return name, list(ckpt.get("heldout_idx") or [])
