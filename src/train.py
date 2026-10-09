# Training loop: train and evaluate the detector on one (dataset, LLM) source.

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.config import Config
from src.data.dataset import (
    N_CHANNELS,
    QKVFieldDataset,
    _set_stats,
    collate,
    compute_stats,
)
from src.models.classifier import build_model
from src.utils.logger import get_logger, setup_logging
from src.utils.metrics import compute_metrics, format_metrics
from src.utils.progress import progress
from src.utils.seed import pick_device, seed_everything
from src.utils.snapshot import snapshot_code
from src.utils.splits import make_split

logger = get_logger(__name__)


def load_source(
    cfg: Config, dataset_name: str, llm_alias: str, field_source: str = "qkv", **kw
) -> QKVFieldDataset:
    # field_source: "qkv" (default, canonical/ablation-cell tree) or
    # "hidden-states" (the representation ablation's alternative field).
    # The hidden-states tree is never a pooling-ablation cell, so it's
    # addressed directly rather than through dataset_dir_for.
    data_root = Path(cfg.data_root)

    if field_source == "hidden-states":
        root = data_root / "hidden_states" / dataset_name / llm_alias
    else:
        # dataset_name/llm_alias may differ from cfg's own (cross-LLM test()
        # evaluates a checkpoint's dataset against another LLM's corpus).
        # extract.pool still comes from cfg.
        root = cfg.dataset_dir_for(dataset_name, llm_alias, root=str(data_root))

    return QKVFieldDataset(
        root,
        max_tokens=cfg.extract.max_tokens,
        origin=f"{llm_alias}/{dataset_name}",
        keep_channels=cfg.model.keep_channels,
        token_buckets=cfg.model.token_buckets,
        layer_permute_seed=cfg.model.layer_permute_seed,
        segment_permute_seed=cfg.model.segment_permute_seed,
        collapse_axis=cfg.model.collapse_axis,
        **kw,
    )


def _unpack_batch(batch, device):
    # Returns (model_args, labels, origins); model_args is the positional-arg
    # tuple QKVHalluDetector.forward expects.
    images, labels, mask, origins = batch
    images = images.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    return (images, mask), labels.to(device, non_blocking=True), origins


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    # One pass; trains if `optimizer` is given, else evaluates. No scheduler
    # arg -- the LR schedule steps once per epoch, driven by the caller.
    training = optimizer is not None
    model.train(training)

    total_loss, n_batches = 0.0, 0
    all_y, all_p = [], []

    with torch.set_grad_enabled(training):
        for batch in progress(loader, desc=desc, leave=False, ncols=100):
            model_args, labels, _origins = _unpack_batch(batch, device)

            logits = model(*model_args)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            all_y.extend(labels.detach().cpu().numpy())
            all_p.extend(torch.sigmoid(logits).detach().cpu().numpy())

    metrics = compute_metrics(all_y, all_p)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


def _section(title: str) -> None:
    logger.info("")
    logger.info("--- %s %s", title, "-" * max(0, 58 - len(title)))


def log_run_header(cfg: Config, run_dir, device, dataset_name: str) -> None:
    _section("run")
    logger.info("run dir      : %s", run_dir)
    logger.info("device       : %s", device)
    logger.info("seed         : %d", cfg.train.seed)

    _section("data")
    logger.info("dataset      : %s", dataset_name)
    logger.info("llm          : %s (%s)", cfg.llm.alias, cfg.llm.name)
    logger.info("max_tokens   : %s | n_segments: %s | l_eff: %s",
                cfg.extract.max_tokens, cfg.extract.n_segments, cfg.extract.l_eff)
    logger.info("labeling     : %s", cfg.labeling.scheme)

    _section("model")
    logger.info("backbone     : %s", cfg.model.backbone)
    logger.info("input        : 1 field per token x %d channels (Q, K, V), "
                "(L, M) spatial -> temporal encoder", N_CHANNELS)
    logger.info("embed_dim    : %d | dropout: %.3g", cfg.model.embed_dim, cfg.model.dropout)
    logger.info("temporal     : conv1d x%d | bilstm hidden=%d x%d layer(s)",
                cfg.model.conv1d_layers, cfg.model.lstm_hidden, cfg.model.lstm_layers)
    if cfg.model.layer_permute_seed is not None:
        logger.info(
            "ABLATION     : layer order permuted, seed=%d (structure-preservation control)",
            cfg.model.layer_permute_seed,
        )

    _section("train")
    logger.info("epochs       : %d | patience: %d | batch_size: %d",
                cfg.train.epochs, cfg.train.patience, cfg.train.batch_size)
    logger.info("lr           : %.3g | weight_decay: %.3g | backbone_lr_scale: %.3g",
                cfg.train.lr, cfg.train.weight_decay, cfg.train.backbone_lr_scale)
    logger.info("balance_class: %s | val_fraction: %.3g | test_fraction: %.3g",
                cfg.train.balance_classes, cfg.train.val_fraction, cfg.train.test_fraction)


def log_param_counts(model) -> None:
    groups = [
        ("backbone", getattr(model, "backbone", None)),
        ("temporal encoder", getattr(model, "temporal", None)),
        ("head", getattr(model, "head", None)),
    ]
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name, mod in groups:
        if mod is None:
            continue
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        logger.info("  %-22s %11s  (%4.1f%%)", name, f"{n:,}", 100 * n / max(total, 1))
    logger.info("  %-22s %11s", "TOTAL trainable", f"{total:,}")


def log_model_repr(model) -> None:
    # Line by line: the log formatter prefixes every record, so one
    # multi-line message would leave all but the first line misaligned.
    for line in repr(model).splitlines():
        logger.info("  %s", line)


def train(cfg: Config, dataset_name: str, run_name: str | None = None) -> dict:
    seed_everything(cfg.train.seed)
    device = pick_device()

    run_dir = Path(cfg.runs_root) / (run_name or default_run_name(cfg, dataset_name))
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot_code(run_dir)
    setup_logging(log_file=run_dir / "train.log")

    log_run_header(cfg, run_dir, device, dataset_name)

    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    # ---- data ---------------------------------------------------------
    full = load_source(cfg, dataset_name, cfg.llm.alias, field_source=cfg.extract.source)
    logger.info(
        "source %-28s n=%-6d hallucination rate=%.1f%%",
        full.origin, len(full), 100 * np.mean(full.labels),
    )

    logger.info(
        "%s has no separate test corpus; carving out a %.0f%% stratified test slice",
        dataset_name, 100 * cfg.train.test_fraction,
    )

    # HalluShift-style 2-way split: val and heldout are the SAME indices.
    train_idx, val_idx, heldout_idx = make_split(
        full.labels,
        val_fraction=cfg.train.val_fraction,
        test_fraction=cfg.train.test_fraction,
        seed=cfg.train.seed,
        cache=run_dir / "split.json",
    )
    logger.info("split        : train %d | val/heldout %d", len(train_idx), len(val_idx))

    geom = full.geometry
    n_rows, n_segments = geom.get("n_rows"), geom.get("n_segments")
    logger.info("field size   : %s rows (L) x %s segments (M)", n_rows, n_segments)

    _section("normalisation (train split only)")
    stats = compute_stats(full, train_idx)
    (run_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    for proj, s in stats.items():
        logger.info("norm %-3s: mean=%+.4f std=%.4f", proj, s["mean"], s["std"])

    _set_stats(full, stats)

    loader_kw = dict(
        batch_size=cfg.train.batch_size,
        collate_fn=collate,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
    )
    # drop_last on TRAIN only: BatchNorm1d can't compute variance over a
    # batch of size 1 (a trailing remainder batch when len % batch_size == 1).
    # Val/test must never drop data.
    train_loader = DataLoader(
        Subset(full, train_idx), shuffle=True, drop_last=True, **loader_kw
    )
    val_loader = DataLoader(Subset(full, val_idx), shuffle=False, **loader_kw)

    # ---- model --------------------------------------------------------
    # collapse_axis shrinks L or M to 1 at data-loading time; geometry.json's
    # n_rows/n_segments are extraction-time numbers and don't reflect that,
    # so field_shape is overridden here to match what the dataset hands the
    # model (else an eagerly-built backbone gets the wrong input width).
    if cfg.model.collapse_axis == "L":
        n_rows = 1
    elif cfg.model.collapse_axis == "M":
        n_segments = 1
    field_shape = (n_rows, n_segments) if n_rows and n_segments else None
    in_ch = full.n_channels
    model = build_model(cfg, field_shape=field_shape, in_ch=in_ch).to(device)

    _section("architecture")
    log_model_repr(model)

    _section("parameters")
    log_param_counts(model)

    _section("optimisation")
    pos_weight = None
    if cfg.train.balance_classes:
        y = np.asarray([full.labels[i] for i in train_idx])
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        if n_pos > 0 and n_neg > 0:
            # float32 explicitly: numpy int division yields float64, which
            # MPS refuses and which would silently upcast the loss on CUDA.
            pos_weight = torch.tensor(
                [n_neg / n_pos], dtype=torch.float32, device=device
            )
            logger.info("pos_weight = %.3f (neg=%d, pos=%d)", pos_weight.item(), n_neg, n_pos)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Discriminative LR: backbone gets lr * backbone_lr_scale, everything
    # else gets the full lr. scale == 1.0 collapses to a single group.
    scale = cfg.train.backbone_lr_scale
    backbone_lr = cfg.train.lr * scale
    if scale == 1.0:
        param_groups = [{"params": model.parameters(), "lr": cfg.train.lr}]
    else:
        backbone_params = list(model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": other_params, "lr": cfg.train.lr},
        ]
        logger.info(
            "discriminative LR: backbone=%.2e, rest=%.2e (scale=%.3g)",
            backbone_lr, cfg.train.lr, scale,
        )

    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)

    # Linear decay (paper §5.3): flat for lr_decay_start epochs, then ramp
    # to lr_final_scale x initial by the last epoch. Stepped once per EPOCH,
    # not per batch -- deliberately not passed into run_epoch(). If early
    # stopping fires before `epochs`, the LR simply never reaches the floor.
    warm = cfg.train.lr_decay_start
    final = cfg.train.lr_final_scale
    total = cfg.train.epochs

    def lr_lambda(epoch: int) -> float:      # epoch is 0-based
        if epoch < warm:
            return 1.0
        span = max(1, total - warm)
        frac = min(1.0, (epoch - warm) / span)
        return 1.0 + frac * (final - 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    logger.info(
        "linear LR decay: flat for %d epochs, then -> %.3g x initial by epoch %d",
        warm, final, total,
    )

    # ---- loop ---------------------------------------------------------
    _section("training")
    best_auroc, best_epoch, stale = -1.0, -1, 0
    history = []

    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, criterion, device,
                       optimizer, desc=f"epoch {epoch} train")
        va = run_epoch(model, val_loader, criterion, device,
                       desc=f"epoch {epoch} val")

        lrs = [g["lr"] for g in optimizer.param_groups]
        logger.info("epoch %2d | train loss %.4f AUROC %.4f | val %s | lr %s | %.0fs",
                    epoch, tr["loss"], tr["auroc"], format_metrics(va),
                    " ".join(f"{lr:.2e}" for lr in lrs), time.time() - t0)

        scheduler.step()

        record = {"epoch": epoch, "train": tr, "val": va, "lr": lrs}

        if va["auroc"] > best_auroc:
            best_auroc, best_epoch, stale = va["auroc"], epoch, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "stats": stats,
                    "epoch": epoch,
                    "val_auroc": va["auroc"],
                    "train_datasets": [dataset_name],
                    "heldout_idx": heldout_idx,
                    "llm_alias": cfg.llm.alias,
                    "field_shape": field_shape,
                    "in_ch": in_ch,
                },
                run_dir / "best.pt",
            )
            logger.info("  new best (val AUROC %.4f) -> saved", best_auroc)
        else:
            stale += 1

        history.append(record)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

        if stale >= cfg.train.patience:
            logger.info("early stopping: no val improvement for %d epochs", stale)
            break

    # ---- final evaluation with the BEST checkpoint ---------------------
    _section("final evaluation")
    logger.info("loading best checkpoint (epoch %d, val AUROC %.4f)", best_epoch, best_auroc)
    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    results = {"best_epoch": best_epoch, "val_auroc": best_auroc}

    va = run_epoch(model, val_loader, criterion, device, desc="final val")
    results["val"] = va
    logger.info("FINAL val  | %s", format_metrics(va))

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("results written to %s", run_dir / "results.json")
    return results


def default_run_name(cfg: Config, dataset_name: str) -> str:
    return f"{cfg.llm.alias}_{dataset_name}_{cfg.model.backbone}"
