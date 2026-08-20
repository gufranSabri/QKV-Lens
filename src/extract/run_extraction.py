"""Orchestrates feature extraction: generate -> capture Q/K/V -> build field -> save.

One `extract` run generates each example ONCE (a manual decode loop per batch,
see qkv_hooks.capture_all), builds the paper's (T, L, M, 3) QKV feature field
from the captured projections, and writes it.

QKV-Lens wrote FOUR trees per (dataset, LLM) -- every (source, extraction_type)
combination -- because those were ablation axes. QKV-Steer has one field, so
there is one tree:

    {data_root}/{dataset}/{llm_alias}/
        00000/
            tokens.npy      (T, L, M, 3) float16
            meta.txt        human-readable prompt / response / gold / score / label
        00001/
        ...
        manifest.jsonl      one JSON line per example (the training index)
        geometry.json       the model geometry the fields were built with
        progress.log        "i/total" appended every 100 generated examples

Extraction is the expensive step, so it is restartable: an already-complete
example is skipped unless --overwrite is passed. When EVERY requested example
is already complete, this is detected from manifest.jsonl alone -- before
load_examples() (may hit the network/HF hub) or load_llm() (loads the whole
model onto a GPU) ever run -- so re-invoking `extract` on an already-finished
(dataset, LLM) is cheap, not just correct.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.extract.datasets import load_examples
from src.extract.qkv_hooks import VIEWS, capture_all, left_pad_batch, read_geometry
from src.extract.tensor_ops import PROJECTIONS, build_feature_field, pool_layer_axis
from src.label.registry import label_examples
from src.utils.logger import get_logger
from src.utils.progress import progress

logger = get_logger(__name__)

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}


def load_llm(cfg: Config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading %s", cfg.llm.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.name)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm.name,
        dtype=DTYPES[cfg.llm.dtype],
        device_map="auto",
        # Q/K/V come from forward hooks on the projection Linears, so the
        # attention kernel choice does not affect what we capture.
    )
    model.eval()
    return model, tokenizer


def resolve_stop_tokens(tokenizer, model) -> list[int]:
    """Every id that should terminate generation.

    `tokenizer.eos_token_id` only, matching HalluShift exactly (it passes
    `pad_token_id=tokenizer.eos_token_id` and nothing else). No chat-template
    end-of-turn ids, no base-model newline heuristic -- those made generations
    diverge from HalluShift's.
    """
    ids: set[int] = set()

    for source in (tokenizer.eos_token_id, model.config.eos_token_id):
        if source is None:
            continue
        if isinstance(source, (list, tuple)):
            ids.update(int(i) for i in source)
        else:
            ids.add(int(source))

    return sorted(ids)


def build_prompt_ids(prompt: str, tokenizer, device) -> torch.Tensor:
    """Tokenise a prompt to a (1, prompt_len) LongTensor of input ids.

    Raw text for every model, instruct or not -- HalluShift never applies a
    chat template, even for instruct checkpoints, and reproducing its numbers
    means reproducing that choice, not "fixing" it.

    `tokenizer(...)` may hand back either a bare tensor or a dict-like
    BatchEncoding depending on the transformers version, so we normalise rather
    than assuming. Getting a BatchEncoding where a tensor was expected fails
    later and confusingly, at `.ndim`.
    """
    out = tokenizer(prompt, return_tensors="pt")

    # Normalise BatchEncoding / dict -> tensor.
    if not isinstance(out, torch.Tensor):
        out = out["input_ids"]

    if out.ndim == 1:
        out = out.unsqueeze(0)

    return out.to(device)


def format_meta(prompt: str, response: str, gold, score: float, label: int) -> str:
    return (
        f"prompt: {prompt}\n"
        f"response: {response}\n"
        f"gold: {gold}\n"
        f"score: {score}\n"
        f"label: {label}\n"
    )


def write_example(
    out_dir: Path,
    field: torch.Tensor,
    prompt: str,
    response: str,
    gold,
    score: float,
    label: int,
    save_dtype: torch.dtype,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "tokens.npy", field.to(save_dtype).numpy())

    (out_dir / "meta.txt").write_text(
        format_meta(prompt, response, gold, score, label), encoding="utf-8"
    )


def is_complete(out_dir: Path) -> bool:
    """True if this example has both its tensor AND a resolved label.

    An example is only safe to skip on resume when it is fully finished.
    """
    if not (out_dir / "tokens.npy").exists():
        return False
    meta_path = out_dir / "meta.txt"
    if not meta_path.exists():
        return False
    try:
        label = parse_meta(meta_path).get("label", "").strip()
        return label in ("0", "1")
    except OSError:
        return False


def is_generated(out_dir: Path) -> bool:
    """True if the expensive GENERATION step is already done for this example.

    That means both the field tensor and a meta.txt carrying the response/gold
    exist -- regardless of whether a label has been resolved yet. Such an
    example never needs the model again: it only needs labeling + a manifest
    entry, which run_extraction reconstructs from meta.txt.
    """
    if not (out_dir / "tokens.npy").exists():
        return False
    meta_path = out_dir / "meta.txt"
    if not meta_path.exists():
        return False
    try:
        meta = parse_meta(meta_path)
    except OSError:
        return False
    # A response key must be present (it may be empty text, but the field
    # exists once generation wrote meta). gold is needed to label.
    return "response" in meta and "gold" in meta


def _record_from_meta(out_dir: Path, idx: int) -> dict:
    """Rebuild the in-memory record for an already-generated example from disk.

    Mirrors the dict appended during generation, so labeling and the manifest
    treat a reused example identically to a freshly-generated one. n_tokens is
    read from the stored tensor's first axis without loading the whole array.
    """
    meta = parse_meta(out_dir / "meta.txt")
    # mmap so we read only the header, not the full tensor, for the token count.
    n_tokens = int(np.load(out_dir / "tokens.npy", mmap_mode="r").shape[0])
    return {
        "idx": idx,
        "dir": out_dir.name,
        "n_tokens": n_tokens,
        "prompt": meta.get("prompt", ""),
        "response": meta.get("response", ""),
        "gold": meta.get("gold", ""),
    }


def _manifest_complete_indices(root: Path) -> set[int]:
    """Indices manifest.jsonl already records as fully labeled, for `root`.

    Reads ONLY manifest.jsonl (no per-example file opens, no dataset download)
    -- this is what lets `run_extraction` check "is everything already done"
    before paying for `load_examples`/`load_llm`.
    """
    path = root / "manifest.jsonl"
    if not path.exists():
        return set()
    done: set[int] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = str(rec.get("label", "")).strip()
            if label in ("0", "1"):
                done.add(rec["idx"])
    return done


def run_extraction(cfg: Config, chunk: int | None = None, overwrite: bool = False) -> None:
    """Generate responses and write the QKV feature field for one dataset+LLM."""
    root = cfg.example_dir()
    root.mkdir(parents=True, exist_ok=True)

    # Fastest possible pre-check, BEFORE load_examples()/load_llm() AND before
    # the slower per-example is_complete()/is_generated() scan further below:
    # does manifest.jsonl already exist? write_manifest() is the LAST thing a
    # run finishes, so its existence alone (one file-existence check, no
    # contents read) is treated as decisive proof a prior non-chunked run
    # completed. Deliberately NOT stricter than this (e.g. does not verify the
    # record count matches n_samples): an interrupted run whose manifest.jsonl
    # exists but is short needs --overwrite to redo.
    #
    # Chunked runs share one manifest.jsonl across chunks, so this check alone
    # can't tell whether THIS chunk's range is done -- chunked calls use the
    # (slower, but chunk-aware) index-set check instead.
    if not overwrite and chunk is None:
        if (root / "manifest.jsonl").exists():
            logger.info(
                "%s/%s: manifest.jsonl already exists -- treating as already "
                "extracted and skipping load_examples/load_llm entirely "
                "(use --overwrite to redo)",
                cfg.dataset.name, cfg.llm.alias,
            )
            return
    elif not overwrite and chunk is not None:
        lo, hi = (chunk - 1) * 1000, chunk * 1000
        wanted = set(range(lo, hi))
        if wanted and wanted <= _manifest_complete_indices(root):
            logger.info(
                "%s/%s chunk %d: all %d requested examples already complete -- "
                "skipping load_examples/load_llm entirely (use --overwrite to redo)",
                cfg.dataset.name, cfg.llm.alias, chunk, len(wanted),
            )
            return

    examples = load_examples(cfg)
    logger.info("loaded %d examples for %s", len(examples), cfg.dataset.name)

    # Chunking mirrors ACT-ViT: 1-indexed blocks of 1000, so a long extraction
    # can be split across machines and resumed.
    if chunk is not None:
        lo, hi = (chunk - 1) * 1000, chunk * 1000
        examples = [e for e in examples if lo <= e.idx < hi]
        logger.info("chunk %d -> %d examples (idx %d..%d)", chunk, len(examples), lo, hi - 1)
        if not examples:
            logger.warning("chunk %d is empty; nothing to do", chunk)
            return

    # Partition the work BEFORE touching the GPU. Generation is the only step
    # that needs the LLM; labeling + manifest do not.
    #   complete    -> already labeled; skip entirely.
    #   generated   -> tensor + response on disk but unlabeled; REUSE it (no
    #                  LLM), rebuild its record from meta.txt, let labeling
    #                  finish it.
    #   to_generate -> needs the model.
    # This is what lets a run whose generation finished but crashed before
    # labeling pick up at the post-generation step, without re-running the
    # model on 10k prompts.
    reused: list[dict] = []
    to_generate = []
    skipped = 0
    for ex in examples:
        ex_dir = root / f"{ex.idx:05d}"
        if not overwrite and is_complete(ex_dir):
            skipped += 1
        elif not overwrite and is_generated(ex_dir):
            reused.append(_record_from_meta(ex_dir, ex.idx))
        else:
            to_generate.append(ex)

    if skipped:
        logger.info("skipped %d already-complete examples (use --overwrite to redo)", skipped)
    if reused:
        logger.info(
            "reusing %d already-generated (but unlabeled) examples: no re-generation",
            len(reused),
        )

    records: list[dict] = list(reused)

    if to_generate:
        model, tokenizer = load_llm(cfg)
        geom = read_geometry(model)
        device = next(model.parameters()).device
        logger.info("model geometry: %s", geom)
        logger.info("extraction batch_size: %d", cfg.extract.batch_size)

        stop_ids = resolve_stop_tokens(tokenizer, model)
        logger.info(
            "stop tokens: %s (%s)",
            stop_ids,
            [tokenizer.decode([i]) for i in stop_ids],
        )
        if not stop_ids:
            logger.warning(
                "no stop tokens found: every response will run to max_new_tokens=%d",
                cfg.dataset.max_new_tokens,
            )

        # M defaults to the layer count, which makes the field SQUARE -- that is
        # the design: we pool D down to L rather than reshaping a rectangle. Not
        # every model admits that default (Qwen2.5-7B's L=28 does not divide its
        # D_kv=512), so this raises with the valid alternatives if it cannot.
        n_segments = cfg.extract.n_segments or geom.n_layers
        geom.check_n_segments(n_segments)
        for view in VIEWS:
            d = geom.feature_dim(view)
            logger.info(
                "projection %s: D=%d -> %d segments (segment width %d)",
                view, d, n_segments, d // n_segments,
            )

        n_rows = cfg.extract.l_eff or geom.n_layers
        logger.info(
            "field shape per token: %d layers x %d segments x 3 channels (Q, K, V)",
            n_rows, n_segments,
        )

        (root / "geometry.json").write_text(
            json.dumps(
                {
                    "llm": cfg.llm.name,
                    "geometry": asdict(geom),
                    "n_segments": n_segments,
                    "n_rows": n_rows,
                    "projections": list(PROJECTIONS),
                },
                indent=2,
            )
        )

        save_dtype = DTYPES[cfg.extract.dtype]

        # Padding token for left_pad_batch: HalluShift's own convention is
        # `pad_token_id=tokenizer.eos_token_id`, and padded positions are
        # excluded from attention by the mask anyway, so which id fills them is
        # otherwise arbitrary.
        pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError(
                f"{cfg.llm.name} has neither eos_token_id nor pad_token_id; "
                "cannot left-pad a batch."
            )

        # Plain-text progress log next to geometry.json: one line every 100
        # examples ("i/total"), so a long extraction's progress can be checked
        # without tailing a log full of generation internals.
        progress_log = (root / "progress.log").open("a")
        total_to_generate = len(to_generate)
        batch_size = cfg.extract.batch_size
        batches = [
            to_generate[i : i + batch_size]
            for i in range(0, len(to_generate), batch_size)
        ]

        n_done = 0
        for batch in progress(
            batches, desc=f"extract {cfg.dataset.name}/{cfg.llm.alias}", ncols=100
        ):
            prompt_ids = [build_prompt_ids(ex.prompt, tokenizer, device) for ex in batch]
            input_ids, attention_mask = left_pad_batch(prompt_ids, pad_id, device)

            batch_out = capture_all(
                model,
                input_ids,
                max_new_tokens=cfg.dataset.max_new_tokens,
                eos_token_id=stop_ids,
                attention_mask=attention_mask,
            )

            for ex, (activations, gen_ids) in zip(batch, batch_out):
                n_done += 1

                if gen_ids.numel() == 0:
                    logger.warning("example %d generated nothing; skipping", ex.idx)
                    continue

                response = tokenizer.decode(gen_ids, skip_special_tokens=True)

                # Truncate to max_tokens BEFORE building the field: this caps
                # both compute and disk. The response text is left whole so the
                # label reflects what the model actually said.
                if (
                    cfg.extract.max_tokens
                    and activations["Q"].shape[0] > cfg.extract.max_tokens
                ):
                    activations = {
                        v: t[: cfg.extract.max_tokens] for v, t in activations.items()
                    }

                field = build_feature_field(activations, n_segments=n_segments)
                if cfg.extract.l_eff is not None:
                    field = pool_layer_axis(field, cfg.extract.l_eff)

                # Written with a placeholder label; labeling below rewrites meta.
                write_example(
                    root / f"{ex.idx:05d}", field, ex.prompt, response, ex.gold,
                    score=float("nan"), label=-1, save_dtype=save_dtype,
                )

                records.append(
                    {
                        "idx": ex.idx,
                        "dir": f"{ex.idx:05d}",
                        "n_tokens": int(field.shape[0]),
                        "prompt": ex.prompt,
                        "response": response,
                        "gold": ex.gold,
                    }
                )

                if n_done % 100 == 0 or n_done == total_to_generate:
                    progress_log.write(f"{n_done}/{total_to_generate}\n")
                    progress_log.flush()

        progress_log.close()
    elif reused:
        logger.info("nothing to generate; going straight to labeling + manifest")

    if not records:
        logger.warning("no new examples extracted")
        return

    logger.info("labeling %d examples with scheme=%s", len(records), cfg.labeling.scheme)
    scored = label_examples(
        cfg, [r["response"] for r in records], [r["gold"] for r in records]
    )

    for rec, (score, label) in zip(records, scored):
        rec["score"] = score
        rec["label"] = label
        (root / rec["dir"] / "meta.txt").write_text(
            format_meta(rec["prompt"], rec["response"], rec["gold"], score, label),
            encoding="utf-8",
        )

    write_manifest(root, records, chunk)

    n_hall = sum(r["label"] for r in records)
    logger.info(
        "done: %d examples, %d hallucinated (%.1f%%)",
        len(records), n_hall, 100 * n_hall / len(records),
    )
    if n_hall == 0 or n_hall == len(records):
        logger.warning(
            "DEGENERATE LABELS: every example has the same label. The classifier "
            "cannot learn anything. Check the prompt template and the gold field."
        )
    elif min(n_hall, len(records) - n_hall) / len(records) < 0.05:
        logger.warning(
            "labels are highly imbalanced (%.1f%% minority). AUROC will be noisy; "
            "consider a harder dataset.",
            100 * min(n_hall, len(records) - n_hall) / len(records),
        )


def write_manifest(root: Path, new_records: list[dict], chunk: int | None) -> None:
    """Merge new records into manifest.jsonl, keyed by idx (last write wins).

    Chunked runs each append their own slice, so we re-read and merge rather
    than truncating -- otherwise chunk 2 would erase chunk 1's entries.
    """
    path = root / "manifest.jsonl"
    merged: dict[int, dict] = {}

    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    merged[rec["idx"]] = rec

    for rec in new_records:
        slim = {k: rec[k] for k in ("idx", "dir", "n_tokens", "score", "label")}
        merged[rec["idx"]] = slim

    with open(path, "w") as f:
        for idx in sorted(merged):
            f.write(json.dumps(merged[idx]) + "\n")

    logger.info("manifest now has %d examples: %s", len(merged), path)


def relabel(cfg: Config) -> None:
    """Recompute labels from the stored responses WITHOUT re-extracting.

    This is why meta.txt keeps the response and gold: swapping exact_match for
    BLEURT is a cheap CPU pass, not a multi-hour GPU re-run.
    """
    root = cfg.example_dir()
    path = root / "manifest.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}; run `extract` first")

    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    responses, golds = [], []
    for rec in records:
        meta = parse_meta(root / rec["dir"] / "meta.txt")
        responses.append(meta["response"])
        golds.append(meta["gold"])

    logger.info("relabeling %d examples with scheme=%s", len(records), cfg.labeling.scheme)
    scored = label_examples(cfg, responses, golds)

    for rec, (score, label) in zip(records, scored):
        rec["score"], rec["label"] = score, label

    with open(path, "w") as f:
        for rec in sorted(records, key=lambda r: r["idx"]):
            f.write(json.dumps(rec) + "\n")

    n_hall = sum(r["label"] for r in records)
    logger.info("relabeled: %d hallucinated of %d (%.1f%%)",
                n_hall, len(records), 100 * n_hall / len(records))


def parse_meta(path: Path) -> dict:
    """Parse meta.txt. Fields are single-line 'key: value'; the gold field may
    be a stringified list, which the labelers handle."""
    text = path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        for key in ("prompt", "response", "gold", "score", "label"):
            prefix = f"{key}: "
            if line.startswith(prefix):
                out[key] = line[len(prefix):]
                current = key
                break
        else:
            # Continuation of a multi-line field (a response with newlines).
            if current:
                out[current] += "\n" + line
    return out
