# Orchestrates feature extraction: generate -> capture Q/K/V (+ hidden states) -> build fields -> save.
#
# One `extract` run generates each example once (manual decode loop per
# batch, see qkv_hooks.capture_all) and writes every requested variant from
# that same decode:
#
#     {data_root}/{dataset}/{llm_alias}/
#         judge_cache.jsonl  llm_judge labels, shared by every variant
#         progress.log       "i/total" appended every 100 generated examples
#         default/           QKV, mean pool, the model's default M
#             00000/tokens.npy   (T, L, M, C) float16, C=3 (Q, K, V) or 1 (H)
#             00000/meta.txt     prompt / response / gold / score / label
#             manifest.jsonl     one JSON line per example (the training index)
#             geometry.json      the model geometry the fields were built with
#         qkv_M8/ hs/ hs_M16/ qkv_max/ ...   named by Config.variant_name
#
# Restartable: an example already complete in every requested variant is
# skipped unless --overwrite; one missing from any variant is regenerated and
# written to all of them, so every variant of an example comes from one decode.
#
# `methods` selects which scripts/reproducing_baselines/ pipeline(s) also get
# fed from this SAME generation call (default: qkv-lens only).
#   qkv-lens   Always on. The variant fields above.
#   hallushift  Also captures hidden_states/attentions/logits per decode step
#               and writes build_hallushift_row's output to
#               {data_root}/hallushift/{dataset}/{llm_alias}/rows.jsonl.
#               Forces batch_size=1 and attn_implementation="eager" for the
#               whole run (capture_all requires it).
#   haloscope   Not implemented here -- runs as its own separate phase
#               (scripts/reproducing_baselines/haloscope/), not from this loop.

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.extract.datasets import load_examples
from src.extract.hallushift_features import build_hallushift_row
from src.extract.qkv_hooks import VIEWS, capture_all, left_pad_batch, read_geometry
from src.extract.tensor_ops import (
    PROJECTIONS,
    build_feature_field,
    build_hidden_states_field,
    pool_layer_axis,
)
from src.label.registry import label_examples
from src.utils.logger import get_logger
from src.utils.progress import progress

logger = get_logger(__name__)

DTYPES = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}

# "haloscope" is accepted by --methods but handled by its own separate script.
KNOWN_METHODS = ("qkv-lens", "hallushift", "haloscope")
SHARED_GENERATION_METHODS = ("qkv-lens", "hallushift")


def load_llm(cfg: Config, require_eager_attention: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading %s", cfg.llm.name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm.name)
    extra_kwargs = {}
    if require_eager_attention:
        # output_attentions=True (hallushift) needs eager attention -- fused
        # kernels don't materialize the full attention matrix.
        extra_kwargs["attn_implementation"] = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llm.name,
        dtype=DTYPES[cfg.llm.dtype],
        device_map="auto",
        **extra_kwargs,
    )
    model.eval()
    return model, tokenizer


def resolve_stop_tokens(tokenizer, model) -> list[int]:
    # tokenizer.eos_token_id only, matching HalluShift exactly -- no chat
    # template end-of-turn ids, no newline heuristic.
    ids: set[int] = set()

    for source in (tokenizer.eos_token_id, model.config.eos_token_id):
        if source is None:
            continue
        if isinstance(source, (list, tuple)):
            ids.update(int(i) for i in source)
        else:
            ids.add(int(source))

    return sorted(ids)


# eos_token_id alone doesn't reliably stop these models on raw-text prompts
# (~75% of responses ran on without this cut, per corpus audit). A run-on
# generation rolls into a fabricated new "Q: ..." turn, or drifts into a
# second paragraph with no literal "Q:" -- cut at whichever comes first.
RUNON_RE = re.compile(r"\bQ:\s")
NEWLINE_RE = re.compile(r"\n")


def truncate_runon(gen_ids: torch.Tensor, tokenizer) -> tuple[torch.Tensor, str]:
    # Cuts at the first fabricated "Q:" turn or newline, whichever comes
    # first. Operates on the actual generated ids (not a re-tokenized
    # decode) so the cut can't drift relative to what capture_all recorded.
    #
    # A leading newline or "Q:" (nothing real precedes it) is skipped rather
    # than treated as a cut -- Llama-2-family base models routinely emit a
    # leading newline with no run-on intent (confirmed case:
    # truthfulqa/llama2_7b idx 432) -- the search resumes past it.
    #
    # Returns (kept_ids, kept_text), unchanged if no marker survives, or
    # (empty, "") if there's truly no content before the marker.
    if gen_ids.numel() == 0:
        return gen_ids, ""

    full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    lead_ws = re.match(r"\s+", full_text)
    search_from = lead_ws.end() if lead_ws else 0

    q_match = RUNON_RE.search(full_text, search_from)
    nl_match = NEWLINE_RE.search(full_text, search_from)
    candidates = [m.start() for m in (q_match, nl_match) if m is not None]
    if not candidates:
        return gen_ids, full_text

    cutoff_char = min(candidates)
    n_total = gen_ids.shape[0]
    for n in range(1, n_total + 1):
        prefix_text = tokenizer.decode(gen_ids[:n], skip_special_tokens=True)
        if len(prefix_text) >= cutoff_char:
            kept_ids = gen_ids[: max(n - 1, 0)]
            kept_text = tokenizer.decode(kept_ids, skip_special_tokens=True).rstrip()
            if not kept_text:
                # marker appears essentially immediately, nothing real before it
                return gen_ids[:0], ""
            # .rstrip() can change the token count via BPE re-merge; find how
            # many of kept_ids' own tokens survive, rather than trusting len().
            m = len(kept_ids)
            while m > 0 and tokenizer.decode(kept_ids[:m], skip_special_tokens=True).rstrip() != kept_text:
                m -= 1
            return kept_ids[:m], kept_text
    return gen_ids, full_text


def build_prompt_ids(prompt: str, tokenizer, device) -> torch.Tensor:
    # Raw text for every model, instruct or not -- HalluShift never applies a
    # chat template, and reproducing its numbers means reproducing that.
    out = tokenizer(prompt, return_tensors="pt")

    if not isinstance(out, torch.Tensor):
        out = out["input_ids"]

    if out.ndim == 1:
        out = out.unsqueeze(0)

    return out.to(device)


def hallushift_dir(cfg: Config) -> Path:
    return Path(cfg.data_root) / "hallushift" / cfg.dataset.name / cfg.llm.alias


def load_hallushift_rows(path: Path) -> dict[int, list]:
    # {idx: row} from a previous run's rows.jsonl. Tolerates a truncated last
    # line (crash mid-write) by skipping it -- that example regenerates.
    rows: dict[int, list] = {}
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[rec["idx"]] = rec["row"]
    return rows


def append_hallushift_row(path: Path, idx: int, row: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"idx": idx, "row": row}) + "\n")


def format_meta(prompt: str, response: str, gold, score: float, label: int) -> str:
    return (
        f"prompt: {prompt}\n"
        f"response: {response}\n"
        f"gold: {gold}\n"
        f"score: {score}\n"
        f"label: {label}\n"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    # Write-then-rename, so a job killed mid-write never leaves a truncated file behind.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


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

    tmp = out_dir / "tokens.npy.tmp"
    with open(tmp, "wb") as f:
        np.save(f, field.to(save_dtype).numpy())
    os.replace(tmp, out_dir / "tokens.npy")

    _atomic_write_text(out_dir / "meta.txt", format_meta(prompt, response, gold, score, label))


def is_complete(out_dir: Path) -> bool:
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
    # True once generation (the expensive step) is done, regardless of
    # whether a label has been resolved yet.
    if not (out_dir / "tokens.npy").exists():
        return False
    meta_path = out_dir / "meta.txt"
    if not meta_path.exists():
        return False
    try:
        meta = parse_meta(meta_path)
    except OSError:
        return False
    return "response" in meta and "gold" in meta


def _record_from_meta(out_dir: Path, idx: int) -> dict:
    # Rebuilds the in-memory record for an already-generated example from disk.
    meta = parse_meta(out_dir / "meta.txt")
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
    # Indices manifest.jsonl already records as fully labeled, for `root`.
    # Reads ONLY manifest.jsonl, so this is cheap enough to check before
    # paying for load_examples/load_llm.
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


def run_extraction(
    cfg: Config,
    chunk: int | None = None,
    overwrite: bool = False,
    methods: tuple[str, ...] = ("qkv-lens",),
    variants: tuple[str, ...] = (),
) -> None:
    unknown = set(methods) - set(KNOWN_METHODS)
    if unknown:
        raise ValueError(f"unknown method(s) {sorted(unknown)}; known: {KNOWN_METHODS}")
    want_hallushift = "hallushift" in methods

    variant_cfgs = {cfg.variant_name(): cfg}
    for name in variants:
        variant_cfgs.setdefault(name, cfg.for_variant(name))
    pair = cfg.pair_dir(cfg.dataset.name, cfg.llm.alias)
    roots = {name: c.example_dir() for name, c in variant_cfgs.items()}
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    logger.info("variants: %s", ", ".join(roots))
    hs_dir = hallushift_dir(cfg)
    hs_rows_path = hs_dir / "rows.jsonl"

    # Fastest possible pre-check: every requested variant's manifest.jsonl
    # existing (no content read) is treated as proof a prior non-chunked run
    # completed -- it's the last thing a run writes. A short manifest from an
    # interrupted run needs --overwrite to redo.
    #
    # Skipped when hallushift is requested: a manifest from a prior
    # qkv-lens-only run proves nothing about hallushift rows existing.
    # Chunked runs share one manifest across chunks, so this can't tell
    # whether THIS chunk's range is done -- they use the index-set check below.
    if not overwrite and chunk is None and not want_hallushift:
        if all((r / "manifest.jsonl").exists() for r in roots.values()):
            logger.info(
                "%s/%s: every requested variant already has a manifest.jsonl -- "
                "skipping load_examples/load_llm entirely (use --overwrite to redo)",
                cfg.dataset.name, cfg.llm.alias,
            )
            return
    elif not overwrite and chunk is not None and not want_hallushift:
        lo, hi = (chunk - 1) * 1000, chunk * 1000
        wanted = set(range(lo, hi))
        if wanted and all(wanted <= _manifest_complete_indices(r) for r in roots.values()):
            logger.info(
                "%s/%s chunk %d: all %d requested examples already complete -- "
                "skipping load_examples/load_llm entirely (use --overwrite to redo)",
                cfg.dataset.name, cfg.llm.alias, chunk, len(wanted),
            )
            return

    examples = load_examples(cfg)
    logger.info("loaded %d examples for %s", len(examples), cfg.dataset.name)

    # Chunking mirrors ACT-ViT: 1-indexed blocks of 1000.
    if chunk is not None:
        lo, hi = (chunk - 1) * 1000, chunk * 1000
        examples = [e for e in examples if lo <= e.idx < hi]
        logger.info("chunk %d -> %d examples (idx %d..%d)", chunk, len(examples), lo, hi - 1)
        if not examples:
            logger.warning("chunk %d is empty; nothing to do", chunk)
            return

    # Partition before touching the GPU: generation is the only step that
    # needs the LLM.
    #   complete    -> labeled in every requested variant (and hallushift row
    #                  present if asked); skip.
    #   generated   -> tensors+response on disk in every variant but unlabeled;
    #                  reuse (no LLM), let labeling finish it. Still requires
    #                  hallushift's row if requested -- it can't be recovered
    #                  retroactively, so a missing row forces regeneration.
    #   to_generate -> missing from at least one variant; needs the model.
    hs_rows = load_hallushift_rows(hs_rows_path) if want_hallushift else {}

    # A kill between per-variant manifest writes leaves complete examples with no manifest; re-list them.
    manifests_ok = all((r / "manifest.jsonl").exists() for r in roots.values())

    reused: list[dict] = []
    to_generate = []
    skipped = 0
    for ex in examples:
        ex_dirs = [r / f"{ex.idx:05d}" for r in roots.values()]
        hs_ok = (not want_hallushift) or (ex.idx in hs_rows)
        if not overwrite and hs_ok and manifests_ok and all(is_complete(d) for d in ex_dirs):
            skipped += 1
        elif not overwrite and hs_ok and all(is_generated(d) for d in ex_dirs):
            reused.append(_record_from_meta(ex_dirs[0], ex.idx))
        else:
            to_generate.append(ex)

    if skipped:
        logger.info("skipped %d already-complete examples (use --overwrite to redo)", skipped)
    if reused:
        logger.info(
            "reusing %d already-generated (but unlabeled) examples: no re-generation",
            len(reused),
        )
    if want_hallushift and to_generate:
        n_missing_hs = sum(1 for ex in to_generate if ex.idx not in hs_rows)
        if n_missing_hs:
            logger.info(
                "%d example(s) need (re)generation because their hallushift "
                "row is missing (a prior qkv-lens-only extraction can't "
                "supply it retroactively)",
                n_missing_hs,
            )

    records: list[dict] = list(reused)

    if to_generate:
        model, tokenizer = load_llm(cfg, require_eager_attention=want_hallushift)
        geom = read_geometry(model)
        device = next(model.parameters()).device
        logger.info("model geometry: %s", geom)

        # Forced here (not left to config) so a run started with
        # --methods qkv-lens,hallushift can't silently keep an unrelated
        # batch_size and hit capture_all's ValueError deep into generation.
        batch_size = 1 if want_hallushift else cfg.extract.batch_size
        logger.info("extraction batch_size: %d%s", batch_size,
                    " (forced to 1 for hallushift)" if want_hallushift else "")

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

        # M defaults to the layer count (square field); not every model
        # admits that (Qwen2.5-7B's L=28 doesn't divide D_kv=512).
        n_rows = cfg.extract.l_eff or geom.n_layers
        specs: dict[str, tuple[str, str, int]] = {}
        for name, c in variant_cfgs.items():
            source, pool = c.extract.source, c.extract.pool
            n_segments = c.extract.n_segments or geom.n_layers
            if source == "qkv":
                try:
                    geom.check_n_segments(n_segments)
                except ValueError as e:
                    raise ValueError(f"variant {name}: {e}") from e
            elif geom.hidden_size % n_segments != 0:
                raise ValueError(
                    f"variant {name}: n_segments={n_segments} does not evenly "
                    f"divide hidden_size={geom.hidden_size}"
                )
            specs[name] = (source, pool, n_segments)
            projections = list(PROJECTIONS) if source == "qkv" else ["H"]
            logger.info(
                "variant %-10s %s, %s pool, field %d layers x %d segments x %d channel(s)",
                name, source, pool, n_rows, n_segments, len(projections),
            )
            _atomic_write_text(
                roots[name] / "geometry.json",
                json.dumps(
                    {
                        "llm": cfg.llm.name,
                        "geometry": asdict(geom),
                        "n_segments": n_segments,
                        "n_rows": n_rows,
                        "projections": projections,
                        "pool": pool,
                        "source": source,
                    },
                    indent=2,
                )
            )
        want_hidden = any(source == "hidden-states" for source, _, _ in specs.values())

        if want_hallushift:
            hs_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_text(
                hs_dir / "geometry.json",
                json.dumps({"llm": cfg.llm.name, "num_layers": geom.n_layers}, indent=2),
            )

        save_dtype = DTYPES[cfg.extract.dtype]

        pad_id = tokenizer.eos_token_id
        if pad_id is None:
            pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError(
                f"{cfg.llm.name} has neither eos_token_id nor pad_token_id; "
                "cannot left-pad a batch."
            )

        progress_log = (pair / "progress.log").open("a")
        total_to_generate = len(to_generate)
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
                capture_generate_outputs=want_hallushift,
                capture_hidden_states=want_hidden,
            )

            for ex, row_out in zip(batch, batch_out):
                n_done += 1

                if want_hallushift:
                    activations, gen_ids, generate_outputs = row_out
                else:
                    activations, gen_ids = row_out
                    generate_outputs = None

                if gen_ids.numel() == 0:
                    logger.warning("example %d generated nothing; skipping", ex.idx)
                    continue

                # Cut before anything else, by token count, so field[t],
                # response's t-th token, and hallushift's per-token features
                # stay in exact correspondence.
                kept_ids, response = truncate_runon(gen_ids, tokenizer)
                n_keep = kept_ids.shape[0]
                if n_keep < gen_ids.shape[0]:
                    activations = {v: t[:n_keep] for v, t in activations.items()}
                    if generate_outputs is not None:
                        generate_outputs = {
                            k: v[:n_keep] for k, v in generate_outputs.items()
                        }

                # Truncate to max_tokens BEFORE building the field (caps
                # compute+disk); response text stays whole for labeling.
                if (
                    cfg.extract.max_tokens
                    and activations["Q"].shape[0] > cfg.extract.max_tokens
                ):
                    activations = {
                        v: t[: cfg.extract.max_tokens] for v, t in activations.items()
                    }
                    if generate_outputs is not None:
                        generate_outputs = {
                            k: v[: cfg.extract.max_tokens] for k, v in generate_outputs.items()
                        }

                for name, (source, pool, n_segments) in specs.items():
                    if source == "qkv":
                        field = build_feature_field(activations, n_segments=n_segments, pool=pool)
                    else:
                        field = build_hidden_states_field(
                            activations["H"], n_segments=n_segments, pool=pool
                        )
                    if cfg.extract.l_eff is not None:
                        field = pool_layer_axis(field, cfg.extract.l_eff)
                    write_example(
                        roots[name] / f"{ex.idx:05d}", field, ex.prompt, response, ex.gold,
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

                if want_hallushift:
                    hs_row = build_hallushift_row(generate_outputs, geom.n_layers, response)
                    append_hallushift_row(hs_rows_path, ex.idx, hs_row)

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
        cfg, [r["response"] for r in records], [r["gold"] for r in records], cache_dir=pair
    )

    for rec, (score, label) in zip(records, scored):
        rec["score"] = score
        rec["label"] = label
        meta = format_meta(rec["prompt"], rec["response"], rec["gold"], score, label)
        for root in roots.values():
            _atomic_write_text(root / rec["dir"] / "meta.txt", meta)

    for root in roots.values():
        write_manifest(root, records, chunk)

    n_hall = sum(r["label"] for r in records)
    logger.info(
        "done: %d examples x %d variant(s), %d hallucinated (%.1f%%)",
        len(records), len(roots), n_hall, 100 * n_hall / len(records),
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
    # Merges new records into manifest.jsonl, keyed by idx (last write wins).
    # Re-reads and merges rather than truncating, so chunk 2 can't erase chunk 1.
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

    _atomic_write_text(path, "".join(json.dumps(merged[idx]) + "\n" for idx in sorted(merged)))

    logger.info("manifest now has %d examples: %s", len(merged), path)


def relabel(cfg: Config) -> None:
    # Relabels every variant of this (dataset, llm); each unique (response, gold) is scored once.
    pair = cfg.pair_dir(cfg.dataset.name, cfg.llm.alias)
    roots = sorted(p.parent for p in pair.glob("*/manifest.jsonl"))
    if not roots:
        raise FileNotFoundError(f"no */manifest.jsonl under {pair}; run `extract` first")

    unique: dict[tuple[str, str], int] = {}
    corpora = []
    for root in roots:
        with open(root / "manifest.jsonl") as f:
            records = [json.loads(line) for line in f if line.strip()]
        keys = []
        for rec in records:
            meta = parse_meta(root / rec["dir"] / "meta.txt")
            key = (meta["response"], meta["gold"])
            unique.setdefault(key, len(unique))
            keys.append(key)
        corpora.append((root, records, keys))

    pairs = list(unique)
    logger.info(
        "relabeling %d unique response(s) across %d variant(s) with scheme=%s",
        len(pairs), len(roots), cfg.labeling.scheme,
    )
    scored = label_examples(cfg, [r for r, _ in pairs], [g for _, g in pairs], cache_dir=pair)

    for root, records, keys in corpora:
        for rec, key in zip(records, keys):
            rec["score"], rec["label"] = scored[unique[key]]
        _atomic_write_text(
            root / "manifest.jsonl",
            "".join(json.dumps(rec) + "\n" for rec in sorted(records, key=lambda r: r["idx"])),
        )
        n_hall = sum(r["label"] for r in records)
        logger.info("relabeled %s: %d hallucinated of %d (%.1f%%)",
                    root.name, n_hall, len(records), 100 * n_hall / max(len(records), 1))


def parse_meta(path: Path) -> dict:
    # Fields are single-line "key: value"; gold may be a stringified list.
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
            if current:
                out[current] += "\n" + line
    return out
