"""Migrate delta-QKV-data (old QKV-Lens (T,V,L,M,3) layout) to the QKV-Steer
canonical (T,L,M,3) layout, truncating each response at the first run-on
"Q:" turn marker.

Losslessness of the migration step (verified interactively before writing
this script):
  - old tokens.npy is (T, V, L, M, 3); V iterates VIEWS = ("Q","K","V"),
    the SAME order as current PROJECTIONS.
  - channel 0 of the trailing axis-3 is "pooled" (mean-pooled raw activation)
    in BOTH extraction_type=delta and extraction_type=transforms -- verified
    byte-identical on a sample example.
  - pool mode was "mean" (not the old default.yaml's fallback "max") in every
    geometry.json under qkv/delta -- confirmed by grep across all 12 trees.
  - VIEWS order matches between the old and current repos (both "Q","K","V").
  So: new[t, l, m, v] = old[t, v, l, m, 0]  (slice channel 0, move axis 1 -> -1).

Truncation:
  - meta.txt only stores the DECODED response text, not raw generated token
    ids (true in both the old and current repos), so the cut point is found
    by retokenizing the response with that example's own tokenizer and
    locating the first token span whose decoded text contains a new
    r"\bQ:\s" turn marker -- the same heuristic used for the corpus-wide
    overrun scan. Both tokens.npy rows and the response text are cut there.
  - Retokenizing is done incrementally (decode a growing prefix of ids) so
    the cut lands on a token boundary that is faithful to what the model
    actually produced, not a character split of the decoded string.

Output layout (matches src/config.py's example_dir + write_example/write_manifest):
    {DATA_ROOT}/{dataset}/{llm_alias}/
        NNNNN/tokens.npy   (T', L, M, 3) float16, same dtype as source
        NNNNN/meta.txt     prompt/response(truncated)/gold/score(placeholder)/label(placeholder)
        manifest.jsonl
        geometry.json
    Labels/scores are placeholders (nan/-1); `detector.py label` (relabel())
    recomputes them from the truncated response text afterward.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

SRC_ROOT = Path("/scratch/ahmedubc/delta-QKV-data/qkv/delta")
DST_ROOT = Path("/scratch/ahmedubc/QKV-Steer-data")

TURN_RE = re.compile(r"\bQ:\s")

_tokenizer_cache: dict[str, object] = {}


def get_tokenizer(model_name: str):
    if model_name not in _tokenizer_cache:
        from transformers import AutoTokenizer

        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizer_cache[model_name]


def parse_meta(path: Path) -> dict:
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


def format_meta(prompt: str, response: str, gold, score, label) -> str:
    return (
        f"prompt: {prompt}\n"
        f"response: {response}\n"
        f"gold: {gold}\n"
        f"score: {score}\n"
        f"label: {label}\n"
    )


def truncation_cutoff(response: str, tokenizer) -> tuple[int, str]:
    """Return (n_tokens_to_keep, truncated_text).

    Retokenizes `response` (the text that was itself produced by
    tokenizer.decode(gen_ids, skip_special_tokens=True) at extraction time),
    then grows a prefix one token at a time to find the first token whose
    inclusion makes the decoded prefix contain a "Q:" turn marker beyond the
    initial answer. If no such marker exists, keeps everything.
    """
    ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if not ids:
        return 0, response

    full_match = TURN_RE.search(response)
    if not full_match:
        return len(ids), response

    # Grow the decoded prefix token-by-token until it first reaches/exceeds
    # the character offset where the run-on marker starts in the full text.
    cutoff_char = full_match.start()
    for n in range(1, len(ids) + 1):
        prefix_text = tokenizer.decode(ids[:n], skip_special_tokens=True)
        if len(prefix_text) >= cutoff_char:
            # Keep tokens [0, n-1): drop the token(s) that pushed us past the
            # marker start, then strip any trailing whitespace left dangling.
            kept_ids = ids[: max(n - 1, 0)]
            kept_text = tokenizer.decode(kept_ids, skip_special_tokens=True).rstrip()
            if not kept_text:
                # Marker appears essentially immediately; keep at least
                # nothing rather than guess -- caller can inspect idx==0 cases.
                return 0, ""
            # `kept_text` went through .rstrip() and can also retokenize to a
            # different id count than `kept_ids` (BPE re-merges at the new
            # boundary) -- re-tokenize the ACTUAL saved text so n_keep always
            # matches what tokens.npy is sliced to. Without this, tokens.npy
            # and meta.txt's response silently disagree on length for ~10-15%
            # of truncated examples (confirmed empirically).
            final_ids = tokenizer(kept_text, add_special_tokens=False)["input_ids"]
            return len(final_ids), kept_text
    return len(ids), response


def migrate_pair(dataset: str, llm_alias: str, dry_run: bool = False) -> dict:
    src_dir = SRC_ROOT / dataset / llm_alias
    dst_dir = DST_ROOT / dataset / llm_alias
    dst_dir.mkdir(parents=True, exist_ok=True)

    geom = json.loads((src_dir / "geometry.json").read_text())
    model_name = geom["llm"]
    tokenizer = get_tokenizer(model_name)

    manifest_records = []
    stats = {"total": 0, "truncated": 0, "empty_after_truncate": 0}

    example_dirs = sorted(
        (p for p in src_dir.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )

    for ex_dir in tqdm(example_dirs, desc=f"{dataset}/{llm_alias}", ncols=100):
        meta = parse_meta(ex_dir / "meta.txt")
        response = meta.get("response", "")
        prompt = meta.get("prompt", "")
        gold = meta.get("gold", "")

        n_keep, truncated_response = truncation_cutoff(response, tokenizer)
        was_truncated = truncated_response != response

        old_tokens = np.load(ex_dir / "tokens.npy")  # (T, V, L, M, 3)
        raw_channel = old_tokens[..., 0]  # (T, V, L, M)
        new_field = np.transpose(raw_channel, (0, 2, 3, 1))  # (T, L, M, V=Q,K,V)

        # Only slice when truncation_cutoff actually found a cut: `n_keep` is
        # a token count from RETOKENIZING the decoded response, which is not
        # always identical to len(gen_ids) that produced this tensor (BPE
        # re-merges can shift the last token or two at a sequence boundary --
        # confirmed on ~1% of examples for opt_6.7b/qwen2.5_7b). Trusting
        # n_keep as a row count for an UNTRUNCATED example would silently
        # drop or misalign tokens.npy's final row for exactly those cases.
        # Clamp defensively even in the truncated case.
        if was_truncated:
            new_field = new_field[: min(n_keep, new_field.shape[0])]

        stats["total"] += 1
        if was_truncated:
            stats["truncated"] += 1
        if n_keep == 0:
            stats["empty_after_truncate"] += 1

        if dry_run:
            continue

        out_dir = dst_dir / ex_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "tokens.npy", new_field.astype(old_tokens.dtype))
        (out_dir / "meta.txt").write_text(
            format_meta(prompt, truncated_response, gold, score=float("nan"), label=-1),
            encoding="utf-8",
        )
        manifest_records.append(
            {
                "idx": int(ex_dir.name),
                "dir": ex_dir.name,
                "n_tokens": int(new_field.shape[0]),
                "score": float("nan"),
                "label": -1,
            }
        )

    if not dry_run:
        with open(dst_dir / "manifest.jsonl", "w") as f:
            for rec in sorted(manifest_records, key=lambda r: r["idx"]):
                f.write(json.dumps(rec) + "\n")

        new_geometry = {
            "llm": geom["llm"],
            "geometry": geom["geometry"],
            "n_segments": geom["n_cols"],
            "n_rows": geom["n_rows"],
            "projections": ["Q", "K", "V"],
        }
        (dst_dir / "geometry.json").write_text(json.dumps(new_geometry, indent=2))

    return stats


if __name__ == "__main__":
    import sys

    dry_run = "--dry-run" in sys.argv

    pairs = []
    for dataset_dir in sorted(SRC_ROOT.iterdir()):
        if not dataset_dir.is_dir():
            continue
        for model_dir in sorted(dataset_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            pairs.append((dataset_dir.name, model_dir.name))

    grand = {"total": 0, "truncated": 0, "empty_after_truncate": 0}
    for dataset, llm_alias in tqdm(pairs, desc="pairs", ncols=100):
        stats = migrate_pair(dataset, llm_alias, dry_run=dry_run)
        for k in grand:
            grand[k] += stats[k]
        tqdm.write(
            f"{dataset}/{llm_alias:15s} total={stats['total']:6d} "
            f"truncated={stats['truncated']:6d} "
            f"empty_after_truncate={stats['empty_after_truncate']:4d}"
        )

    print(
        f"\nTOTAL total={grand['total']} truncated={grand['truncated']} "
        f"empty_after_truncate={grand['empty_after_truncate']}"
    )
