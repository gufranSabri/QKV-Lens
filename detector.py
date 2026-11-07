#!/usr/bin/env python3
"""Detector CLI: build QKV feature fields and train the hallucination detector.

The detector-stage CLI only (extract/train/test/cam/...). Steering entry
points are separate and will not live in this file.

Subcommands:
    extract   phase 1: generate responses, capture Q/K/V (+ hidden states), save field variants (unlabeled)
    label     phase 2: label every variant's stored responses (no LLM, no re-extraction)
    train     train the detector on one dataset
    test      evaluate a saved checkpoint
    inspect   render feature fields to PNG so you can actually look at them
    cam       Integrated Gradients: where the detector looks, per token
"""

from __future__ import annotations

import argparse
import sys

from src.config import ALL_VARIANTS, load_config
from src.utils.logger import setup_logging


def _hf_login() -> None:
    import os

    token = os.environ.get("HF_TOKEN")
    if not token:
        return

    from huggingface_hub import login

    login(token=token)


def _overrides(args) -> dict:
    # Turns --set a.b=c flags into a nested override dict.
    out: dict = {}
    for item in args.set or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)

        import yaml   # parses the value so ints/floats/bools/lists come through typed

        parsed = yaml.safe_load(value)

        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed
    return out


def main(argv=None) -> int:
    # --config/--set declared on a shared parent parser so both orderings work:
    #     detector.py --config c.yaml train --set train.epochs=30
    #     detector.py train --config c.yaml --set train.epochs=30
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", help="path to a YAML config")
    common.add_argument(
        "--set", action="append", metavar="KEY=VAL", default=[],
        help="override any config key, e.g. --set train.epochs=30 "
             "--set extract.n_segments=64",
    )
    parser = argparse.ArgumentParser(
        prog="detector.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ex = sub.add_parser("extract", parents=[common],
                          help="phase 1: generate + capture Q/K/V + save feature fields (no labels)")
    p_ex.add_argument("--chunk", type=int, default=None,
                      help="1-indexed block of 1000 examples (for splitting a long run)")
    p_ex.add_argument("--overwrite", action="store_true",
                      help="re-extract examples that already have tokens.npy")
    p_ex.add_argument(
        "--methods", default="qkv-lens",
        help="comma-separated: which scripts/reproducing_baselines/ pipeline(s) to also "
             "feed from this SAME generation (default: qkv-lens only). "
             "'hallushift' adds hidden_states/attentions capture and forces "
             "batch_size=1 + eager attention for the whole run. 'haloscope' is "
             "accepted but NOT extracted here -- see scripts/reproducing_baselines/"
             "haloscope's own script for its separate beam-search generation.",
    )
    p_ex.add_argument(
        "--variants", default="",
        help="comma-separated extra variants saved from the SAME generation, on top "
             "of the config's own (default/), e.g. qkv_M8,hs,hs_M16,qkv_max. "
             "'all' = " + ",".join(ALL_VARIANTS) + ".",
    )

    p_lb = sub.add_parser("label", parents=[common],
                          help="phase 2: label every variant of this (dataset, llm) from stored responses")
    p_lb.add_argument("--mode", choices=("api", "export"), default="api",
                      help="api: judge via labeling.scheme (default). export: first run writes a CSV "
                           "per (dataset, llm) with a blank label column; once you fill it in, the "
                           "next run imports those labels into every variant")
    p_lb.add_argument("--sheets", default=None,
                      help="export mode: folder for the CSVs (default scripts/exports/label_sheets)")
    p_lb.add_argument("--overwrite", action="store_true",
                      help="api mode: relabel rows that already have a label (default: only unlabeled ones)")

    p_tr = sub.add_parser("train", parents=[common], help="train the detector")
    p_tr.add_argument("--dataset", default=None, help="defaults to the config's dataset")
    p_tr.add_argument("--run-name", type=str, default=None)

    p_te = sub.add_parser("test", parents=[common], help="evaluate a checkpoint")
    p_te.add_argument("--checkpoint", required=True)
    p_te.add_argument("--dataset", default=None, help="defaults to the config's dataset")
    p_te.add_argument(
        "--recompute-stats", action="store_true",
        help="normalize with stats recomputed from the eval corpus instead of "
             "the checkpoint's training stats (for a genuine cross-LLM eval)",
    )
    p_te.add_argument(
        "--out-name", default=None,
        help="write test_<name>.json under this name instead of the dataset "
             "name (avoids clobbering an in-distribution result, e.g. for a "
             "cross-LLM eval reusing the same checkpoint)",
    )

    p_in = sub.add_parser("inspect", parents=[common], help="render feature fields to PNG")
    p_in.add_argument("--idx", type=int, default=0, help="example index")
    p_in.add_argument("--tokens", type=int, default=4, help="how many tokens to render")
    p_in.add_argument("--out", default=None)

    p_cam = sub.add_parser(
        "cam", parents=[common],
        help="Integrated Gradients: where the detector looks, per token",
    )
    p_cam.add_argument("--checkpoint", required=True)
    p_cam.add_argument("--dataset", default=None, help="defaults to the config's dataset")
    p_cam.add_argument("--idx", type=int, default=0, help="example index")
    p_cam.add_argument("--method", choices=["ig"], default="ig")
    p_cam.add_argument(
        "--max-tokens", type=int, default=20,
        help="cap on generated-token columns shown (0 or negative = show all)",
    )
    p_cam.add_argument("--out", default=None)

    # Two-stage parse: the subparser's default would otherwise silently
    # overwrite a flag captured before the subcommand (argparse runs the
    # subparser last), e.g. `--config c.yaml train` ending with config=None.
    pre, _ = common.parse_known_args(argv)
    args = parser.parse_args(argv)

    if not args.config:
        args.config = pre.config
    args.set = list(pre.set or []) + [s for s in (args.set or []) if s not in (pre.set or [])]

    if not args.config:
        parser.error("--config is required (before or after the subcommand)")
    setup_logging()
    _hf_login()

    cfg = load_config(args.config, overrides=_overrides(args))

    if args.cmd == "extract":
        from src.extract.run_extraction import KNOWN_METHODS, run_extraction

        methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
        unknown = set(methods) - set(KNOWN_METHODS)
        if unknown:
            parser.error(f"--methods: unknown method(s) {sorted(unknown)}; known: {KNOWN_METHODS}")
        variants: list[str] = []
        for v in (v.strip() for v in args.variants.split(",")):
            if v:
                variants += ALL_VARIANTS if v == "all" else [v]
        for v in variants:
            try:
                cfg.for_variant(v)
            except ValueError as e:
                parser.error(f"--variants: {e}")
        run_extraction(
            cfg, chunk=args.chunk, overwrite=args.overwrite, methods=methods,
            variants=tuple(dict.fromkeys(variants)),
        )

    elif args.cmd == "label":
        from src.extract.run_extraction import relabel

        if args.mode == "export":
            from src.label.sheets import label_from_sheet, sheet_path

            remaining = label_from_sheet(cfg, args.sheets)
            if remaining:
                print(f"{remaining} example(s) still unlabeled -- fill in {sheet_path(cfg, args.sheets)} and rerun")
                return 3
        else:
            relabel(cfg, overwrite=args.overwrite)

    elif args.cmd == "train":
        from src.train import train

        train(cfg, args.dataset or cfg.dataset.name, run_name=args.run_name)

    elif args.cmd == "test":
        from src.test import test

        test(
            cfg, args.checkpoint, dataset_name=args.dataset,
            recompute_stats=args.recompute_stats, out_name=args.out_name,
        )

    elif args.cmd == "inspect":
        from src.inspect_images import inspect

        inspect(cfg, idx=args.idx, n_tokens=args.tokens, out=args.out)

    elif args.cmd == "cam":
        from src.cam import run_cam

        max_tokens_shown = None if args.max_tokens <= 0 else args.max_tokens
        run_cam(
            cfg, args.checkpoint, dataset_name=args.dataset,
            idx=args.idx, method=args.method, out=args.out,
            max_tokens_shown=max_tokens_shown,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
