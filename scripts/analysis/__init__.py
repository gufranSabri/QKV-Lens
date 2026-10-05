"""Analyses and data exports that are not a headline table or figure.

    forecasting*.py, run_forecasting.py, run_all.py
        How early in a response can the detector call a hallucination? Re-runs
        a trained detector over every PREFIX of every held-out response and
        reports both when the verdict becomes usable and when it stops
        changing. `run_all.py` is the entry point; it writes into
        docs/{figures,tables,reports}/forecasting/.

    significance.py     class-separability statistics for the QKV fields
    corpus_stats.py     per-corpus descriptive statistics
    export_responses_csv.py   prompt/gold/response dumps for manual inspection
"""
