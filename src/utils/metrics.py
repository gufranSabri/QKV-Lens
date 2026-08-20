"""Evaluation metrics. AUROC is primary (both baselines report it)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.05) -> float:
    """TPR at the largest FPR that does not exceed `target_fpr`. ACT-ViT's metric.

    Useful because a hallucination detector is deployed at a low false-alarm
    budget: "how many hallucinations do we catch if we may only bother the user
    5% of the time on correct answers".
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ok = np.where(fpr <= target_fpr)[0]
    return float(tpr[ok[-1]]) if len(ok) else 0.0


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # A single-class split makes AUROC undefined; report NaN rather than crash,
    # since this legitimately happens on tiny per-origin test slices.
    if len(np.unique(y_true)) < 2:
        return {
            "auroc": float("nan"),
            "pr_auc": float("nan"),
            "tpr@5fpr": float("nan"),
            "accuracy": float(accuracy_score(y_true, y_score >= threshold)),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "n": int(len(y_true)),
            "pos_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        }

    y_pred = (y_score >= threshold).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_score)

    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(auc(recall, precision)),
        "tpr@5fpr": tpr_at_fpr(y_true, y_score, 0.05),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "pos_rate": float(y_true.mean()),
    }


def format_metrics(m: dict) -> str:
    def g(k):
        v = m.get(k, float("nan"))
        return "  nan" if v != v else f"{v:.4f}"

    return (
        f"AUROC {g('auroc')} | PR-AUC {g('pr_auc')} | TPR@5%FPR {g('tpr@5fpr')} "
        f"| F1 {g('f1')} | acc {g('accuracy')} | n={m.get('n', 0)}"
    )
