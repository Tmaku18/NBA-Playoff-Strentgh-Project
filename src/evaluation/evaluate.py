"""Evaluate ranking, upset ROC-AUC, walk-forward runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from .baselines import rank_by_srs, rank_by_net_rating
from .metrics import ndcg_score, spearman, mrr, roc_auc_upset


def evaluate_ranking(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    k: int | None = 10,
) -> dict[str, float]:
    return {
        "ndcg": ndcg_score(y_true, y_score, k=k),
        "spearman": spearman(y_true, y_score),
        "mrr": mrr(y_true, y_score, top_k=1),
    }


def evaluate_upset(y_binary: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {"roc_auc_upset": roc_auc_upset(y_binary, y_score)}


def run_walk_forward(
    seasons: list[str],
    *,
    train_val_test: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Stub: train/val/test by season blocks. Returns metrics per season + aggregate."""
    return {"seasons": seasons, "metrics": {}, "aggregate": {}}
