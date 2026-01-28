"""Ranking metrics: NDCG, Spearman, MRR. Optional: Brier, ROC-AUC upset."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def ndcg_score(y_true: np.ndarray, y_score: np.ndarray, k: int | None = None) -> float:
    """y_true: relevance (higher=better). y_score: predicted scores. NDCG@k."""
    order = np.argsort(y_score)[::-1]
    y_true = np.asarray(y_true)[order]
    dcg = 0.0
    for i, r in enumerate(y_true):
        if k is not None and i >= k:
            break
        dcg += (2 ** float(r) - 1) / np.log2(i + 2)
    ideal = np.sort(np.asarray(y_true))[::-1]
    idcg = 0.0
    for i in range(min(len(ideal), k or len(ideal))):
        idcg += (2 ** float(ideal[i]) - 1) / np.log2(i + 2)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def spearman(y_true: np.ndarray, y_score: np.ndarray) -> float:
    r, _ = spearmanr(y_true, y_score, nan_policy="omit")
    return float(r) if np.isfinite(r) else 0.0


def mrr(y_true: np.ndarray, y_score: np.ndarray, top_k: int = 1) -> float:
    """MRR: 1/rank of first relevant in ranking by y_score. For binary rel, top item."""
    order = np.argsort(y_score)[::-1]
    y = np.asarray(y_true)
    for i in range(min(top_k, len(order))):
        if y[order[i]] >= np.max(y) - 1e-9:
            return 1.0 / (i + 1)
    return 0.0


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier: mean (p - y)^2 for binary y."""
    return float(np.mean((np.asarray(y_prob).ravel() - np.asarray(y_true).ravel()) ** 2))


def roc_auc_upset(y_binary: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC for binary upset (1=upset) vs continuous score (higher=more likely upset)."""
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(np.asarray(y_binary).ravel(), np.asarray(y_score).ravel()))
    except Exception:
        return 0.5
