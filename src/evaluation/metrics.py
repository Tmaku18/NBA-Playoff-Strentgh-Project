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


def mrr(y_true: np.ndarray, y_score: np.ndarray, top_k: int = 2) -> float:
    """MRR: 1/rank of first relevant in ranking by y_score. top_k=2 allows two 'rank 1' (e.g. two conferences)."""
    order = np.argsort(y_score)[::-1]
    y = np.asarray(y_true)
    max_rel = np.max(y)
    for i in range(min(top_k, len(order))):
        if y[order[i]] >= max_rel - 1e-9:
            return 1.0 / (i + 1)
    return 0.0


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier: mean (p - y)^2 for binary y."""
    return float(np.mean((np.asarray(y_prob).ravel() - np.asarray(y_true).ravel()) ** 2))


def roc_auc_upset(y_binary: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC for binary upset (1=upset) vs continuous score (higher=more likely upset). Returns 0.5 if constant labels."""
    y_b = np.asarray(y_binary).ravel()
    if np.unique(y_b).size < 2:
        return 0.5
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_b, np.asarray(y_score).ravel()))
    except Exception:
        return 0.5


def ndcg_at_4(y_true_rank: np.ndarray, y_score: np.ndarray) -> float:
    """NDCG@4 for ranking: y_true_rank is ground-truth rank (1=best, 30=worst). Relevance = 30 - rank + 1."""
    relevance = (30.0 - np.asarray(y_true_rank).ravel() + 1.0).clip(1, 30)
    return ndcg_score(relevance, np.asarray(y_score).ravel(), k=4)


def brier_champion(y_onehot: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score for champion prediction: y_onehot is 1 for champion, 0 else; y_prob are championship probabilities."""
    return brier_score(np.asarray(y_onehot).ravel(), np.asarray(y_prob).ravel())
