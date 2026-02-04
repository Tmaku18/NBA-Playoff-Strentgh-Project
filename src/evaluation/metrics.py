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


def mrr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_n_teams: int = 2,
) -> float:
    """MRR: 1/rank of first team in top N (by actual rank) in predicted order.
    top_n_teams=2: champion + runner-up (either conference).
    top_n_teams=4: conference finals (top 4 teams).
    y_true = relevance (higher = better, e.g. 31 - actual_rank).
    """
    order = np.argsort(y_score)[::-1]
    y = np.asarray(y_true)
    max_rel = float(np.max(y))
    min_rel = max_rel - top_n_teams + 1
    for i in range(len(order)):
        if y[order[i]] >= min_rel - 1e-9:
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


def ndcg_at_10(y_true_rank: np.ndarray, y_score: np.ndarray) -> float:
    """NDCG@10 for ranking: y_true_rank is ground-truth rank (1=best, 30=worst). Relevance = 30 - rank + 1."""
    relevance = (30.0 - np.asarray(y_true_rank).ravel() + 1.0).clip(1, 30)
    return ndcg_score(relevance, np.asarray(y_score).ravel(), k=10)


def brier_champion(y_onehot: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score for champion prediction: y_onehot is 1 for champion, 0 else; y_prob are championship probabilities."""
    return brier_score(np.asarray(y_onehot).ravel(), np.asarray(y_prob).ravel())


def rank_mae(y_pred_rank: np.ndarray, y_actual_rank: np.ndarray) -> float:
    """Mean Absolute Error of rank predictions vs actual ranks (1=best, 30=worst). Lower is better."""
    pred = np.asarray(y_pred_rank).ravel()
    actual = np.asarray(y_actual_rank).ravel()
    if len(pred) != len(actual) or len(pred) == 0:
        return float("nan")
    return float(np.mean(np.abs(pred - actual)))


def rank_rmse(y_pred_rank: np.ndarray, y_actual_rank: np.ndarray) -> float:
    """Root Mean Squared Error of rank predictions vs actual ranks. Lower is better; penalizes large errors more."""
    pred = np.asarray(y_pred_rank).ravel()
    actual = np.asarray(y_actual_rank).ravel()
    if len(pred) != len(actual) or len(pred) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((pred - actual) ** 2)))


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: weighted mean of |acc(bin) - conf(bin)| over n_bins. y_true binary 0/1, y_prob in [0,1]."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_prob = np.clip(y_prob, 1e-8, 1.0 - 1e-8)
    n = len(y_true)
    if n == 0 or n_bins < 1:
        return 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece_val += mask.sum() / n * abs(acc - conf)
    return float(ece_val)
