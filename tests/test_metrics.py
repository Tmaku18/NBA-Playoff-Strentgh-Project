"""Tests for src.evaluation.metrics."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics import (
    brier_champion,
    brier_score,
    ndcg_at_4,
    ndcg_at_10,
    ndcg_score,
    mrr,
    roc_auc_upset,
    spearman,
)


def test_ndcg_score_perfect_order():
    relevance = np.array([3.0, 2.0, 1.0])
    score = np.array([3.0, 2.0, 1.0])  # same order
    assert ndcg_score(relevance, score, k=3) == pytest.approx(1.0, abs=1e-6)


def test_ndcg_score_worst_order():
    relevance = np.array([3.0, 2.0, 1.0])
    score = np.array([1.0, 2.0, 3.0])  # reversed
    out = ndcg_score(relevance, score, k=3)
    assert 0 <= out <= 1 and out < 1.0


def test_ndcg_score_k_cap():
    relevance = np.array([3.0, 2.0, 1.0, 0.0])
    score = np.array([3.0, 2.0, 1.0, 0.0])
    assert ndcg_score(relevance, score, k=2) == pytest.approx(1.0, abs=1e-6)


def test_spearman_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert spearman(y, y) == pytest.approx(1.0, abs=1e-6)


def test_spearman_anti():
    y = np.array([1.0, 2.0, 3.0])
    assert spearman(y, -y) == pytest.approx(-1.0, abs=1e-6)


def test_spearman_constant_returns_zero():
    y = np.array([1.0, 1.0, 1.0])
    s = np.array([1.0, 2.0, 3.0])
    r = spearman(y, s)
    assert np.isfinite(r) and -1 <= r <= 1


def test_mrr_first_is_best():
    relevance = np.array([3.0, 1.0, 2.0])  # max rel = 3; top 2 = rel >= 2
    score = np.array([10.0, 1.0, 2.0])  # top by score is index 0 (rel 3)
    assert mrr(relevance, score, top_n_teams=2) == pytest.approx(1.0, abs=1e-6)


def test_mrr_second_is_best():
    relevance = np.array([1.0, 3.0, 2.0])  # top 2 = rel >= 2
    score = np.array([10.0, 9.0, 1.0])  # order 0, 1, 2; first top-2 at pos 2
    assert mrr(relevance, score, top_n_teams=2) == pytest.approx(0.5, abs=1e-6)


def test_brier_score_perfect():
    y = np.array([1.0, 0.0, 1.0])
    p = np.array([1.0, 0.0, 1.0])
    assert brier_score(y, p) == pytest.approx(0.0, abs=1e-6)


def test_brier_score_nonzero():
    y = np.array([1.0, 0.0])
    p = np.array([0.5, 0.5])
    b = brier_score(y, p)
    assert b > 0 and np.isfinite(b)


def test_roc_auc_upset_constant_labels():
    y = np.array([0.0, 0.0, 0.0])
    s = np.array([1.0, 2.0, 3.0])
    assert roc_auc_upset(y, s) == 0.5


def test_ndcg_at_4():
    rank = np.array([1.0, 2.0, 3.0, 4.0])  # 1=best
    score = np.array([4.0, 3.0, 2.0, 1.0])  # higher = better predicted
    out = ndcg_at_4(rank, score)
    assert 0 <= out <= 1 and np.isfinite(out)


def test_ndcg_at_10():
    rank = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # 1=best
    score = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # higher = better predicted
    out = ndcg_at_10(rank, score)
    assert 0 <= out <= 1 and np.isfinite(out)


def test_brier_champion():
    onehot = np.array([0.0, 0.0, 1.0, 0.0])  # team 3 is champion
    prob = np.array([0.1, 0.2, 0.5, 0.2])
    b = brier_champion(onehot, prob)
    assert b >= 0 and np.isfinite(b)
