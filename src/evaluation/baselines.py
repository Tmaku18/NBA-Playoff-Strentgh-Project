"""Baselines: rank-by-SRS, rank-by-Net-Rating (from off/def, not as feature)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_by_srs(
    df: pd.DataFrame,
    team_col: str = "team_id",
    srs_col: str = "srs",
) -> pd.DataFrame:
    """Return df with team_col and rank (1=best by SRS)."""
    u = df[[team_col, srs_col]].drop_duplicates()
    u = u.dropna(subset=[srs_col])
    u["rank"] = u[srs_col].rank(ascending=False, method="min").astype(int)
    return u


def rank_by_net_rating(
    df: pd.DataFrame,
    team_col: str = "team_id",
    off_col: str = "off_rtg",
    def_col: str = "def_rtg",
) -> pd.DataFrame:
    """Net rating = off_rtg - def_rtg. Rank 1=best. Not used as feature per plan."""
    u = df[[team_col, off_col, def_col]].copy()
    u = u.dropna(subset=[off_col, def_col])
    u["net_rtg"] = u[off_col] - u[def_col]
    u["rank"] = u["net_rtg"].rank(ascending=False, method="min").astype(int)
    return u
