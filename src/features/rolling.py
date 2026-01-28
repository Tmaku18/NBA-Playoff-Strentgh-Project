"""Rolling features with strict t-1: shift(1) before rolling. L10/L30, DNP, availability fraction."""
from __future__ import annotations

import pandas as pd


def compute_rolling_stats(
    df: pd.DataFrame,
    *,
    player_id_col: str = "player_id",
    date_col: str = "game_date",
    windows: list[int] | None = None,
    stat_cols: list[str] | None = None,
    as_of_date: str | None = None,
    min_col: str = "min",
) -> pd.DataFrame:
    """
    Compute rolling per-game averages over L10/L30 using only past data (t-1).
    Apply shift(1) before rolling so the value for row i uses rows 0..i-1.
    DNP: compute over games played; add availability = fraction of games played in window.
    If as_of_date is set, filter to rows with date_col < as_of_date before computing.
    """
    if windows is None:
        windows = [10, 30]
    if stat_cols is None:
        stat_cols = ["pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]
    # normalize column names to lower for lookups
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if as_of_date is not None:
        ad = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
        df = df[df[date_col].dt.date < ad].copy()

    out_cols = [player_id_col, date_col]
    by_player = df.sort_values([player_id_col, date_col]).groupby(player_id_col, sort=False)

    for w in windows:
        for c in stat_cols:
            if c not in df.columns:
                continue
            # shifted then rolling: value at row i = mean of rows i-w..i-1
            roll = by_player[c].transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
            df[f"{c}_L{w}"] = roll
            if f"{c}_L{w}" not in out_cols:
                out_cols.append(f"{c}_L{w}")

        # availability: fraction of games in window with min > 0 (or not null)
        if min_col in df.columns:
            played = (df[min_col].fillna(0) > 0).astype(float)
            avail = by_player[played.name].transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
            df[f"availability_L{w}"] = avail
            if f"availability_L{w}" not in out_cols:
                out_cols.append(f"availability_L{w}")

    return df
