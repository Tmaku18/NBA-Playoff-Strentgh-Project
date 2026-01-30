"""Roster set builder: top-N by minutes as-of date, pad to 15, key_padding_mask. Hash-trick for unseen players."""
from __future__ import annotations

from typing import Any

import pandas as pd


def hash_trick_index(player_id: int | str, num_embeddings: int) -> int:
    return hash(str(player_id)) % num_embeddings


def get_roster_as_of_date(
    pgl: pd.DataFrame,
    team_id: int,
    as_of_date: str | pd.Timestamp,
    *,
    date_col: str = "game_date",
    player_id_col: str = "player_id",
    team_id_col: str = "team_id",
    min_col: str = "min",
    n: int = 15,
    season_start: str | pd.Timestamp | None = None,
    latest_team_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    """
    For one team and as_of_date, select rows with game_date < as_of_date (optionally season-scoped),
    keep only players whose latest team matches team_id, sum minutes per player, take top-N by minutes,
    return a DataFrame with player_id, total_min, rank.
    """
    ad = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
    dates = pd.to_datetime(pgl[date_col]).dt.date
    mask = (pgl[team_id_col] == team_id) & (dates < ad)
    if season_start is not None:
        ss = pd.to_datetime(season_start).date() if isinstance(season_start, str) else season_start
        mask &= dates >= ss
    past = pgl.loc[mask]
    if past.empty:
        return pd.DataFrame(columns=[player_id_col, "total_min", "rank"])

    if latest_team_map is None:
        latest_team_map = latest_team_map_as_of(
            pgl,
            as_of_date,
            date_col=date_col,
            player_id_col=player_id_col,
            team_id_col=team_id_col,
            season_start=season_start,
        )
    if latest_team_map:
        latest_team = past[player_id_col].map(latest_team_map)
        past = past.loc[latest_team == team_id]
        if past.empty:
            return pd.DataFrame(columns=[player_id_col, "total_min", "rank"])

    tot = past.groupby(player_id_col, as_index=False)[min_col].sum()
    tot = tot[tot[min_col] > 0].nlargest(n, min_col).reset_index(drop=True)
    tot["rank"] = range(len(tot))
    return tot.rename(columns={player_id_col: "player_id", min_col: "total_min"})


def latest_team_map_as_of(
    pgl: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    *,
    date_col: str = "game_date",
    player_id_col: str = "player_id",
    team_id_col: str = "team_id",
    season_start: str | pd.Timestamp | None = None,
) -> dict[int, int]:
    """Return player_id -> latest team_id as of date (optionally season-scoped)."""
    ad = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
    dates = pd.to_datetime(pgl[date_col]).dt.date
    mask = dates < ad
    if season_start is not None:
        ss = pd.to_datetime(season_start).date() if isinstance(season_start, str) else season_start
        mask &= dates >= ss
    past = pgl.loc[mask, [player_id_col, team_id_col, date_col]]
    if past.empty:
        return {}
    past = past.copy()
    past[date_col] = pd.to_datetime(past[date_col])
    past = past.sort_values(date_col)
    latest = past.drop_duplicates(subset=[player_id_col], keep="last")
    return {
        int(pid): int(tid)
        for pid, tid in zip(latest[player_id_col].tolist(), latest[team_id_col].tolist())
    }


def build_roster_set(
    roster_df: pd.DataFrame,
    player_stats: pd.DataFrame,
    *,
    player_id_col: str = "player_id",
    n_pad: int = 15,
    stat_cols: list[str] | None = None,
    num_embeddings: int = 500,
) -> tuple[list[int], list[list[float]], list[float], list[bool]]:
    """
    From roster_df (top-N from get_roster_as_of_date) and player_stats (rolling stats keyed by player_id),
    build:
    - embedding_indices: list of length n_pad (hash_trick for each player; 0 for padding)
    - player_stats_matrix: list of n_pad lists of stat values (0 for padding)
    - minutes_weights: list of n_pad (e.g. total_min/max or 0 for padding)
    - key_padding_mask: list of n_pad bools, True = ignore (padded), False = valid
    """
    if stat_cols is None:
        stat_cols = ["pts_L10", "reb_L10", "ast_L10", "stl_L10", "blk_L10", "tov_L10", "availability_L10"]

    order = roster_df.sort_values("rank")["player_id"].tolist()
    pad = n_pad - len(order)
    if pad < 0:
        order = order[:n_pad]
        pad = 0

    embedding_indices: list[int] = []
    rows: list[list[float]] = []
    minutes_weights: list[float] = []
    key_padding_mask: list[bool] = []

    max_min = float(roster_df["total_min"].max()) if "total_min" in roster_df.columns and len(roster_df) else 1.0
    for pid in order:
        embedding_indices.append(hash_trick_index(pid, num_embeddings))
        r = player_stats[player_stats[player_id_col] == pid]
        vec = [float(r[c].iloc[0]) if c in r.columns and len(r) and pd.notna(r[c].iloc[0]) else 0.0 for c in stat_cols]
        rows.append(vec)
        m = float(roster_df.loc[roster_df["player_id"] == pid, "total_min"].iloc[0]) if pid in roster_df["player_id"].values else 0.0
        minutes_weights.append(m / max_min if max_min else 0.0)
        key_padding_mask.append(False)

    for _ in range(pad):
        embedding_indices.append(num_embeddings)  # padding index, distinct from hash range [0, num_embeddings-1]
        rows.append([0.0] * len(stat_cols))
        minutes_weights.append(0.0)
        key_padding_mask.append(True)

    return embedding_indices, rows, minutes_weights, key_padding_mask
