"""Playoff performance ground truth: wins per team, rank 1-30 (playoff wins, tiebreak reg-season win %, lottery 17-30)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def get_playoff_wins(
    playoff_games: pd.DataFrame,
    playoff_tgl: pd.DataFrame,
    season: str,
) -> dict[int, int]:
    """
    Playoff wins per team for one season (exclude Play-In; we use only games in playoff_* tables).
    Returns team_id -> count of wins (WL='W' in playoff_team_game_logs).
    """
    if playoff_games.empty or playoff_tgl.empty:
        return {}
    pg = playoff_games.copy()
    pt = playoff_tgl.copy()
    if "season" not in pg.columns and "game_date" in pg.columns:
        pg["season"] = pd.to_datetime(pg["game_date"]).dt.to_period("Y").astype(str)
    if "season" in pg.columns:
        gids = set(pg.loc[pg["season"].astype(str) == str(season), "game_id"].astype(str))
    else:
        gids = set(pg["game_id"].astype(str))
    pt = pt[pt["game_id"].astype(str).isin(gids)]
    wl_col = "wl" if "wl" in pt.columns else "WL"
    if wl_col not in pt.columns:
        return {}
    pt["win"] = (pt[wl_col].astype(str).str.upper() == "W").astype(int)
    wins = pt.groupby("team_id")["win"].sum().to_dict()
    return {int(k): int(v) for k, v in wins.items()}


def get_reg_season_win_pct(
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    season: str,
) -> dict[int, float]:
    """Regular-season win % per team for one season. Returns team_id -> win rate (0-1)."""
    if games.empty or tgl.empty:
        return {}
    g = games.copy()
    t = tgl.copy()
    if "season" not in g.columns:
        g["game_date"] = pd.to_datetime(g["game_date"])
        g["season"] = g["game_date"].dt.year.astype(str)
    g = g[g["season"].astype(str) == str(season)]
    gids = set(g["game_id"].astype(str))
    t = t[t["game_id"].astype(str).isin(gids)]
    wl_col = "wl" if "wl" in t.columns else "WL"
    if wl_col not in t.columns:
        return {}
    t["win"] = (t[wl_col].astype(str).str.upper() == "W").astype(int)
    agg = t.groupby("team_id").agg({"win": "sum", "game_id": "count"}).rename(columns={"game_id": "gp"})
    agg["win_pct"] = agg["win"] / agg["gp"].clip(lower=1)
    return agg["win_pct"].to_dict()


def compute_playoff_performance_rank(
    playoff_games: pd.DataFrame,
    playoff_tgl: pd.DataFrame,
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    season: str,
    all_team_ids: list[int] | None = None,
) -> dict[int, int]:
    """
    Playoff performance rank 1-30 for one season.
    Phase 1: Rank playoff teams by total playoff wins (desc).
    Phase 2: Tie-break by regular-season win % (desc).
    Phase 3: Teams with 0 playoff wins ranked 17-30 by regular-season win %.
    Returns team_id -> rank (1-30) for the top 30 teams by this scheme.
    """
    pw = get_playoff_wins(playoff_games, playoff_tgl, season)
    reg_wp = get_reg_season_win_pct(games, tgl, season)
    if all_team_ids is None:
        all_team_ids = sorted(set(list(pw.keys()) + list(reg_wp.keys())))
    if not all_team_ids:
        return {}

    def _safe_wp(tid: int) -> float:
        v = reg_wp.get(tid, 0.0)
        return float(v) if pd.notna(v) else 0.0

    # Teams with at least one playoff win: sort by (playoff_wins desc, reg_win_pct desc, team_id asc)
    playoff_teams = [tid for tid in all_team_ids if pw.get(tid, 0) > 0]
    playoff_teams.sort(key=lambda t: (-pw.get(t, 0), -_safe_wp(t), t))

    # Lottery teams (0 playoff wins): sort by reg_win_pct desc, team_id asc
    lottery = [tid for tid in all_team_ids if pw.get(tid, 0) == 0]
    lottery.sort(key=lambda t: (-_safe_wp(t), t))

    playoff_top = playoff_teams[:16]
    lottery_top = lottery[:14]

    out: dict[int, int] = {}
    for r, tid in enumerate(playoff_top, start=1):
        out[tid] = r
    for r, tid in enumerate(lottery_top, start=17):
        out[tid] = r
    return out


def get_playoff_finish_label(
    playoff_wins: dict[int, int],
    team_id: int,
) -> str:
    """Human-readable finish for reporting, e.g. 'NBA Finals Runner-Up (13 Playoff Wins)'."""
    w = playoff_wins.get(team_id, 0)
    if w == 0:
        return "Did not qualify"
    if w >= 16:
        return "Champion (16 Playoff Wins)"
    if w >= 12:
        return "Finals Runner-Up"
    if w >= 8:
        return "Conference Finals"
    if w >= 4:
        return "Second Round"
    return f"First Round ({w} Playoff Wins)"


def load_playoff_rank_for_season(
    db_path: str | Path,
    season: str,
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    teams: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    """
    Load playoff data from DB and compute playoff rank, wins, and finish labels for one season.
    Returns (team_id -> playoff_rank, team_id -> playoff_wins, team_id -> finish_label).
    """
    from src.data.db_loader import load_playoff_data

    path = Path(db_path)
    if not path.exists():
        return {}, {}, {}
    pg, ptgl, _ = load_playoff_data(db_path)
    if pg.empty or ptgl.empty:
        return {}, {}, {}
    all_team_ids = sorted(teams["team_id"].astype(int).unique().tolist()) if not teams.empty else None
    rank = compute_playoff_performance_rank(pg, ptgl, games, tgl, season, all_team_ids=all_team_ids)
    wins = get_playoff_wins(pg, ptgl, season)
    labels = {tid: get_playoff_finish_label(wins, tid) for tid in rank}
    return rank, wins, labels
