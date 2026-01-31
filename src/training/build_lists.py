"""Training list construction for ListMLE: conference-date lists, target = standings-to-date (win-rate)."""
from __future__ import annotations

from typing import Any

import pandas as pd

# Fallback when teams.conference is null
TEAM_CONFERENCE: dict[str, str] = {
    "BOS": "E", "BKN": "E", "NYK": "E", "PHI": "E", "TOR": "E",
    "CHI": "E", "CLE": "E", "DET": "E", "IND": "E", "MIL": "E",
    "ATL": "E", "CHA": "E", "MIA": "E", "ORL": "E", "WAS": "E",
    "DAL": "W", "HOU": "W", "MEM": "W", "NOP": "W", "SAS": "W",
    "DEN": "W", "MIN": "W", "OKC": "W", "POR": "W", "UTA": "W",
    "GSW": "W", "LAC": "W", "LAL": "W", "PHX": "W", "SAC": "W",
}


def _win_rate_to_date(games: pd.DataFrame, as_of: pd.Timestamp | str, team_id_col: str = "team_id") -> pd.DataFrame:
    """For each team, compute W and total games with game_date < as_of. Return df with team_id, w, g, win_rate."""
    ad = pd.to_datetime(as_of).date() if isinstance(as_of, str) else as_of
    g = games.copy()
    g["game_date"] = pd.to_datetime(g["game_date"]).dt.date
    g = g[g["game_date"] < ad]
    # games has home_team_id, away_team_id. We need to melt to (game_id, team_id, w). w=1 if team won.
    home = g[["game_id", "home_team_id"]].copy()
    home["team_id"] = home["home_team_id"]
    away = g[["game_id", "away_team_id"]].copy()
    away["team_id"] = away["away_team_id"]
    # we need WL from team_game_logs. Pass in tgl or we need to join. Simpler: infer from home/away and we need to know who won. games doesn't have WL. We need tgl with (game_id, team_id, wl). wl is 'W' or 'L'.
    return pd.DataFrame(columns=[team_id_col, "w", "g", "win_rate"])


def build_lists_for_conference_date(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    teams: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    conference: str,
    *,
    team_id_col: str = "team_id",
    date_col: str = "game_date",
    wl_col: str = "wl",
) -> list[tuple[Any, ...]]:
    """
    For one (conference, as_of_date): get teams in that conference, compute standings-to-date (win-rate)
    from tgl/games with game_date < as_of_date, sort by win_rate desc. Return list of (team_id, win_rate).
    """
    ad = pd.to_datetime(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
    # teams in conference: from teams or TEAM_CONFERENCE by abbreviation
    if "conference" in teams.columns and teams["conference"].notna().any():
        tc = teams[teams["conference"] == conference][team_id_col].tolist()
    else:
        tc = [
            int(teams.loc[teams["abbreviation"] == abbr, "team_id"].iloc[0])
            for abbr, c in TEAM_CONFERENCE.items() if c == conference
            if abbr in teams["abbreviation"].values
        ]
    if not tc:
        return []

    tgl = tgl.copy()
    tgl[date_col] = pd.to_datetime(tgl[date_col]).dt.date
    past = tgl[tgl[date_col] < ad]
    past = past[past[team_id_col].isin(tc)]

    if past.empty:
        return [(tid, 0.0) for tid in tc]

    past["w"] = (past[wl_col].astype(str).str.upper() == "W").astype(int)
    agg = past.groupby(team_id_col).agg({"w": "sum", "game_id": "nunique"}).rename(columns={"game_id": "g"})
    agg["win_rate"] = agg["w"] / agg["g"].replace(0, 1)
    agg = agg.sort_values("win_rate", ascending=False)
    return [(tid, float(agg.loc[tid, "win_rate"])) for tid in agg.index]


def build_lists(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    teams: pd.DataFrame,
    *,
    by_week: bool = False,
) -> list[dict[str, Any]]:
    """
    Build many conference-date (or conference-week) lists. Each element:
    { "conference": str, "as_of_date": str, "team_ids": [id,...], "win_rates": [float,...] }
    by_week: if True, group dates by week.
    """
    games = games.copy()
    games["game_date"] = pd.to_datetime(games["game_date"])
    dates = sorted(games["game_date"].dt.date.unique())
    if not dates:
        return []
    # Subsample dates when very many to keep runtime reasonable
    max_dates = 200
    if len(dates) > max_dates:
        step = max(1, len(dates) // max_dates)
        dates = dates[::step]

    conferences = ["E", "W"]
    if "conference" in teams.columns and teams["conference"].notna().any():
        conferences = [c for c in teams["conference"].dropna().unique() if c]
    if not conferences:
        conferences = ["E", "W"]

    out: list[dict[str, Any]] = []
    for d in dates:
        for conf in conferences:
            lst = build_lists_for_conference_date(tgl, games, teams, d, conf)
            if len(lst) < 2:
                continue
            out.append({
                "conference": conf,
                "as_of_date": str(d),
                "team_ids": [x[0] for x in lst],
                "win_rates": [x[1] for x in lst],
            })
    return out
