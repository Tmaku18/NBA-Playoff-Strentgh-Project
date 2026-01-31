"""Team-context features for Model B: Four Factors, pace, SOS/SRS. FORBIDDEN: net_rating."""

from __future__ import annotations

import pandas as pd

from .four_factors import four_factors_from_team_logs

FORBIDDEN = {"net_rating", "NET_RATING", "net rating"}


def build_team_context(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    *,
    sos_srs: pd.DataFrame | None = None,
    team_key: str = "team_id",
    season_key: str = "season",
) -> pd.DataFrame:
    """
    Build Model B feature set: Four Factors (eFG, TOV%, FT_rate, ORB%), pace, SOS, SRS.
    sos_srs: optional with team_abbreviation or team_id, season, sos, srs.
    Enforce: no net_rating in the output. FORBIDDEN is checked by leakage_tests.
    """
    ff = four_factors_from_team_logs(tgl, games)

    # pace: from games we need poss. Approx: 0.96 * (FGA + 0.44*FTA - ORB + TOV) per team; sum both teams per game / 2?
    # Simpler: use POSS from tgl if available; else approximate from tgl: FGA + 0.44*FTA - ORB + TOV (one team).
    if "fga" in tgl.columns and "fta" in tgl.columns and "oreb" in tgl.columns and "tov" in tgl.columns:
        tgl = tgl.copy()
        tgl["_poss"] = tgl["fga"].fillna(0) + 0.44 * tgl["fta"].fillna(0) - tgl["oreb"].fillna(0) + tgl["tov"].fillna(0)
        pace = tgl.groupby("game_id")["_poss"].sum().reset_index()
        pace = pace.rename(columns={"_poss": "pace"})
    else:
        pace = pd.DataFrame(columns=["game_id", "pace"])

    out = ff.merge(pace, on="game_id", how="left")

    if sos_srs is not None and not sos_srs.empty:
        # join on team+season. games has game_id, season; tgl has game_id, team_id. We need team->abbreviation from elsewhere or sos_srs has team_id.
        if "team_abbreviation" in sos_srs.columns and "team_id" not in sos_srs.columns:
            # would need teams table to map; for now skip if we don't have team_id in sos_srs
            pass
        elif "team_id" in sos_srs.columns and season_key in sos_srs.columns:
            gs = games[["game_id", "season"]].drop_duplicates()
            tgl_s = tgl[["game_id", "team_id"]].drop_duplicates().merge(gs, on="game_id")
            tgl_s = tgl_s.merge(sos_srs, left_on=["team_id", "season"], right_on=["team_id", season_key], how="left")
            out = out.merge(tgl_s[["game_id", "team_id", "sos", "srs"]], on=["game_id", "team_id"], how="left", suffixes=("", "_s"))

    for c in list(out.columns):
        if any(f in str(c).lower() for f in FORBIDDEN):
            raise ValueError(f"Model B must not include net_rating; found: {c}")

    return out


# Model B feature column names (no net_rating)
TEAM_CONTEXT_FEATURE_COLS: list[str] = ["eFG", "TOV_pct", "FT_rate", "ORB_pct", "pace"]


def build_team_context_as_of_dates(
    tgl: pd.DataFrame,
    games: pd.DataFrame,
    team_dates: list[tuple[int, str]],
    *,
    date_col: str = "game_date",
    team_id_col: str = "team_id",
) -> pd.DataFrame:
    """
    Build Model B features per (team_id, as_of_date): season-to-date mean of eFG, TOV_pct, FT_rate, ORB_pct, pace.
    tgl must have game_date (e.g. from load_training_data join). team_dates = [(team_id, as_of_date), ...].
    """
    if not team_dates:
        return pd.DataFrame(columns=[team_id_col, "as_of_date"] + TEAM_CONTEXT_FEATURE_COLS)
    ctx = build_team_context(tgl, games)
    if "game_date" not in ctx.columns and "game_id" in games.columns and "game_date" in games.columns:
        ctx = ctx.merge(games[["game_id", "game_date"]].drop_duplicates(), on="game_id", how="left")
    ctx[date_col] = pd.to_datetime(ctx[date_col]).dt.date
    feat_cols = [c for c in TEAM_CONTEXT_FEATURE_COLS if c in ctx.columns]
    rows = []
    for tid, as_of in team_dates:
        ad = pd.to_datetime(as_of).date() if isinstance(as_of, str) else as_of
        past = ctx[(ctx[team_id_col] == tid) & (ctx[date_col] < ad)]
        if past.empty or not feat_cols:
            rows.append({team_id_col: tid, "as_of_date": as_of, **{c: 0.0 for c in TEAM_CONTEXT_FEATURE_COLS}})
            continue
        agg = past[feat_cols].mean()
        row = {team_id_col: tid, "as_of_date": as_of, **{c: 0.0 for c in TEAM_CONTEXT_FEATURE_COLS}}
        for c in feat_cols:
            row[c] = float(agg[c]) if pd.notna(agg[c]) else 0.0
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in TEAM_CONTEXT_FEATURE_COLS:
        if c not in out.columns:
            out[c] = 0.0
    return out[[team_id_col, "as_of_date"] + TEAM_CONTEXT_FEATURE_COLS]
