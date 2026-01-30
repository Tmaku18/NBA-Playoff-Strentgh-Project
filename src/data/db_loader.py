"""Load raw CSV/Parquet into DuckDB. Build games from team MATCHUP (@ = away, vs = home)."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .db import get_connection
from .db_schema import create_schema


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _parse_matchup(matchup: str) -> tuple[str | None, str | None]:
    """Return (away_abbrev, home_abbrev). 'BOS @ MIA' => (BOS,MIA); 'MIA vs. BOS' => (BOS,MIA)."""
    if pd.isna(matchup):
        return None, None
    s = str(matchup).strip()
    if " @ " in s:
        parts = s.split(" @ ", 1)
        return parts[0].strip(), parts[1].strip()
    m = re.split(r"\s+vs\.?\s+", s, flags=re.I, maxsplit=1)
    if len(m) == 2:
        return m[1].strip(), m[0].strip()  # home first, away second
    return None, None


def _season_from_grp(grp: pd.DataFrame) -> str:
    if "SEASON_ID" in grp.columns and grp["SEASON_ID"].notna().any():
        v = str(int(float(grp["SEASON_ID"].dropna().iloc[0])))
        if len(v) >= 5 and v.startswith("2"):
            y = v[1:5]
            return f"{y}-{str(int(y) % 100 + 1).zfill(2)}"
    d = pd.to_datetime(grp["GAME_DATE"].dropna().iloc[0])
    y = d.year
    if d.month >= 10:
        return f"{y}-{str((y + 1) % 100).zfill(2)}"
    return f"{y - 1}-{str(y % 100).zfill(2)}"


def _min_int(x) -> int | None:
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return int(x)
    s = str(x)
    if ":" in s:
        parts = s.split(":")
        return int(float(parts[0]) * 60 + float(parts[1])) if len(parts) >= 2 else int(float(parts[0]) or 0)
    return int(float(s))


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).strip().upper() if isinstance(c, str) else c)


def load_raw_into_db(
    raw_dir: str | Path,
    db_path: str | Path,
    seasons: list[str] | None = None,
) -> None:
    """
    Load team and player logs from raw_dir into DuckDB.
    Builds games from team MATCHUP; creates teams, players, games, team_game_logs, player_game_logs.
    Uses ON CONFLICT for idempotent inserts.
    """
    raw_dir = Path(raw_dir)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    con = get_connection(db_path)
    create_schema(con)

    for t in ("player_game_logs", "team_game_logs", "games", "players", "teams"):
        con.execute(f"DELETE FROM {t}")

    def _season_from_stem(stem: str) -> str | None:
        m = re.search(r"(\d{4})_(\d{2})", stem)
        return f"{m.group(1)}-{m.group(2)}" if m else None

    team_paths = sorted(raw_dir.glob("team_logs_*.parquet")) + sorted(raw_dir.glob("team_logs_*.csv"))
    player_paths = sorted(raw_dir.glob("player_logs_*.parquet")) + sorted(raw_dir.glob("player_logs_*.csv"))
    if seasons:
        sset = set(seasons)
        team_paths = [p for p in team_paths if _season_from_stem(p.stem) in sset]
        player_paths = [p for p in player_paths if _season_from_stem(p.stem) in sset]

    # 1) Teams from team logs
    all_teams = []
    all_team_rows = []
    for path in team_paths:
        df = _read_raw(path)
        if df.empty:
            continue
        df = _norm(df)
        for c in ("TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN",
                  "PTS", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PLUS_MINUS"):
            if c not in df.columns:
                df[c] = None
        all_team_rows.append(df)
        u = df[["TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME"]].drop_duplicates(subset=["TEAM_ID"])
        all_teams.append(u)
    if all_teams:
        teams_df = pd.concat(all_teams, ignore_index=True).drop_duplicates(subset=["TEAM_ID"])
        for _, r in teams_df.iterrows():
            con.execute(
                "INSERT INTO teams (team_id, abbreviation, name, conference) VALUES (?, ?, ?, ?) ON CONFLICT (team_id) DO UPDATE SET abbreviation=excluded.abbreviation, name=excluded.name",
                [int(r["TEAM_ID"]), str(r.get("TEAM_ABBREVIATION", "")), str(r.get("TEAM_NAME", "")), None],
            )

    # 2) Games + team_game_logs from team rows
    abbrev_to_id = {}
    if all_teams:
        for _, r in teams_df.iterrows():
            abbrev_to_id[str(r.get("TEAM_ABBREVIATION", ""))] = int(r["TEAM_ID"])

    for df in all_team_rows:
        for gid, grp in df.groupby("GAME_ID"):
            grp = grp.reset_index(drop=True)
            if len(grp) < 2:
                continue
            m0 = str(grp.iloc[0].get("MATCHUP") or "")
            away_abbr, home_abbr = _parse_matchup(m0)
            if not away_abbr or not home_abbr:
                away_abbr, home_abbr = _parse_matchup(grp.iloc[1].get("MATCHUP"))
            if not away_abbr or not home_abbr:
                home_abbr = str(grp.iloc[0].get("TEAM_ABBREVIATION", ""))
                away_abbr = str(grp.iloc[1].get("TEAM_ABBREVIATION", ""))
            home_row = grp[grp["TEAM_ABBREVIATION"].astype(str) == home_abbr]
            away_row = grp[grp["TEAM_ABBREVIATION"].astype(str) == away_abbr]
            home_tid = int(home_row["TEAM_ID"].iloc[0]) if len(home_row) else (abbrev_to_id.get(home_abbr) or 0)
            away_tid = int(away_row["TEAM_ID"].iloc[0]) if len(away_row) else (abbrev_to_id.get(away_abbr) or 0)
            gdate = pd.to_datetime(grp["GAME_DATE"].iloc[0]).date() if pd.notna(grp["GAME_DATE"].iloc[0]) else None
            seas = _season_from_grp(grp)
            con.execute(
                "INSERT INTO games (game_id, game_date, season, home_team_id, away_team_id) VALUES (?, ?, ?, ?, ?) ON CONFLICT (game_id) DO UPDATE SET game_date=excluded.game_date, season=excluded.season, home_team_id=excluded.home_team_id, away_team_id=excluded.away_team_id",
                [str(gid), gdate, seas, home_tid, away_tid],
            )
            for _, r in grp.iterrows():
                tid = int(r["TEAM_ID"])
                is_home = 1 if str(r.get("TEAM_ABBREVIATION", "")) == home_abbr else 0
                con.execute(
                    """INSERT INTO team_game_logs (game_id, team_id, is_home, wl, min, pts, oreb, dreb, reb, ast, stl, blk, tov, fgm, fga, fg3m, fg3a, ftm, fta, plus_minus)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (game_id, team_id) DO UPDATE SET is_home=excluded.is_home, wl=excluded.wl, min=excluded.min, pts=excluded.pts, oreb=excluded.oreb, dreb=excluded.dreb, reb=excluded.reb, ast=excluded.ast, stl=excluded.stl, blk=excluded.blk, tov=excluded.tov, fgm=excluded.fgm, fga=excluded.fga, fg3m=excluded.fg3m, fg3a=excluded.fg3a, ftm=excluded.ftm, fta=excluded.fta, plus_minus=excluded.plus_minus""",
                    [str(gid), tid, is_home, str(r.get("WL") or ""), _min_int(r.get("MIN")),
                     int(r["PTS"]) if pd.notna(r.get("PTS")) else None, int(r["OREB"]) if pd.notna(r.get("OREB")) else None,
                     int(r["DREB"]) if pd.notna(r.get("DREB")) else None, int(r["REB"]) if pd.notna(r.get("REB")) else None,
                     int(r["AST"]) if pd.notna(r.get("AST")) else None, int(r["STL"]) if pd.notna(r.get("STL")) else None,
                     int(r["BLK"]) if pd.notna(r.get("BLK")) else None, int(r["TOV"]) if pd.notna(r.get("TOV")) else None,
                     int(r["FGM"]) if pd.notna(r.get("FGM")) else None, int(r["FGA"]) if pd.notna(r.get("FGA")) else None,
                     int(r["FG3M"]) if pd.notna(r.get("FG3M")) else None, int(r["FG3A"]) if pd.notna(r.get("FG3A")) else None,
                     int(r["FTM"]) if pd.notna(r.get("FTM")) else None, int(r["FTA"]) if pd.notna(r.get("FTA")) else None,
                     int(r["PLUS_MINUS"]) if pd.notna(r.get("PLUS_MINUS")) else None],
                )

    # 3) Players + player_game_logs (bulk for speed)
    all_player_dfs: list[pd.DataFrame] = []
    for path in player_paths:
        df = _read_raw(path)
        if df.empty:
            continue
        df = _norm(df)
        for m in ("PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "PLUS_MINUS"):
            if m not in df.columns:
                df[m] = None
        all_player_dfs.append(df)

    if all_player_dfs:
        combined = pd.concat(all_player_dfs, ignore_index=True)
        # Unique players
        up = combined[["PLAYER_ID", "PLAYER_NAME"]].dropna(subset=["PLAYER_ID"]).drop_duplicates(subset=["PLAYER_ID"])
        for _, r in up.iterrows():
            con.execute(
                "INSERT INTO players (player_id, player_name) VALUES (?, ?) ON CONFLICT (player_id) DO UPDATE SET player_name=excluded.player_name",
                [int(r["PLAYER_ID"]), str(r.get("PLAYER_NAME") or "")],
            )
        # player_game_logs: filter and build bulk df
        pgl = combined.dropna(subset=["PLAYER_ID", "GAME_ID", "TEAM_ID"])
        pgl = pgl[pgl["GAME_ID"].astype(str).str.len() > 0]
        if not pgl.empty:
            pgl_df = pd.DataFrame({
                "game_id": pgl["GAME_ID"].astype(str),
                "player_id": pgl["PLAYER_ID"].astype(int),
                "team_id": pgl["TEAM_ID"].astype(int),
                "min": pgl["MIN"].apply(_min_int),
                "pts": pgl["PTS"],
                "reb": pgl["REB"],
                "ast": pgl["AST"],
                "stl": pgl["STL"],
                "blk": pgl["BLK"],
                "tov": pgl["TOV"],
                "fgm": pgl["FGM"],
                "fga": pgl["FGA"],
                "fg3m": pgl["FG3M"],
                "fg3a": pgl["FG3A"],
                "ftm": pgl["FTM"],
                "fta": pgl["FTA"],
                "oreb": pgl["OREB"],
                "dreb": pgl["DREB"],
                "plus_minus": pgl["PLUS_MINUS"],
            })
            con.register("_pgl", pgl_df)
            con.execute("""
                INSERT INTO player_game_logs (game_id, player_id, team_id, min, pts, reb, ast, stl, blk, tov, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, plus_minus)
                SELECT game_id, player_id, team_id, min, pts, reb, ast, stl, blk, tov, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, plus_minus FROM _pgl
                ON CONFLICT (game_id, player_id) DO UPDATE SET team_id=excluded.team_id, min=excluded.min, pts=excluded.pts, reb=excluded.reb, ast=excluded.ast, stl=excluded.stl, blk=excluded.blk, tov=excluded.tov, fgm=excluded.fgm, fga=excluded.fga, fg3m=excluded.fg3m, fg3a=excluded.fg3a, ftm=excluded.ftm, fta=excluded.fta, oreb=excluded.oreb, dreb=excluded.dreb, plus_minus=excluded.plus_minus
            """)
            con.unregister("_pgl")

    con.close()


def load_playoff_into_db(
    raw_dir: str | Path,
    db_path: str | Path,
    seasons: list[str] | None = None,
) -> None:
    """
    Load playoff team and player logs from raw_dir into DuckDB playoff_* tables.
    Expects teams to already exist (run load_raw_into_db first).
    File naming: playoffs_team_logs_{y1}_{y2}.parquet, playoffs_player_logs_{y1}_{y2}.parquet.
    """

    def _season_from_stem(stem: str) -> str | None:
        m = re.search(r"(\d{4})_(\d{2})", stem)
        return f"{m.group(1)}-{m.group(2)}" if m else None

    raw_dir = Path(raw_dir)
    con = get_connection(db_path)
    create_schema(con)

    for t in ("playoff_player_game_logs", "playoff_team_game_logs", "playoff_games"):
        try:
            con.execute(f"DELETE FROM {t}")
        except Exception:
            pass

    team_paths = sorted(raw_dir.glob("playoffs_team_logs_*.parquet")) + sorted(raw_dir.glob("playoffs_team_logs_*.csv"))
    player_paths = sorted(raw_dir.glob("playoffs_player_logs_*.parquet")) + sorted(raw_dir.glob("playoffs_player_logs_*.csv"))
    if seasons:
        sset = set(seasons)
        team_paths = [p for p in team_paths if _season_from_stem(p.stem) in sset]
        player_paths = [p for p in player_paths if _season_from_stem(p.stem) in sset]

    abbrev_to_id: dict[str, int] = {}
    teams_df = con.execute("SELECT team_id, abbreviation FROM teams").df()
    for _, r in teams_df.iterrows():
        abbrev_to_id[str(r.get("abbreviation", "")).strip()] = int(r["team_id"])

    all_team_rows: list[pd.DataFrame] = []
    for path in team_paths:
        df = _read_raw(path)
        if df.empty:
            continue
        df = _norm(df)
        for c in ("TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "PLUS_MINUS"):
            if c not in df.columns:
                df[c] = None
        all_team_rows.append(df)

    for df in all_team_rows:
        for gid, grp in df.groupby("GAME_ID"):
            grp = grp.reset_index(drop=True)
            if len(grp) < 2:
                continue
            m0 = str(grp.iloc[0].get("MATCHUP") or "")
            away_abbr, home_abbr = _parse_matchup(m0)
            if not away_abbr or not home_abbr:
                away_abbr, home_abbr = _parse_matchup(grp.iloc[1].get("MATCHUP"))
            if not away_abbr or not home_abbr:
                home_abbr = str(grp.iloc[0].get("TEAM_ABBREVIATION", ""))
                away_abbr = str(grp.iloc[1].get("TEAM_ABBREVIATION", ""))
            home_tid = abbrev_to_id.get(home_abbr) or int(grp[grp["TEAM_ABBREVIATION"].astype(str) == home_abbr]["TEAM_ID"].iloc[0]) if len(grp[grp["TEAM_ABBREVIATION"].astype(str) == home_abbr]) else 0
            away_tid = abbrev_to_id.get(away_abbr) or int(grp[grp["TEAM_ABBREVIATION"].astype(str) == away_abbr]["TEAM_ID"].iloc[0]) if len(grp[grp["TEAM_ABBREVIATION"].astype(str) == away_abbr]) else 0
            if home_tid == 0 and len(grp[grp["TEAM_ABBREVIATION"].astype(str) == home_abbr]):
                home_tid = int(grp[grp["TEAM_ABBREVIATION"].astype(str) == home_abbr]["TEAM_ID"].iloc[0])
            if away_tid == 0 and len(grp[grp["TEAM_ABBREVIATION"].astype(str) == away_abbr]):
                away_tid = int(grp[grp["TEAM_ABBREVIATION"].astype(str) == away_abbr]["TEAM_ID"].iloc[0])
            gdate = pd.to_datetime(grp["GAME_DATE"].iloc[0]).date() if pd.notna(grp["GAME_DATE"].iloc[0]) else None
            seas = _season_from_grp(grp)
            con.execute(
                """INSERT INTO playoff_games (game_id, game_date, season, home_team_id, away_team_id)
VALUES (?, ?, ?, ?, ?) ON CONFLICT (game_id) DO UPDATE SET game_date=excluded.game_date, season=excluded.season, home_team_id=excluded.home_team_id, away_team_id=excluded.away_team_id""",
                [str(gid), gdate, seas, home_tid, away_tid],
            )
            for _, r in grp.iterrows():
                tid = int(r["TEAM_ID"])
                is_home = 1 if str(r.get("TEAM_ABBREVIATION", "")) == home_abbr else 0
                con.execute(
                    """INSERT INTO playoff_team_game_logs (game_id, team_id, is_home, wl, min, pts, oreb, dreb, reb, ast, stl, blk, tov, fgm, fga, fg3m, fg3a, ftm, fta, plus_minus)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (game_id, team_id) DO UPDATE SET is_home=excluded.is_home, wl=excluded.wl, min=excluded.min, pts=excluded.pts, oreb=excluded.oreb, dreb=excluded.dreb, reb=excluded.reb, ast=excluded.ast, stl=excluded.stl, blk=excluded.blk, tov=excluded.tov, fgm=excluded.fgm, fga=excluded.fga, fg3m=excluded.fg3m, fg3a=excluded.fg3a, ftm=excluded.ftm, fta=excluded.fta, plus_minus=excluded.plus_minus""",
                    [str(gid), tid, is_home, str(r.get("WL") or ""), _min_int(r.get("MIN")),
                     int(r["PTS"]) if pd.notna(r.get("PTS")) else None, int(r["OREB"]) if pd.notna(r.get("OREB")) else None,
                     int(r["DREB"]) if pd.notna(r.get("DREB")) else None, int(r["REB"]) if pd.notna(r.get("REB")) else None,
                     int(r["AST"]) if pd.notna(r.get("AST")) else None, int(r["STL"]) if pd.notna(r.get("STL")) else None,
                     int(r["BLK"]) if pd.notna(r.get("BLK")) else None, int(r["TOV"]) if pd.notna(r.get("TOV")) else None,
                     int(r["FGM"]) if pd.notna(r.get("FGM")) else None, int(r["FGA"]) if pd.notna(r.get("FGA")) else None,
                     int(r["FG3M"]) if pd.notna(r.get("FG3M")) else None, int(r["FG3A"]) if pd.notna(r.get("FG3A")) else None,
                     int(r["FTM"]) if pd.notna(r.get("FTM")) else None, int(r["FTA"]) if pd.notna(r.get("FTA")) else None,
                     int(r["PLUS_MINUS"]) if pd.notna(r.get("PLUS_MINUS")) else None],
                )

    all_player_dfs: list[pd.DataFrame] = []
    for path in player_paths:
        df = _read_raw(path)
        if df.empty:
            continue
        df = _norm(df)
        for m in ("PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "PLUS_MINUS"):
            if m not in df.columns:
                df[m] = None
        all_player_dfs.append(df)

    if all_player_dfs:
        combined = pd.concat(all_player_dfs, ignore_index=True)
        pgl = combined.dropna(subset=["PLAYER_ID", "GAME_ID", "TEAM_ID"])
        pgl = pgl[pgl["GAME_ID"].astype(str).str.len() > 0]
        playoff_game_ids = set(con.execute("SELECT game_id FROM playoff_games").df()["game_id"].astype(str))
        pgl = pgl[pgl["GAME_ID"].astype(str).isin(playoff_game_ids)]
        if not pgl.empty:
            for _, r in pgl.drop_duplicates(subset=["PLAYER_ID"]).iterrows():
                con.execute(
                    "INSERT INTO players (player_id, player_name) VALUES (?, ?) ON CONFLICT (player_id) DO UPDATE SET player_name=excluded.player_name",
                    [int(r["PLAYER_ID"]), str(r.get("PLAYER_NAME") or "")],
                )
            pgl_df = pd.DataFrame({
                "game_id": pgl["GAME_ID"].astype(str),
                "player_id": pgl["PLAYER_ID"].astype(int),
                "team_id": pgl["TEAM_ID"].astype(int),
                "min": pgl["MIN"].apply(_min_int),
                "pts": pgl["PTS"], "reb": pgl["REB"], "ast": pgl["AST"], "stl": pgl["STL"], "blk": pgl["BLK"], "tov": pgl["TOV"],
                "fgm": pgl["FGM"], "fga": pgl["FGA"], "fg3m": pgl["FG3M"], "fg3a": pgl["FG3A"], "ftm": pgl["FTM"], "fta": pgl["FTA"],
                "oreb": pgl["OREB"], "dreb": pgl["DREB"], "plus_minus": pgl["PLUS_MINUS"],
            })
            con.register("_ppgl", pgl_df)
            con.execute("""
                INSERT INTO playoff_player_game_logs (game_id, player_id, team_id, min, pts, reb, ast, stl, blk, tov, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, plus_minus)
                SELECT game_id, player_id, team_id, min, pts, reb, ast, stl, blk, tov, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, plus_minus FROM _ppgl
                ON CONFLICT (game_id, player_id) DO UPDATE SET team_id=excluded.team_id, min=excluded.min, pts=excluded.pts, reb=excluded.reb, ast=excluded.ast, stl=excluded.stl, blk=excluded.blk, tov=excluded.tov, fgm=excluded.fgm, fga=excluded.fga, fg3m=excluded.fg3m, fg3a=excluded.fg3a, ftm=excluded.ftm, fta=excluded.fta, oreb=excluded.oreb, dreb=excluded.dreb, plus_minus=excluded.plus_minus
            """)
            con.unregister("_ppgl")

    con.close()


def load_playoff_data(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load playoff_games, playoff_team_game_logs (with game_date), playoff_player_game_logs (with game_date).
    Returns (playoff_games, playoff_tgl, playoff_pgl). Snake_case columns.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")
    con = get_connection(path, read_only=True)
    try:
        pg = con.execute("SELECT * FROM playoff_games").df()
    except Exception:
        con.close()
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if pg.empty:
        con.close()
        return pg, pd.DataFrame(), pd.DataFrame()
    pg["game_date"] = pd.to_datetime(pg["game_date"]).dt.date
    ptgl = con.execute("""
        SELECT tgl.*, g.game_date
        FROM playoff_team_game_logs tgl
        JOIN playoff_games g ON tgl.game_id = g.game_id
    """).df()
    ptgl["game_date"] = pd.to_datetime(ptgl["game_date"]).dt.date
    ppgl = con.execute("""
        SELECT pgl.*, g.game_date
        FROM playoff_player_game_logs pgl
        JOIN playoff_games g ON pgl.game_id = g.game_id
    """).df()
    ppgl["game_date"] = pd.to_datetime(ppgl["game_date"]).dt.date
    con.close()
    return pg, ptgl, ppgl


def load_training_data(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load games, team_game_logs (with game_date), teams, player_game_logs (with game_date)
    from DuckDB for training. Raises if db_path does not exist.
    Returns (games, tgl, teams, pgl) with snake_case columns.
    """
    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found: {path}. Run scripts 1_download_raw and 2_build_db first."
        )
    con = get_connection(path, read_only=True)
    games = con.execute("SELECT * FROM games").df()
    teams = con.execute("SELECT * FROM teams").df()
    if games.empty:
        con.close()
        return games, pd.DataFrame(), teams, pd.DataFrame()
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date
    tgl = con.execute("""
        SELECT tgl.*, g.game_date
        FROM team_game_logs tgl
        JOIN games g ON tgl.game_id = g.game_id
    """).df()
    tgl["game_date"] = pd.to_datetime(tgl["game_date"]).dt.date
    pgl = con.execute("""
        SELECT pgl.*, g.game_date
        FROM player_game_logs pgl
        JOIN games g ON pgl.game_id = g.game_id
    """).df()
    pgl["game_date"] = pd.to_datetime(pgl["game_date"]).dt.date
    con.close()
    return games, tgl, teams, pgl
