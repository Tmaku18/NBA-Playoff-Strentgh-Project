"""nba_api wrapper: LeagueGameLog with rate limiting and optional parquet cache."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

try:
    from nba_api.stats.endpoints import LeagueGameLog
    from nba_api.stats.library.parameters import PlayerOrTeamAbbreviation
    _HAS_NBA_API = True
except ImportError:
    _HAS_NBA_API = False


_RATE_DELAY = 0.6  # seconds between requests


def _season_to_api(season: str) -> str:
    """'2023-24' -> '2023-24' (nba_api uses same format)."""
    return season


def fetch_season_logs(
    season: str,
    raw_dir: str | Path,
    *,
    kind: str = "T",
    use_cache: bool = True,
    cache_fmt: str = "parquet",
    season_type: str = "Regular Season",
) -> pd.DataFrame:
    """
    Fetch LeagueGameLog for one season.
    kind: 'T' = team logs, 'P' = player logs.
    season_type: 'Regular Season' or 'Playoffs' (Play-In excluded when using Playoffs).
    Writes to raw_dir and returns DataFrame. Uses cache if use_cache and file exists.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    abbrev = "P" if kind.upper() == "P" else "T"
    season_api = _season_to_api(season)
    suf = "team" if kind.upper() == "T" else "player"
    y1, y2 = season_api.split("-")[0], season_api.split("-")[1]
    prefix = "playoffs_" if season_type == "Playoffs" else ""
    stem = f"{prefix}{suf}_logs_{y1}_{y2}"
    cache_path = raw_dir / f"{stem}.{cache_fmt}"

    if use_cache and cache_path.exists():
        if cache_fmt == "parquet":
            return pd.read_parquet(cache_path)
        return pd.read_csv(cache_path)

    if not _HAS_NBA_API:
        raise ImportError("nba_api is required for fetch. Install: pip install nba_api")

    time.sleep(_RATE_DELAY)
    ep = LeagueGameLog(
        player_or_team_abbreviation=abbrev,
        season=season_api,
        season_type_all_star=season_type,
    )
    df = ep.get_data_frames()[0]
    if df is None or df.empty:
        df = pd.DataFrame()

    if cache_path.suffix.lower() == ".parquet":
        df.to_parquet(cache_path, index=False)
    else:
        df.to_csv(cache_path, index=False)

    return df
