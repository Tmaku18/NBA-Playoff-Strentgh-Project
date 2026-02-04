"""Build ListMLE batches from DB-loaded (games, tgl, teams, pgl) for Model A."""
from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from src.features.build_roster_set import build_roster_set, get_roster_as_of_date, latest_team_map_as_of
from src.features.lineup_continuity import pct_min_returning_per_team
from src.features.on_off import get_on_court_pm_as_of_date
from src.features.rolling import (
    ON_OFF_STAT_COLS,
    PLAYER_STAT_COLS_L10_L30,
    PLAYER_STAT_COLS_WITH_ON_OFF,
    get_player_stats_as_of_date,
    get_prior_season_stats,
)
from src.training.build_lists import build_lists


def build_batches_from_lists(
    lists: list[dict[str, Any]],
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    teams: pd.DataFrame,
    pgl: pd.DataFrame,
    config: dict,
    *,
    device: torch.device | str = "cpu",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Build ListMLE batches for a given subset of lists. Returns (batches, list_metas).
    list_metas[i] = {"team_ids": [...], "as_of_date": str, "win_rates": [...]} for the i-th batch.
    
    If training.use_prior_season_baseline is true, players with all-zero stats will have
    their stats filled from prior season averages.
    """
    if pgl.empty or tgl.empty or games.empty or not lists:
        return [], []

    ma = config.get("model_a", {})
    training_cfg = config.get("training", {})
    num_emb = ma.get("num_embeddings", 500)
    roster_size = training_cfg.get("roster_size", 15)
    roster_debug = bool((config.get("logging") or {}).get("roster_debug", False))
    
    # Prior season baseline config
    use_prior_baseline = bool(training_cfg.get("use_prior_season_baseline", False))
    lookback_days = int(training_cfg.get("prior_season_lookback_days", 365))
    
    # Cache for prior season stats (keyed by season_start)
    prior_stats_cache: dict[str, pd.DataFrame] = {}

    batches: list[dict[str, Any]] = []
    list_metas: list[dict[str, Any]] = []
    for lst in lists:
        as_of_date = lst["as_of_date"]
        season_start = _season_start_for_date(config, as_of_date)
        team_ids = lst["team_ids"]
        win_rates = lst["win_rates"]
        final_rank = lst.get("final_rank")
        rel_values = [31.0 - float(r) for r in final_rank] if final_rank else win_rates
        if len(team_ids) < 2:
            continue
        
        # Compute or retrieve prior season stats if baseline is enabled
        prior_season_stats = None
        if use_prior_baseline and season_start:
            if season_start not in prior_stats_cache:
                prior_stats_cache[season_start] = get_prior_season_stats(
                    pgl, season_start,
                    stat_cols=PLAYER_STAT_COLS_L10_L30,
                    lookback_days=lookback_days,
                )
            prior_season_stats = prior_stats_cache[season_start]
        
        player_stats_df = get_player_stats_as_of_date(
            pgl, as_of_date,
            windows=training_cfg.get("rolling_windows", [10, 30]),
            stat_cols=PLAYER_STAT_COLS_L10_L30,
            prior_season_stats=prior_season_stats,
        )
        on_court_df = get_on_court_pm_as_of_date(pgl, tgl, games, as_of_date)
        if not on_court_df.empty:
            player_stats_df = player_stats_df.merge(on_court_df, on="player_id", how="left")
        for c in ON_OFF_STAT_COLS:
            if c not in player_stats_df.columns:
                player_stats_df[c] = 0.0

        continuity = pct_min_returning_per_team(
            pgl, games, as_of_date, team_ids=list(team_ids),
            season_start=season_start,
        )
        latest_team_map = latest_team_map_as_of(
            pgl,
            as_of_date,
            season_start=season_start,
            debug=roster_debug,
            warn_missing_season=True,
        )
        embs_list = []
        stats_list = []
        min_list = []
        mask_list = []
        player_ids_per_team: list[list[int | None]] = []
        for tid in team_ids:
            roster = get_roster_as_of_date(
                pgl,
                int(tid),
                as_of_date,
                n=roster_size,
                season_start=season_start,
                latest_team_map=latest_team_map,
                debug=roster_debug,
                warn_missing_season=True,
            )
            order = roster.sort_values("rank")["player_id"].tolist() if not roster.empty else []
            pad = roster_size - len(order)
            if pad < 0:
                order = order[:roster_size]
                pad = 0
            player_ids_per_team.append([int(pid) for pid in order] + [None] * pad)
            emb, rows, minutes, mask = build_roster_set(
                roster,
                player_stats_df,
                n_pad=roster_size,
                stat_cols=PLAYER_STAT_COLS_WITH_ON_OFF,
                num_embeddings=num_emb,
                team_continuity_scalar=continuity.get(int(tid), 0.0),
            )
            embs_list.append(emb)
            stats_list.append(rows)
            min_list.append(minutes)
            mask_list.append(mask)
        K, P = len(team_ids), roster_size
        embedding_indices = torch.tensor(embs_list, dtype=torch.long, device=device).unsqueeze(0)
        player_stats = torch.tensor(stats_list, dtype=torch.float32, device=device).unsqueeze(0)
        minutes = torch.tensor(min_list, dtype=torch.float32, device=device).unsqueeze(0)
        key_padding_mask = torch.tensor(mask_list, dtype=torch.bool, device=device).unsqueeze(0)
        rel = torch.tensor([rel_values], dtype=torch.float32, device=device)
        # Skip batches where every team has all players masked (no valid roster for any team)
        all_masked_per_team = key_padding_mask.squeeze(0).all(dim=-1)
        if all_masked_per_team.all().item():
            if roster_debug:
                print(
                    f"Batch skip: all {len(team_ids)} teams have empty roster (as_of_date={as_of_date}).",
                    flush=True,
                )
            continue
        batches.append({
            "embedding_indices": embedding_indices,
            "player_stats": player_stats,
            "minutes": minutes,
            "key_padding_mask": key_padding_mask,
            "rel": rel,
        })
        list_metas.append({
            "team_ids": list(team_ids),
            "as_of_date": as_of_date,
            "win_rates": list(win_rates),
            "rel_values": rel_values,
            "player_ids_per_team": player_ids_per_team,
        })
    return batches, list_metas


def build_batches_from_db(
    games: pd.DataFrame,
    tgl: pd.DataFrame,
    teams: pd.DataFrame,
    pgl: pd.DataFrame,
    config: dict,
    *,
    device: torch.device | str = "cpu",
    playoff_games: pd.DataFrame | None = None,
    playoff_tgl: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """
    Build ListMLE batches from DB data. Each batch = one conference-date list:
    embedding_indices (1, K, P), player_stats (1, K, P, 7), minutes, key_padding_mask, rel (1, K).
    
    If training.use_prior_season_baseline is true, players with all-zero stats will have
    their stats filled from prior season averages.
    playoff_games, playoff_tgl: required when listmle_target=playoff_outcome.
    """
    if pgl.empty or tgl.empty or games.empty:
        return []

    lists = build_lists(
        tgl, games, teams,
        config=config,
        playoff_games=playoff_games,
        playoff_tgl=playoff_tgl,
    )
    ma = config.get("model_a", {})
    training_cfg = config.get("training", {})
    num_emb = ma.get("num_embeddings", 500)
    roster_size = training_cfg.get("roster_size", 15)
    stat_dim = int(ma.get("stat_dim", 14))
    roster_debug = bool((config.get("logging") or {}).get("roster_debug", False))
    
    # Prior season baseline config
    use_prior_baseline = bool(training_cfg.get("use_prior_season_baseline", False))
    lookback_days = int(training_cfg.get("prior_season_lookback_days", 365))
    
    # Cache for prior season stats (keyed by season_start)
    prior_stats_cache: dict[str, pd.DataFrame] = {}

    batches: list[dict[str, Any]] = []
    for lst in lists:
        as_of_date = lst["as_of_date"]
        season_start = _season_start_for_date(config, as_of_date)
        team_ids = lst["team_ids"]
        win_rates = lst["win_rates"]
        final_rank = lst.get("final_rank")
        rel_values = [31.0 - float(r) for r in final_rank] if final_rank else win_rates
        if len(team_ids) < 2:
            continue
        
        # Compute or retrieve prior season stats if baseline is enabled
        prior_season_stats = None
        if use_prior_baseline and season_start:
            if season_start not in prior_stats_cache:
                prior_stats_cache[season_start] = get_prior_season_stats(
                    pgl, season_start,
                    stat_cols=PLAYER_STAT_COLS_L10_L30,
                    lookback_days=lookback_days,
                )
            prior_season_stats = prior_stats_cache[season_start]
        
        windows = training_cfg.get("rolling_windows", [10, 30])
        player_stats_df = get_player_stats_as_of_date(
            pgl, as_of_date,
            windows=windows,
            stat_cols=PLAYER_STAT_COLS_L10_L30,
            prior_season_stats=prior_season_stats,
        )
        on_court_df = get_on_court_pm_as_of_date(pgl, tgl, games, as_of_date)
        if not on_court_df.empty:
            player_stats_df = player_stats_df.merge(on_court_df, on="player_id", how="left")
        for c in ON_OFF_STAT_COLS:
            if c not in player_stats_df.columns:
                player_stats_df[c] = 0.0

        continuity = pct_min_returning_per_team(
            pgl, games, as_of_date, team_ids=list(team_ids),
            season_start=season_start,
        )
        latest_team_map = latest_team_map_as_of(
            pgl,
            as_of_date,
            season_start=season_start,
            debug=roster_debug,
            warn_missing_season=True,
        )
        embs_list: list[list[int]] = []
        stats_list: list[list[list[float]]] = []
        min_list: list[list[float]] = []
        mask_list: list[list[bool]] = []
        for tid in team_ids:
            roster = get_roster_as_of_date(
                pgl,
                int(tid),
                as_of_date,
                n=roster_size,
                season_start=season_start,
                latest_team_map=latest_team_map,
                debug=roster_debug,
                warn_missing_season=True,
            )
            emb, rows, minutes, mask = build_roster_set(
                roster,
                player_stats_df,
                n_pad=roster_size,
                stat_cols=PLAYER_STAT_COLS_WITH_ON_OFF,
                num_embeddings=num_emb,
                team_continuity_scalar=continuity.get(int(tid), 0.0),
            )
            embs_list.append(emb)
            stats_list.append(rows)
            min_list.append(minutes)
            mask_list.append(mask)
        K, P = len(team_ids), roster_size
        embedding_indices = torch.tensor(embs_list, dtype=torch.long, device=device).unsqueeze(0)  # (1, K, P)
        player_stats = torch.tensor(stats_list, dtype=torch.float32, device=device).unsqueeze(0)  # (1, K, P, 7)
        minutes = torch.tensor(min_list, dtype=torch.float32, device=device).unsqueeze(0)  # (1, K, P)
        key_padding_mask = torch.tensor(mask_list, dtype=torch.bool, device=device).unsqueeze(0)  # (1, K, P)
        rel = torch.tensor([rel_values], dtype=torch.float32, device=device)  # (1, K)
        all_masked_per_team = key_padding_mask.squeeze(0).all(dim=-1)
        if all_masked_per_team.all().item():
            if roster_debug:
                print(
                    f"Batch skip: all {len(team_ids)} teams have empty roster (as_of_date={as_of_date}).",
                    flush=True,
                )
            continue
        batches.append({
            "embedding_indices": embedding_indices,
            "player_stats": player_stats,
            "minutes": minutes,
            "key_padding_mask": key_padding_mask,
            "rel": rel,
        })
    return batches


def _season_start_for_date(config: dict, as_of_date: str) -> str | None:
    seasons_cfg = config.get("seasons") or {}
    if not seasons_cfg:
        return None
    d = pd.to_datetime(as_of_date).date()
    for season, rng in seasons_cfg.items():
        start = pd.to_datetime(rng.get("start")).date()
        end = pd.to_datetime(rng.get("end")).date()
        if start <= d <= end:
            return str(start)
    return None
