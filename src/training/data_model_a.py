"""Build ListMLE batches from DB-loaded (games, tgl, teams, pgl) for Model A."""
from __future__ import annotations

from typing import Any

import pandas as pd
import torch

from src.features.build_roster_set import build_roster_set, get_roster_as_of_date
from src.features.rolling import PLAYER_STAT_COLS_L10, get_player_stats_as_of_date
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
    """
    if pgl.empty or tgl.empty or games.empty or not lists:
        return [], []

    ma = config.get("model_a", {})
    num_emb = ma.get("num_embeddings", 500)
    roster_size = config.get("training", {}).get("roster_size", 15)

    batches: list[dict[str, Any]] = []
    list_metas: list[dict[str, Any]] = []
    for lst in lists:
        as_of_date = lst["as_of_date"]
        team_ids = lst["team_ids"]
        win_rates = lst["win_rates"]
        if len(team_ids) < 2:
            continue
        player_stats_df = get_player_stats_as_of_date(pgl, as_of_date, stat_cols=PLAYER_STAT_COLS_L10)
        embs_list = []
        stats_list = []
        min_list = []
        mask_list = []
        player_ids_per_team: list[list[int | None]] = []
        for tid in team_ids:
            roster = get_roster_as_of_date(pgl, int(tid), as_of_date, n=roster_size)
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
                stat_cols=PLAYER_STAT_COLS_L10,
                num_embeddings=num_emb,
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
        rel = torch.tensor([win_rates], dtype=torch.float32, device=device)
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
) -> list[dict[str, Any]]:
    """
    Build ListMLE batches from DB data. Each batch = one conference-date list:
    embedding_indices (1, K, P), player_stats (1, K, P, 7), minutes, key_padding_mask, rel (1, K).
    """
    if pgl.empty or tgl.empty or games.empty:
        return []

    lists = build_lists(tgl, games, teams)
    ma = config.get("model_a", {})
    num_emb = ma.get("num_embeddings", 500)
    roster_size = config.get("training", {}).get("roster_size", 15)
    stat_dim = 7

    batches: list[dict[str, Any]] = []
    for lst in lists:
        as_of_date = lst["as_of_date"]
        team_ids = lst["team_ids"]
        win_rates = lst["win_rates"]
        if len(team_ids) < 2:
            continue
        player_stats_df = get_player_stats_as_of_date(pgl, as_of_date, stat_cols=PLAYER_STAT_COLS_L10)
        embs_list: list[list[int]] = []
        stats_list: list[list[list[float]]] = []
        min_list: list[list[float]] = []
        mask_list: list[list[bool]] = []
        for tid in team_ids:
            roster = get_roster_as_of_date(pgl, int(tid), as_of_date, n=roster_size)
            emb, rows, minutes, mask = build_roster_set(
                roster,
                player_stats_df,
                n_pad=roster_size,
                stat_cols=PLAYER_STAT_COLS_L10,
                num_embeddings=num_emb,
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
        rel = torch.tensor([win_rates], dtype=torch.float32, device=device)  # (1, K)
        batches.append({
            "embedding_indices": embedding_indices,
            "player_stats": player_stats,
            "minutes": minutes,
            "key_padding_mask": key_padding_mask,
            "rel": rel,
        })
    return batches
