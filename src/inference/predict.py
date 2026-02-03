"""Inference: load A/B/stacker, produce per-team JSON (predicted_strength, ensemble_score, delta, contributors)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


def load_models(
    model_a_path: str | Path | None = None,
    xgb_path: str | Path | None = None,
    rf_path: str | Path | None = None,
    meta_path: str | Path | None = None,
    config: dict | None = None,
):
    """Load Model A, XGB, RF, RidgeCV meta. Returns (model_a, xgb, rf, meta) or Nones."""
    from src.models.deep_set_rank import DeepSetRank

    model_a, xgb, rf, meta = None, None, None, None
    cfg = config or {}
    ma = cfg.get("model_a", {})

    if model_a_path and Path(model_a_path).exists():
        ck = torch.load(model_a_path, map_location="cpu", weights_only=False)
        attn_cfg = ma.get("attention", {})
        stat_dim = int(ma.get("stat_dim", ma.get("expected_stat_dim", 14)))
        model_a = DeepSetRank(
            ma.get("num_embeddings", 500),
            ma.get("embedding_dim", 32),
            stat_dim,
            ma.get("encoder_hidden", [128, 64]),
            ma.get("attention_heads", 4),
            ma.get("dropout", 0.2),
            minutes_bias_weight=float(ma.get("minutes_bias_weight", 0.3)),
            minutes_sum_min=float(ma.get("minutes_sum_min", 1e-6)),
            fallback_strategy=str(ma.get("attention_fallback_strategy", "minutes")),
            attention_temperature=float(attn_cfg.get("temperature", 1.0)),
            attention_input_dropout=float(attn_cfg.get("input_dropout", 0.0)),
            attention_use_pre_norm=bool(attn_cfg.get("use_pre_norm", True)),
            attention_use_residual=bool(attn_cfg.get("use_residual", True)),
        )
        if "model_state" in ck:
            model_a.load_state_dict(ck["model_state"])
        model_a.eval()

    if xgb_path and Path(xgb_path).exists():
        import joblib
        xgb = joblib.load(xgb_path)
        if isinstance(xgb, dict) or not callable(getattr(xgb, "predict", None)):
            xgb = None
    if rf_path and Path(rf_path).exists():
        import joblib
        rf = joblib.load(rf_path)
        if isinstance(rf, dict) or not callable(getattr(rf, "predict", None)):
            rf = None
    if meta_path and Path(meta_path).exists():
        import joblib
        meta = joblib.load(meta_path)
        if isinstance(meta, dict):
            meta = meta.get("model") or meta.get("meta")
        if meta is None or not callable(getattr(meta, "predict", None)):
            meta = None

    return model_a, xgb, rf, meta


def predict_teams(
    team_ids: list[int],
    team_names: list[str],
    model_a_scores: np.ndarray | None = None,
    xgb_scores: np.ndarray | None = None,
    rf_scores: np.ndarray | None = None,
    meta_model: Any = None,
    actual_ranks: dict[int, int] | None = None,
    actual_global_ranks: dict[int, int] | None = None,
    attention_by_team: dict[int, list[tuple[str, float]]] | None = None,
    attention_fallback_by_team: dict[int, bool] | None = None,
    team_id_to_conference: dict[int, str] | None = None,
    playoff_rank: dict[int, int] | None = None,
    eos_playoff_standings: dict[int, int] | None = None,
    model_presence: dict[str, bool] | None = None,
    *,
    true_strength_scale: str = "percentile",
    odds_temperature: float = 1.0,
    championship_odds_method: str = "softmax",
    monte_carlo_config: dict | None = None,
) -> list[dict]:
    """
    Combine base scores, run meta if present. For each team output:
    conference_rank (1-15), predicted_strength (global rank 1-30, used internally for eval), ensemble_score,
    championship_odds, delta, classification, analysis.actual_conference_rank, post_playoff_rank and rank_delta_playoffs when available.
    """
    n = len(team_ids)
    if model_a_scores is not None and len(model_a_scores) == n:
        sa = np.asarray(model_a_scores).ravel()
    else:
        sa = np.zeros(n)
    if xgb_scores is not None and len(xgb_scores) == n:
        sx = np.asarray(xgb_scores).ravel()
    else:
        sx = np.zeros(n)
    if rf_scores is not None and len(rf_scores) == n:
        sr = np.asarray(rf_scores).ravel()
    else:
        sr = np.zeros(n)
    sa = np.nan_to_num(sa, nan=0.0, posinf=0.0, neginf=0.0)
    sx = np.nan_to_num(sx, nan=0.0, posinf=0.0, neginf=0.0)
    sr = np.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)

    X = np.column_stack([sa, sx, sr])
    if meta_model is not None and not isinstance(meta_model, dict) and callable(getattr(meta_model, "predict", None)):
        ens = meta_model.predict(X).ravel()
    else:
        ens = (sa + sx + sr) / 3.0
    ens = np.nan_to_num(ens, nan=0.0, posinf=0.0, neginf=0.0)

    pred_rank = np.argsort(np.argsort(-ens)) + 1  # global rank 1-30
    if true_strength_scale == "percentile":
        rank_order = (np.argsort(np.argsort(ens)) + 1).astype(float)
        tss = (rank_order - 1.0) / (n - 1) if n > 1 else np.zeros(n)
    else:
        tss = (ens - ens.min()) / (ens.max() - ens.min() + 1e-12)

    # Championship odds: softmax(ens / temperature)
    T = max(odds_temperature, 1e-6)
    exp_s = np.exp(np.clip(ens / T, -50, 50))
    odds = exp_s / exp_s.sum()

    # Conference rank (1-15 within E/W)
    team_id_to_conf = team_id_to_conference or {}
    conf_rank: dict[int, int] = {}
    for conf in ("E", "W"):
        idx = [i for i in range(n) if team_id_to_conf.get(team_ids[i], "E") == conf]
        if not idx:
            continue
        sub_ens = ens[idx]
        sub_rank = np.argsort(np.argsort(-sub_ens)) + 1
        for k, i in enumerate(idx):
            conf_rank[team_ids[i]] = int(sub_rank[k])

    actual_ranks = actual_ranks or {}
    actual_global_ranks = actual_global_ranks or {}
    attention_by_team = attention_by_team or {}
    attention_fallback_by_team = attention_fallback_by_team or {}
    playoff_rank = playoff_rank or {}
    eos_playoff_standings = eos_playoff_standings or {}
    model_presence = model_presence or {"a": True, "xgb": True, "rf": True}

    out = []
    for i, (tid, tname) in enumerate(zip(team_ids, team_names)):
        act = actual_ranks.get(tid)
        act_global = actual_global_ranks.get(tid)
        act_for_class = act_global if act_global is not None else act
        delta = (act_for_class - pred_rank[i]) if act_for_class is not None else None
        if delta is not None:
            if delta > 0:
                classification = f"Over-ranked by {delta} slots"
            elif delta < 0:
                classification = f"Under-ranked by {-delta} slots"
            else:
                classification = "Aligned"
        else:
            classification = "Unknown"

        # deep_set_rank: global rank (1-30) by Model A score. Note: Model A was trained with ListMLE
        # on standings-ordered lists; actual_conference_rank is that same list position. So deep_set_rank often
        # matches actual_conference_rank within conference—that is by design (same target), not independent accuracy.
        r_a = np.argsort(np.argsort(-sa))[i] + 1 if model_presence.get("a", True) and len(sa) == n else None
        r_x = np.argsort(np.argsort(-sx))[i] + 1 if model_presence.get("xgb", True) and len(sx) == n else None
        r_r = np.argsort(np.argsort(-sr))[i] + 1 if model_presence.get("rf", True) and len(sr) == n else None
        ranks_present = [r for r in (r_a, r_x, r_r) if r is not None]
        if len(ranks_present) >= 2:
            spread = max(ranks_present) - min(ranks_present)
            threshold_high = max(2, n // 10)
            threshold_med = max(5, n // 5)
            if spread <= threshold_high:
                agreement = "High"
            elif spread <= threshold_med:
                agreement = "Medium"
            else:
                agreement = "Low"
        elif len(ranks_present) == 1:
            agreement = "Single"
        else:
            agreement = "Unknown"

        contrib = attention_by_team.get(tid, [])
        p_rank = playoff_rank.get(tid)
        rank_delta_playoffs = (p_rank - pred_rank[i]) if p_rank is not None else None

        pred_dict: dict[str, Any] = {
            "predicted_strength": int(pred_rank[i]),
            "ensemble_score": float(tss[i]),
            "ensemble_score_100": round(float(tss[i]) * 100.0, 1),
            "conference_rank": conf_rank.get(tid),
            "championship_odds": f"{float(odds[i]) * 100:.1f}%",
        }
        eos_standings = eos_playoff_standings.get(tid) if eos_playoff_standings else None
        analysis_dict: dict[str, Any] = {
            "actual_conference_rank": int(act) if act is not None else None,
            "EOS_global_rank": int(act_global) if act_global is not None else None,
            "EOS_playoff_standings": int(eos_standings) if eos_standings is not None else None,
            "classification": classification,
            "post_playoff_rank": int(p_rank) if p_rank is not None else None,
            "rank_delta_playoffs": int(rank_delta_playoffs) if rank_delta_playoffs is not None else None,
        }

        conf = team_id_to_conference.get(tid) if team_id_to_conference else None
        out.append({
            "team_id": int(tid),
            "team_name": tname,
            "conference": conf,
            "prediction": pred_dict,
            "analysis": analysis_dict,
            "ensemble_diagnostics": {"model_agreement": agreement, "deep_set_rank": int(r_a) if r_a is not None else None, "xgboost_rank": int(r_x) if r_x is not None else None, "random_forest_rank": int(r_r) if r_r is not None else None},
            "roster_dependence": {
                "primary_contributors": [
                    {"player": str(p), "attention_weight": float(w)}
                    for p, w in contrib if np.isfinite(w)
                ],
                "contributors_are_fallback": bool(attention_fallback_by_team.get(int(tid), False)),
            },
        })
    return out


def run_inference_from_db(
    output_dir: str | Path,
    config: dict,
    db_path: str | Path,
    run_id: str | None = None,
) -> Path:
    """Run inference using real DB: load data, build lists for target date, run Model A/B, write predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.data.db import get_connection
    from src.data.db_loader import load_training_data
    from src.features.team_context import build_team_context_as_of_dates
    from src.training.build_lists import TEAM_CONFERENCE, build_lists
    from src.training.data_model_a import build_batches_from_lists
    from src.training.train_model_a import predict_batches_with_attention
    from src.utils.split import date_to_season, load_split_info

    out = Path(output_dir)
    if run_id:
        out = out / run_id
    out.mkdir(parents=True, exist_ok=True)
    # Load models from the outputs directory (same as script 3/4/4b)
    outputs_path = Path(output_dir).resolve()
    model_a, xgb, rf, meta = load_models(
        model_a_path=outputs_path / "best_deep_set.pt",
        xgb_path=outputs_path / "xgb_model.joblib",
        rf_path=outputs_path / "rf_model.joblib",
        meta_path=outputs_path / "ridgecv_meta.joblib",
        config=config,
    )

    games, tgl, teams, pgl = load_training_data(db_path)
    if games.empty or tgl.empty:
        raise ValueError("DB has no games/tgl. Run 2_build_db with raw data first.")
    # Load player_id -> player_name for attention primary_contributors
    player_id_to_name: dict[int, str] = {}
    try:
        con = get_connection(Path(db_path), read_only=True)
        players_df = con.execute("SELECT player_id, player_name FROM players").df()
        con.close()
        if not players_df.empty:
            player_id_to_name = dict(
                zip(players_df["player_id"].astype(int), players_df["player_name"].astype(str))
            )
    except Exception:
        pass
    lists = build_lists(tgl, games, teams)
    if not lists:
        raise ValueError("No lists from build_lists.")
    dates_sorted = sorted(set(lst["as_of_date"] for lst in lists))
    # Use split_info: primary = last test date; optional second run = last train date
    try:
        split_info = load_split_info(Path(output_dir))
        test_dates = split_info.get("test_dates", [])
        train_dates = split_info.get("train_dates", [])
        test_seasons = split_info.get("test_seasons")
    except FileNotFoundError:
        split_info = {}
        test_dates = []
        train_dates = []
        test_seasons = None
    if test_seasons is None:
        test_seasons = config.get("training", {}).get("test_seasons") or []
    seasons_cfg = config.get("seasons") or {}

    run_specs: list[tuple[str | None, list, str, str | None]] = []
    if test_seasons and seasons_cfg and test_dates:
        for season in test_seasons:
            season_dates = [d for d in test_dates if date_to_season(d, seasons_cfg) == season]
            if not season_dates:
                continue
            target_date = sorted(season_dates)[-1]
            target_lists = [lst for lst in lists if lst["as_of_date"] == target_date]
            if not target_lists:
                target_lists = [lst for lst in lists if lst["as_of_date"] == season_dates[-1]]
            if target_lists:
                run_specs.append((target_date, target_lists, f"predictions_{season}.json", season))
    if not run_specs:
        target_date = test_dates[-1] if test_dates else (dates_sorted[-1] if dates_sorted else None)
        target_lists = [lst for lst in lists if lst["as_of_date"] == target_date]
        if not target_lists:
            target_lists = [lists[-1]] if lists else []
        run_specs = [(target_date, target_lists, "predictions.json", None)]
    also_train = bool(config.get("inference", {}).get("also_train_predictions", False))
    if also_train and train_dates:
        train_date = train_dates[-1]
        train_target_lists = [lst for lst in lists if lst["as_of_date"] == train_date]
        if train_target_lists:
            run_specs.append((train_date, train_target_lists, "train_predictions.json", None))

    test_specs = [s for s in run_specs if "train_predictions" not in s[2]]
    train_specs = [s for s in run_specs if "train_predictions" in s[2]]
    if not test_specs:
        test_specs = [(dates_sorted[-1] if dates_sorted else None, lists[-1:] if lists else [], "predictions.json", None)]

    def _run_inference_for_spec(target_date: str | None, target_lists: list, output_file: str, season: str | None, *, draw_figures: bool = True) -> Path:
        pj = out / output_file
        fig_suffix = f"_{season}" if season else ""

            # Flatten to one list of (team_id, as_of_date) across target lists; keep unique team_id for naming/rank
        team_id_to_as_of: dict[int, str] = {}
        team_id_to_actual_rank: dict[int, int] = {}
        team_id_to_win_rate: dict[int, float] = {}
        for lst in target_lists:
            win_rates = lst.get("win_rates", [])
            for idx, tid in enumerate(lst["team_ids"]):
                tid = int(tid)
                if tid not in team_id_to_as_of:
                    team_id_to_as_of[tid] = lst["as_of_date"]
                if tid not in team_id_to_actual_rank:
                    team_id_to_actual_rank[tid] = idx + 1
                if tid not in team_id_to_win_rate:
                    team_id_to_win_rate[tid] = float(win_rates[idx]) if idx < len(win_rates) else 0.0
        unique_team_ids = list(dict.fromkeys(tid for lst in target_lists for tid in lst["team_ids"]))
        unique_team_ids = [int(t) for t in unique_team_ids]
        if not unique_team_ids:
            raise ValueError("No teams in target lists.")
        team_dates = [(tid, team_id_to_as_of.get(tid, target_date or "")) for tid in unique_team_ids]
        as_of_date = target_date or team_dates[0][1]
        win_rate_map = {tid: float(team_id_to_win_rate.get(tid, 0.0)) for tid in unique_team_ids}
        sorted_global = sorted(win_rate_map.items(), key=lambda x: (-x[1], x[0]))
        actual_global_rank = {tid: i + 1 for i, (tid, _) in enumerate(sorted_global)}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tid_to_score_a: dict[int, float] = {}
        attention_by_team: dict[int, list[tuple[str, float]]] = {}  # team_id -> [(player_name, weight), ...]
        attention_fallback_by_team: dict[int, bool] = {}
        team_id_to_batch: dict[int, tuple[int, int]] = {}
        team_id_to_player_ids: dict[int, list[int | None]] = {}
        batches_a: list[dict[str, Any]] = []
        attn_debug = {"teams": 0, "empty_roster": 0, "all_zero": 0, "attn_sum": [], "attn_max": []}
        if model_a is not None:
            model_a = model_a.to(device)
            batches_a, list_metas = build_batches_from_lists(target_lists, games, tgl, teams, pgl, config, device=device)
            if batches_a:
                scores_list, attn_list = predict_batches_with_attention(model_a, batches_a, device)
                for i, meta in enumerate(list_metas):
                    if i >= len(attn_list):
                        break
                    attn_tensor = attn_list[i]
                    for k, tid in enumerate(meta["team_ids"]):
                        tid = int(tid)
                        tid_to_score_a[tid] = float(scores_list[i][0, k].item())
                        player_ids = meta.get("player_ids_per_team", [[]])[k] if k < len(meta.get("player_ids_per_team", [])) else []
                        if tid not in team_id_to_batch:
                            team_id_to_batch[tid] = (i, k)
                            team_id_to_player_ids[tid] = [int(pid) if pid is not None else None for pid in player_ids]
                        attn_weights = attn_tensor[0, k].numpy() if attn_tensor.dim() >= 2 else attn_tensor[k].numpy()
                        attn_weights = np.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
                        max_len = min(len(player_ids), len(attn_weights))
                        attn_weights = attn_weights[:max_len]
                        player_ids = player_ids[:max_len]
                        if max_len == 0:
                            attn_debug["empty_roster"] += 1
                            continue
                        attn_debug["teams"] += 1
                        attn_sum = float(np.sum(attn_weights))
                        attn_max = float(np.max(attn_weights)) if max_len else 0.0
                        attn_debug["attn_sum"].append(attn_sum)
                        attn_debug["attn_max"].append(attn_max)
                        if attn_sum <= 0:
                            attn_debug["all_zero"] += 1
                        order = np.argsort(-attn_weights)
                        contrib: list[tuple[str, float]] = []
                        for idx in order[:10]:
                            if idx >= len(player_ids) or player_ids[idx] is None:
                                continue
                            w = float(attn_weights[idx])
                            if not np.isfinite(w) or w <= 0:
                                continue
                            pid = player_ids[idx]
                            name = player_id_to_name.get(int(pid), f"Player_{pid}")
                            contrib.append((name, w))
                        fallback_used = False
                        if not contrib and max_len > 0:
                            # Fallback 1: take top-k by raw weight even if <= 0
                            fallback_used = True
                            for idx in order[:10]:
                                if idx >= len(player_ids) or player_ids[idx] is None:
                                    continue
                                w = float(attn_weights[idx])
                                if not np.isfinite(w) or w <= 0:
                                    continue
                                pid = player_ids[idx]
                                name = player_id_to_name.get(int(pid), f"Player_{pid}")
                                contrib.append((name, w))
                        if not contrib and max_len > 0 and i < len(batches_a):
                            # Fallback 2: top-3 by minutes when attention yields nothing
                            b = batches_a[i]
                            if "minutes" in b:
                                min_t = b["minutes"]
                                if min_t.dim() >= 3 and k < min_t.shape[1]:
                                    minutes_row = min_t[0, k].cpu().numpy()
                                    order_min = np.argsort(-minutes_row)
                                    for idx in order_min[:3]:
                                        if idx < len(player_ids) and player_ids[idx] is not None:
                                            pid = player_ids[idx]
                                            name = player_id_to_name.get(int(pid), f"Player_{pid}")
                                            contrib.append((name, float(minutes_row[idx])))
                            fallback_used = True
                        if contrib:
                            attention_by_team[tid] = contrib
                        if fallback_used:
                            attention_fallback_by_team[tid] = True
        if attn_debug["teams"] > 0:
            mean_sum = float(np.mean(attn_debug["attn_sum"])) if attn_debug["attn_sum"] else 0.0
            mean_max = float(np.mean(attn_debug["attn_max"])) if attn_debug["attn_max"] else 0.0
            print(
                "Attention debug:",
                f"teams={attn_debug['teams']}",
                f"empty_roster={attn_debug['empty_roster']}",
                f"all_zero={attn_debug['all_zero']}",
                f"attn_sum_mean={mean_sum:.4f}",
                f"attn_max_mean={mean_max:.4f}",
                flush=True,
            )
        sa = np.array([tid_to_score_a.get(tid, 0.0) for tid in unique_team_ids], dtype=np.float32)

        sx = np.zeros(len(unique_team_ids), dtype=np.float32)
        sr = np.zeros(len(unique_team_ids), dtype=np.float32)
        feat_df = build_team_context_as_of_dates(
            tgl, games, team_dates,
            config=config, teams=teams, pgl=pgl,
        )
        if not feat_df.empty and xgb is not None and rf is not None:
            from src.features.team_context import get_team_context_feature_cols
            all_feat = get_team_context_feature_cols(config)
            feat_cols = [c for c in all_feat if c in feat_df.columns]
            if feat_cols:
                for i, tid in enumerate(unique_team_ids):
                    row = feat_df[(feat_df["team_id"] == tid) & (feat_df["as_of_date"] == team_id_to_as_of.get(tid, as_of_date))]
                    if not row.empty:
                        X_row = row[feat_cols].values.astype(np.float32)
                        if xgb is not None:
                            sx[i] = float(xgb.predict(X_row)[0])
                        if rf is not None:
                            sr[i] = float(rf.predict(X_row)[0])

        actual_ranks = {tid: team_id_to_actual_rank.get(tid) for tid in unique_team_ids}
        team_names = []
        for tid in unique_team_ids:
            r = teams[teams["team_id"] == tid]
            name = r["name"].iloc[0] if not r.empty and "name" in r.columns else f"Team_{tid}"
            team_names.append(str(name))

        # Conference map for conference_rank and plot
        team_id_to_conf: dict[int, str] = {}
        abbr_col = "abbreviation" if "abbreviation" in teams.columns else "ABBREVIATION"
        conf_col = "conference" if "conference" in teams.columns else "CONFERENCE"
        for _, row in teams.iterrows():
            tid = int(row["team_id"])
            c = row.get(conf_col)
            if c is not None and str(c).strip():
                c = str(c).strip().upper()
                team_id_to_conf[tid] = "E" if c in ("E", "EAST") else "W" if c in ("W", "WEST") else c[0]
            else:
                abbr = row.get(abbr_col)
                team_id_to_conf[tid] = TEAM_CONFERENCE.get(str(abbr).strip() if abbr is not None else "", "E")

        # Playoff rank, EOS final rank (Option B), and EOS playoff standings for target season
        playoff_rank_map: dict[int, int] = {}
        eos_final_rank_map: dict[int, int] = {}
        eos_playoff_standings_map: dict[int, int] = {}
        eos_rank_source = "standings"
        seasons_cfg = config.get("seasons") or {}
        target_season = None
        season_start = None
        season_end = None
        as_of_d = pd.to_datetime(as_of_date).date() if as_of_date else None
        for season, rng in seasons_cfg.items():
            start = pd.to_datetime(rng.get("start")).date()
            end = pd.to_datetime(rng.get("end")).date()
            if as_of_d and start <= as_of_d <= end:
                target_season = season
                season_start = rng.get("start")
                season_end = rng.get("end")
                break
        if target_season and season_start and season_end:
            try:
                from src.data.db_loader import load_playoff_data
                from src.evaluation.playoffs import (
                    _filtered_playoff_tgl,
                    compute_eos_final_rank,
                    compute_eos_playoff_standings,
                    compute_playoff_performance_rank,
                )
                season_end_d = pd.to_datetime(season_end).date()
                reg_season_complete = as_of_d and as_of_d >= season_end_d
                pg, ptgl, _ = load_playoff_data(db_path)
                if pg is not None and ptgl is not None and not pg.empty and not ptgl.empty:
                    pt_check = _filtered_playoff_tgl(pg, ptgl, target_season)
                    tid_col = "team_id" if "team_id" in pt_check.columns else "TEAM_ID"
                    if not pt_check.empty and len(pt_check[tid_col].unique()) >= 16:
                        reg_season_complete = True
                if reg_season_complete:
                    eos_playoff_standings_map = compute_eos_playoff_standings(
                        games, tgl, target_season,
                        season_start=season_start,
                        season_end=season_end,
                        all_team_ids=unique_team_ids,
                    )
                if not pg.empty and not ptgl.empty:
                    playoff_debug = bool((config.get("logging") or {}).get("playoff_debug", False))
                    playoff_rank_map = compute_playoff_performance_rank(
                        pg, ptgl, games, tgl, target_season,
                        all_team_ids=unique_team_ids,
                        season_start=season_start,
                        season_end=season_end,
                        debug=playoff_debug,
                    )
                    eos_final_rank_map = compute_eos_final_rank(
                        pg, ptgl, games, tgl, target_season,
                        all_team_ids=unique_team_ids,
                        season_start=season_start,
                        season_end=season_end,
                        debug=playoff_debug,
                    )
                    if eos_final_rank_map and len(eos_final_rank_map) >= 16:
                        actual_global_rank = {int(tid): int(r) for tid, r in eos_final_rank_map.items()}
                        eos_rank_source = "eos_final_rank"
            except Exception as e:
                print(
                    f"EOS/playoff rank failed (falling back to standings): {e}",
                    file=sys.stderr,
                )

        if config.get("inference", {}).get("require_eos_final_rank", False) and eos_rank_source != "eos_final_rank":
            print(
                "Inference requires eos_final_rank (playoff-based EOS) but DB returned standings. "
                "Ensure DB has playoff_games and playoff_team_game_logs populated (run 2_build_db with playoff raw data).",
                file=sys.stderr,
            )
            sys.exit(1)

        preds = predict_teams(
            unique_team_ids,
            team_names,
            model_a_scores=sa,
            xgb_scores=sx,
            rf_scores=sr,
            meta_model=meta,
            actual_ranks=actual_ranks,
            actual_global_ranks=actual_global_rank,
            attention_by_team=attention_by_team if attention_by_team else None,
            attention_fallback_by_team=attention_fallback_by_team if attention_fallback_by_team else None,
            team_id_to_conference=team_id_to_conf,
            playoff_rank=playoff_rank_map if playoff_rank_map else None,
            eos_playoff_standings=eos_playoff_standings_map if eos_playoff_standings_map else None,
            model_presence={"a": model_a is not None, "xgb": xgb is not None, "rf": rf is not None},
            true_strength_scale=config.get("output", {}).get("true_strength_scale", "percentile"),
            odds_temperature=float(config.get("output", {}).get("odds_temperature", 1.0)),
            championship_odds_method=config.get("output", {}).get("championship_odds_method", "softmax"),
            monte_carlo_config=config.get("monte_carlo"),
        )

        # Integrated Gradients summary in predictions.json (optional, top-K per conference)
        ig_by_team: dict[int, list[dict[str, Any]]] = {}
        ig_top_k = int(config.get("output", {}).get("ig_inference_top_k", 1))
        if ig_top_k > 0 and model_a is not None and batches_a and team_id_to_batch:
            try:
                from src.viz.integrated_gradients import ig_attr, _HAS_CAPTUM
                if _HAS_CAPTUM:
                    ig_steps = int(config.get("output", {}).get("ig_inference_steps", 50))
                    for conf in ("E", "W"):
                        conf_preds = [t for t in preds if team_id_to_conf.get(t["team_id"], "E") == conf]
                        conf_preds = sorted(
                            conf_preds,
                            key=lambda t: t["prediction"].get("conference_rank") or t["prediction"]["predicted_strength"],
                        )
                        for t in conf_preds[:ig_top_k]:
                            tid = int(t["team_id"])
                            if tid not in team_id_to_batch:
                                continue
                            b_idx, k = team_id_to_batch[tid]
                            if b_idx >= len(batches_a):
                                continue
                            batch = batches_a[b_idx]
                            emb = batch["embedding_indices"][:, k, :]
                            stats = batch["player_stats"][:, k, :, :]
                            minu = batch["minutes"][:, k, :]
                            msk = batch["key_padding_mask"][:, k, :]
                            with torch.no_grad():
                                s_check, _, _ = model_a(emb, stats, minu, msk)
                            if not torch.isfinite(s_check).all():
                                continue
                            attr, _ = ig_attr(model_a, emb, stats, minu, msk, n_steps=ig_steps)
                            if attr is None or attr.numel() == 0:
                                continue
                            attr = torch.nan_to_num(attr, nan=0.0, posinf=0.0, neginf=0.0)
                            if not torch.isfinite(attr).all():
                                continue
                            norms = torch.norm(attr[0].float(), dim=1)
                            if norms.numel() == 0:
                                continue
                            topk = min(5, norms.shape[0])
                            vals, idxs = norms.topk(topk, largest=True)
                            player_ids = team_id_to_player_ids.get(tid, [])
                            contrib: list[dict[str, Any]] = []
                            for v, idx in zip(vals.tolist(), idxs.tolist()):
                                if idx >= len(player_ids) or player_ids[idx] is None:
                                    continue
                                if not np.isfinite(v):
                                    continue
                                pid = player_ids[idx]
                                name = player_id_to_name.get(int(pid), f"Player_{pid}")
                                contrib.append({"player": name, "attribution_norm": float(v)})
                            if contrib:
                                ig_by_team[tid] = contrib
                else:
                    print("Integrated Gradients skipped (captum not installed).", file=sys.stderr)
            except Exception as e:
                print(f"Integrated Gradients inference failed: {e}", file=sys.stderr)

        if ig_by_team:
            for t in preds:
                tid = int(t["team_id"])
                if tid in ig_by_team:
                    rd = t.get("roster_dependence") or {}
                    rd["ig_contributors"] = ig_by_team[tid]
                    t["roster_dependence"] = rd

        pred_payload: dict[str, Any] = {"teams": preds}
        pred_payload["eos_rank_source"] = eos_rank_source
        with open(pj, "w", encoding="utf-8") as f:
            json.dump(pred_payload, f, indent=2, allow_nan=False)

        if draw_figures:
            east_preds = [t for t in preds if team_id_to_conf.get(t["team_id"], "E") == "E"]
            west_preds = [t for t in preds if team_id_to_conf.get(t["team_id"], "W") == "W"]

            fig, (ax_east, ax_west) = plt.subplots(1, 2, figsize=(14, 6))

            def _draw_panel(ax, pred_list, title, marker="o"):
                """East = circle (o), West = diamond (D); color by tab20."""
                if not pred_list:
                    ax.text(0.5, 0.5, f"No {title} teams", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(title)
                    ax.set_xlabel("Actual Conference Rank")
                    ax.set_ylabel("Predicted Conference Rank")
                    ax.grid(True, linestyle="--", alpha=0.7)
                    return
                points = []
                for t in pred_list:
                    pr = t["prediction"].get("conference_rank")
                    ar = t["analysis"].get("actual_conference_rank") or t["analysis"].get("EOS_conference_rank")
                    if pr is None or ar is None:
                        continue
                    points.append((ar, pr, t["team_name"]))
                if not points:
                    ax.text(0.5, 0.5, f"No valid {title} ranks", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(title)
                    ax.set_xlabel("Actual Conference Rank")
                    ax.set_ylabel("Predicted Conference Rank")
                    ax.grid(True, linestyle="--", alpha=0.7)
                    return
                ar, pr, names = zip(*points)
                max_r = max(max(ar or [1]), max(pr or [1]), 1) + 1
                ax.plot([0, max_r], [0, max_r], "k--", alpha=0.5, label="identity")
                cmap = plt.get_cmap("tab20")
                for i, (a, p) in enumerate(zip(ar, pr)):
                    color = cmap(i % 20)
                    ax.scatter(a, p, c=[color], label=names[i], s=60, marker=marker, edgecolors="k", linewidths=0.5)
                ax.set_xlabel("Actual Conference Rank")
                ax.set_ylabel("Predicted Conference Rank")
                ax.set_title(title)
                ax.grid(True, linestyle="--", alpha=0.7)
                ax.legend(loc="best", fontsize=7, ncol=2)
                ax.set_xlim(-0.5, max_r)
                ax.set_ylim(-0.5, max_r)

            _draw_panel(ax_east, east_preds, "East", marker="o")
            _draw_panel(ax_west, west_preds, "West", marker="D")
            fig.suptitle("Predicted vs actual rank (conference rank 1-15)", fontsize=12)
            fig.tight_layout()
            fig.savefig(out / f"pred_vs_actual{fig_suffix}.png", bbox_inches="tight")
            plt.close()

            # pred_vs_playoff_rank: global rank (1-30) vs playoff performance rank (1-30)
            # Legend outside so all points visible; East = circle (o), West = diamond (D); color coordinated
            if playoff_rank_map:
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                pts = [(t["analysis"].get("post_playoff_rank"), t["prediction"].get("global_rank") or t["prediction"]["predicted_strength"], t["team_name"], team_id_to_conf.get(t["team_id"], "E")) for t in preds if t["analysis"].get("post_playoff_rank") is not None]
                if not pts:
                    ax2.text(0.5, 0.5, "No playoff ranks available", ha="center", va="center", transform=ax2.transAxes)
                    ax2.set_xlabel("Playoff performance rank (1-30)")
                    ax2.set_ylabel("Predicted global rank (1-30)")
                    ax2.set_title("Predicted global rank vs playoff performance rank")
                    ax2.grid(True, linestyle="--", alpha=0.7)
                    fig2.savefig(out / f"pred_vs_playoff_rank{fig_suffix}.png", bbox_inches="tight")
                    plt.close(fig2)
                else:
                    east_pts = [(p_rank, g_rank, name) for (p_rank, g_rank, name, conf) in pts if conf == "E"]
                    west_pts = [(p_rank, g_rank, name) for (p_rank, g_rank, name, conf) in pts if conf == "W"]
                    max_r = max(max(g for _, g, _, _ in pts), max(p for p, _, _, _ in pts), 1) + 1
                    ax2.plot([0, max_r], [0, max_r], "k--", alpha=0.5, label="identity")
                    cmap = plt.get_cmap("tab20")
                    for i, (p_rank, g_rank, name) in enumerate(east_pts):
                        ax2.scatter(p_rank, g_rank, c=[cmap(i % 20)], label=name, s=50, marker="o", edgecolors="k", linewidths=0.5)
                    for i, (p_rank, g_rank, name) in enumerate(west_pts):
                        ax2.scatter(p_rank, g_rank, c=[cmap((len(east_pts) + i) % 20)], label=name, s=50, marker="D", edgecolors="k", linewidths=0.5)
                    ax2.set_xlabel("Playoff performance rank (1-30)")
                    ax2.set_ylabel("Predicted global rank (1-30)")
                    ax2.set_title("Predicted global rank vs playoff performance rank")
                    ax2.grid(True, linestyle="--", alpha=0.7)
                    ax2.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=6, ncol=1)
                    ax2.set_xlim(-0.5, max_r)
                    ax2.set_ylim(-0.5, max_r)
                    fig2.tight_layout(rect=[0, 0, 0.72, 1])
                    fig2.savefig(out / f"pred_vs_playoff_rank{fig_suffix}.png", bbox_inches="tight")
                    plt.close(fig2)

            # Championship odds top-10 bar chart
            sorted_preds = sorted(preds, key=lambda t: float(t["prediction"]["championship_odds"].rstrip("%")), reverse=True)[:10]
            if sorted_preds:
                fig3, ax3 = plt.subplots(figsize=(10, 5))
                names10 = [t["team_name"] for t in sorted_preds]
                odds10 = [float(t["prediction"]["championship_odds"].rstrip("%")) for t in sorted_preds]
                ax3.barh(range(len(names10)), odds10, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(names10))))
                ax3.set_yticks(range(len(names10)))
                ax3.set_yticklabels(names10, fontsize=9)
                ax3.set_xlabel("Championship odds (%)")
                ax3.set_title("Top 10 championship odds")
                ax3.grid(True, axis="x", linestyle="--", alpha=0.7)
                fig3.tight_layout()
                fig3.savefig(out / f"odds_top10{fig_suffix}.png", bbox_inches="tight")
                plt.close(fig3)

            # Title contender scatter: championship odds vs regular-season wins (win rate * games proxy)
            # East = circle (o), West = diamond (D); color coordinated (tab20)
            team_id_to_wins: dict[int, float] = team_id_to_win_rate
            n_games = 82.0
            fig4, ax4 = plt.subplots(figsize=(8, 6))
            east_preds_4 = [t for t in preds if team_id_to_conf.get(t["team_id"], "E") == "E"]
            west_preds_4 = [t for t in preds if team_id_to_conf.get(t["team_id"], "W") == "W"]
            cmap4 = plt.get_cmap("tab20")
            for i, t in enumerate(east_preds_4):
                ax4.scatter(team_id_to_wins.get(t["team_id"], 0.0) * n_games, float(t["prediction"]["championship_odds"].rstrip("%")), s=80, label=t["team_name"], alpha=0.8, c=[cmap4(i % 20)], marker="o", edgecolors="k", linewidths=0.5)
            for i, t in enumerate(west_preds_4):
                ax4.scatter(team_id_to_wins.get(t["team_id"], 0.0) * n_games, float(t["prediction"]["championship_odds"].rstrip("%")), s=80, label=t["team_name"], alpha=0.8, c=[cmap4((len(east_preds_4) + i) % 20)], marker="D", edgecolors="k", linewidths=0.5)
            ax4.set_xlabel("Regular-season wins (proxy from standings-to-date win rate × 82)")
            ax4.set_ylabel("Championship odds (%)")
            ax4.set_title("Title contender: odds vs wins (top-left = sleeper)")
            ax4.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
            ax4.grid(True, linestyle="--", alpha=0.7)
            fig4.tight_layout()
            fig4.savefig(out / f"title_contender_scatter{fig_suffix}.png", bbox_inches="tight")
            plt.close(fig4)

            # EOS playoff standings vs EOS_global_rank: playoff standings (x) vs playoff outcome (y)
            # East = circle (o), West = diamond (D); color coordinated (tab20)
            if eos_playoff_standings_map and eos_final_rank_map:
                fig5, ax5 = plt.subplots(figsize=(8, 6))
                pts = []
                for t in preds:
                    standings = t["analysis"].get("EOS_playoff_standings")
                    eos = t["analysis"].get("EOS_global_rank")
                    if standings is not None and eos is not None:
                        conf = team_id_to_conf.get(t["team_id"], "E")
                        pts.append((standings, eos, t["team_name"], conf))
                if pts:
                    east_pts5 = [(x, y, name) for (x, y, name, conf) in pts if conf == "E"]
                    west_pts5 = [(x, y, name) for (x, y, name, conf) in pts if conf == "W"]
                    all_x = [p[0] for p in pts]
                    all_y = [p[1] for p in pts]
                    max_r = max(max(all_x or [1]), max(all_y or [1]), 1) + 1
                    ax5.plot([0, max_r], [0, max_r], "k--", alpha=0.5, label="identity")
                    cmap = plt.get_cmap("tab20")
                    for i, (x, y, name) in enumerate(east_pts5):
                        ax5.scatter(x, y, c=[cmap(i % 20)], label=name, s=50, marker="o", edgecolors="k", linewidths=0.5)
                    for i, (x, y, name) in enumerate(west_pts5):
                        ax5.scatter(x, y, c=[cmap((len(east_pts5) + i) % 20)], label=name, s=50, marker="D", edgecolors="k", linewidths=0.5)
                    ax5.set_xlabel("EOS playoff standings (1-30)")
                    ax5.set_ylabel("EOS global rank (playoff outcome, 1-30)")
                    ax5.set_title("EOS playoff standings vs EOS global rank (identity = agreement)")
                    ax5.grid(True, linestyle="--", alpha=0.7)
                    ax5.legend(loc="best", fontsize=6, ncol=2)
                    ax5.set_xlim(-0.5, max_r)
                    ax5.set_ylim(-0.5, max_r)
                fig5.savefig(out / f"eos_playoff_standings_vs_eos_global_rank{fig_suffix}.png", bbox_inches="tight")
                plt.close(fig5)

        return pj

    last_pj = None
    for target_date, target_lists, output_file, season in test_specs:
        last_pj = _run_inference_for_spec(target_date, target_lists, output_file, season, draw_figures=True)
    if last_pj is not None and last_pj.name != "predictions.json":
        import shutil
        shutil.copy(last_pj, out / "predictions.json")
    for target_date, target_lists, output_file, _ in train_specs:
        _run_inference_for_spec(target_date, target_lists, output_file, None, draw_figures=False)

    return last_pj if last_pj is not None else out / "predictions.json"



def run_inference(output_dir: str | Path, config: dict, run_id: str | None = None) -> Path:
    """Run inference: from DB if present and has data, else exit with message (real run only)."""
    out = Path(output_dir)
    if run_id:
        out = out / run_id
    out.mkdir(parents=True, exist_ok=True)
    paths_cfg = config.get("paths", {})
    db_path = Path(paths_cfg.get("db", "data/processed/nba_build.duckdb"))
    if not db_path.is_absolute():
        from pathlib import Path as P
        root = P(__file__).resolve().parents[2]
        db_path = root / db_path
    if db_path.exists():
        try:
            return run_inference_from_db(output_dir, config, db_path, run_id=run_id)
        except Exception as e:
            raise RuntimeError(f"Inference from DB failed: {e}") from e
    raise FileNotFoundError(
        f"Database not found at {db_path}. Run scripts 1_download_raw and 2_build_db first."
    )
