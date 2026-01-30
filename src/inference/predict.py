"""Inference: load A/B/stacker, produce per-team JSON (predicted_rank, true_strength, delta, ensemble, contributors)."""

from __future__ import annotations

import json
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
        model_a = DeepSetRank(
            ma.get("num_embeddings", 500),
            ma.get("embedding_dim", 32),
            7,
            ma.get("encoder_hidden", [128, 64]),
            ma.get("attention_heads", 4),
            ma.get("dropout", 0.2),
        )
        if "model_state" in ck:
            model_a.load_state_dict(ck["model_state"])
        model_a.eval()

    if xgb_path and Path(xgb_path).exists():
        import joblib
        xgb = joblib.load(xgb_path)
    if rf_path and Path(rf_path).exists():
        import joblib
        rf = joblib.load(rf_path)
    if meta_path and Path(meta_path).exists():
        import joblib
        meta = joblib.load(meta_path)

    return model_a, xgb, rf, meta


def predict_teams(
    team_ids: list[int],
    team_names: list[str],
    model_a_scores: np.ndarray | None = None,
    xgb_scores: np.ndarray | None = None,
    rf_scores: np.ndarray | None = None,
    meta_model: Any = None,
    actual_ranks: dict[int, int] | None = None,
    attention_by_team: dict[int, list[tuple[str, float]]] | None = None,
    team_id_to_conference: dict[int, str] | None = None,
    playoff_rank: dict[int, int] | None = None,
    *,
    true_strength_scale: str = "percentile",
    odds_temperature: float = 1.0,
) -> list[dict]:
    """
    Combine base scores, run meta if present. For each team output:
    global_rank (1-30), conference_rank (1-15), predicted_rank (legacy), true_strength_score,
    championship_odds, delta, classification, analysis.playoff_rank and rank_delta_playoffs when available.
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
    if meta_model is not None:
        ens = meta_model.predict(X).ravel()
    else:
        ens = (sa + sx + sr) / 3.0

    pred_rank = np.argsort(np.argsort(-ens)) + 1  # global rank 1-30
    if true_strength_scale == "percentile":
        tss = (np.argsort(np.argsort(ens)) + 1).astype(float) / (n + 1)
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
    attention_by_team = attention_by_team or {}
    playoff_rank = playoff_rank or {}

    out = []
    for i, (tid, tname) in enumerate(zip(team_ids, team_names)):
        act = actual_ranks.get(tid)
        delta = (act - pred_rank[i]) if act is not None else None
        if delta is not None:
            if delta > 0:
                classification = f"Sleeper (Under-ranked by {delta} slots)"
            elif delta < 0:
                classification = f"Paper Tiger (Over-ranked by {-delta} slots)"
            else:
                classification = "Aligned"
        else:
            classification = "Unknown"

        # deep_set_rank: global rank (1-30) by Model A score. Note: Model A was trained with ListMLE
        # on standings-ordered lists; actual_rank is that same list position. So deep_set_rank often
        # matches actual_rank within conference—that is by design (same target), not independent accuracy.
        r_a = np.argsort(np.argsort(-sa))[i] + 1 if len(sa) == n else None
        r_x = np.argsort(np.argsort(-sx))[i] + 1 if len(sx) == n else None
        r_r = np.argsort(np.argsort(-sr))[i] + 1 if len(sr) == n else None
        spread = max(r or 0 for r in [r_a, r_x, r_r]) - min(r or 0 for r in [r_a, r_x, r_r]) if any([r_a, r_x, r_r]) else 0
        agreement = "High" if spread <= 2 else "Low"

        contrib = attention_by_team.get(tid, [])
        p_rank = playoff_rank.get(tid)
        rank_delta_playoffs = (p_rank - pred_rank[i]) if p_rank is not None else None

        pred_dict: dict[str, Any] = {
            "predicted_rank": int(pred_rank[i]),
            "global_rank": int(pred_rank[i]),
            "true_strength_score": float(tss[i]),
            "true_strength_score_100": round(float(tss[i]) * 100.0, 1),
            "conference_rank": conf_rank.get(tid),
            "championship_odds": f"{float(odds[i]) * 100:.1f}%",
        }
        analysis_dict: dict[str, Any] = {
            "actual_rank": int(act) if act is not None else None,
            "classification": classification,
            "playoff_rank": int(p_rank) if p_rank is not None else None,
            "rank_delta_playoffs": int(rank_delta_playoffs) if rank_delta_playoffs is not None else None,
        }

        out.append({
            "team_id": int(tid),
            "team_name": tname,
            "prediction": pred_dict,
            "analysis": analysis_dict,
            "ensemble_diagnostics": {"model_agreement": agreement, "deep_set_rank": int(r_a) if r_a is not None else None, "xgboost_rank": int(r_x) if r_x is not None else None, "random_forest_rank": int(r_r) if r_r is not None else None},
            "roster_dependence": {"primary_contributors": [{"player": str(p), "attention_weight": float(w)} for p, w in contrib]},
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
    from src.features.team_context import TEAM_CONTEXT_FEATURE_COLS, build_team_context_as_of_dates
    from src.training.build_lists import TEAM_CONFERENCE, build_lists
    from src.training.data_model_a import build_batches_from_lists
    from src.training.train_model_a import predict_batches_with_attention

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
    # Target: latest date in DB (or last list date)
    dates_sorted = sorted(set(lst["as_of_date"] for lst in lists))
    target_date = dates_sorted[-1] if dates_sorted else None
    target_lists = [lst for lst in lists if lst["as_of_date"] == target_date]
    if not target_lists:
        target_lists = lists[-2:] if len(lists) >= 2 else lists
    # Flatten to one list of (team_id, as_of_date) across target lists; keep unique team_id for naming/rank
    team_id_to_as_of: dict[int, str] = {}
    team_id_to_actual_rank: dict[int, int] = {}
    team_id_to_win_rate: dict[int, float] = {}
    for lst in target_lists:
        for r, tid in enumerate(lst["team_ids"], start=1):
            tid = int(tid)
            team_id_to_as_of[tid] = lst["as_of_date"]
            team_id_to_actual_rank[tid] = r
            team_id_to_win_rate[tid] = lst["win_rates"][lst["team_ids"].index(tid)] if tid in lst["team_ids"] else 0.0
    unique_team_ids = list(dict.fromkeys(tid for lst in target_lists for tid in lst["team_ids"]))
    unique_team_ids = [int(t) for t in unique_team_ids]
    if not unique_team_ids:
        raise ValueError("No teams in target lists.")
    team_dates = [(tid, team_id_to_as_of.get(tid, target_date or "")) for tid in unique_team_ids]
    as_of_date = target_date or team_dates[0][1]

    device = torch.device("cpu")  # Match load_models map_location="cpu"
    tid_to_score_a: dict[int, float] = {}
    attention_by_team: dict[int, list[tuple[str, float]]] = {}  # team_id -> [(player_name, weight), ...]
    if model_a is not None:
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
                    attn_weights = attn_tensor[0, k].numpy() if attn_tensor.dim() >= 2 else attn_tensor[k].numpy()
                    order = np.argsort(-attn_weights)
                    contrib: list[tuple[str, float]] = []
                    for idx in order[:10]:
                        if idx >= len(player_ids) or player_ids[idx] is None:
                            continue
                        w = float(attn_weights[idx])
                        if w <= 0:
                            continue
                        pid = player_ids[idx]
                        name = player_id_to_name.get(int(pid), f"Player_{pid}")
                        contrib.append((name, w))
                    if contrib:
                        attention_by_team[tid] = contrib
    sa = np.array([tid_to_score_a.get(tid, 0.0) for tid in unique_team_ids], dtype=np.float32)

    sx = np.zeros(len(unique_team_ids), dtype=np.float32)
    sr = np.zeros(len(unique_team_ids), dtype=np.float32)
    feat_df = build_team_context_as_of_dates(tgl, games, team_dates)
    if not feat_df.empty and xgb is not None and rf is not None:
        feat_cols = [c for c in TEAM_CONTEXT_FEATURE_COLS if c in feat_df.columns]
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

    # Playoff rank for target season (when available)
    playoff_rank_map: dict[int, int] = {}
    seasons_cfg = config.get("seasons") or {}
    target_season = None
    for season, rng in seasons_cfg.items():
        start = pd.to_datetime(rng.get("start")).date()
        end = pd.to_datetime(rng.get("end")).date()
        if start <= pd.to_datetime(as_of_date).date() <= end:
            target_season = season
            break
    if target_season:
        try:
            from src.data.db_loader import load_playoff_data
            from src.evaluation.playoffs import compute_playoff_performance_rank
            pg, ptgl, _ = load_playoff_data(db_path)
            if not pg.empty and not ptgl.empty:
                playoff_rank_map = compute_playoff_performance_rank(
                    pg, ptgl, games, tgl, target_season,
                    all_team_ids=unique_team_ids,
                )
        except Exception:
            pass

    preds = predict_teams(
        unique_team_ids,
        team_names,
        model_a_scores=sa,
        xgb_scores=sx,
        rf_scores=sr,
        meta_model=meta,
        actual_ranks=actual_ranks,
        attention_by_team=attention_by_team if attention_by_team else None,
        team_id_to_conference=team_id_to_conf,
        playoff_rank=playoff_rank_map if playoff_rank_map else None,
        true_strength_scale=config.get("output", {}).get("true_strength_scale", "percentile"),
        odds_temperature=float(config.get("output", {}).get("odds_temperature", 1.0)),
    )
    pj = out / "predictions.json"
    with open(pj, "w", encoding="utf-8") as f:
        json.dump({"teams": preds}, f, indent=2)

    east_preds = [t for t in preds if team_id_to_conf.get(t["team_id"], "E") == "E"]
    west_preds = [t for t in preds if team_id_to_conf.get(t["team_id"], "W") == "W"]

    fig, (ax_east, ax_west) = plt.subplots(1, 2, figsize=(14, 6))

    def _draw_panel(ax, pred_list, title):
        if not pred_list:
            ax.text(0.5, 0.5, f"No {title} teams", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel("Actual conference rank")
            ax.set_ylabel("Predicted conference rank")
            ax.grid(True, linestyle="--", alpha=0.7)
            return
        pr = [t["prediction"].get("conference_rank") or t["prediction"]["predicted_rank"] for t in pred_list]
        ar = [t["analysis"]["actual_rank"] for t in pred_list]
        pr = [p if p is not None else 0 for p in pr]
        ar = [a if a is not None else 0 for a in ar]
        names = [t["team_name"] for t in pred_list]
        max_r = max(max(ar or [1]), max(pr or [1]), 1) + 1
        ax.plot([0, max_r], [0, max_r], "k--", alpha=0.5, label="identity")
        cmap = plt.get_cmap("tab20")
        for i, (a, p) in enumerate(zip(ar, pr)):
            color = cmap(i % 20)
            ax.scatter(a, p, c=[color], label=names[i], s=60, edgecolors="k", linewidths=0.5)
        ax.set_xlabel("Actual conference rank")
        ax.set_ylabel("Predicted conference rank")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.legend(loc="best", fontsize=7, ncol=2)
        ax.set_xlim(-0.5, max_r)
        ax.set_ylim(-0.5, max_r)

    _draw_panel(ax_east, east_preds, "East")
    _draw_panel(ax_west, west_preds, "West")
    fig.suptitle("Predicted vs actual rank (conference rank 1-15)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "pred_vs_actual.png", bbox_inches="tight")
    plt.close()

    # pred_vs_playoff_rank: global rank (1-30) vs playoff performance rank (1-30)
    if playoff_rank_map:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        g_rank = [t["prediction"]["global_rank"] for t in preds]
        p_rank = [t["analysis"].get("playoff_rank") for t in preds]
        p_rank = [r if r is not None else 0 for r in p_rank]
        names = [t["team_name"] for t in preds]
        max_r = max(max(g_rank or [1]), max(p_rank or [1]), 1) + 1
        ax2.plot([0, max_r], [0, max_r], "k--", alpha=0.5, label="identity")
        cmap = plt.get_cmap("tab20")
        for i, (g, p) in enumerate(zip(g_rank, p_rank)):
            ax2.scatter(p, g, c=[cmap(i % 20)], label=names[i], s=50, edgecolors="k", linewidths=0.5)
        ax2.set_xlabel("Playoff performance rank (1-30)")
        ax2.set_ylabel("Predicted global rank (1-30)")
        ax2.set_title("Predicted global rank vs playoff performance rank")
        ax2.grid(True, linestyle="--", alpha=0.7)
        ax2.legend(loc="best", fontsize=6, ncol=2)
        ax2.set_xlim(-0.5, max_r)
        ax2.set_ylim(-0.5, max_r)
        fig2.savefig(out / "pred_vs_playoff_rank.png", bbox_inches="tight")
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
        fig3.savefig(out / "odds_top10.png", bbox_inches="tight")
        plt.close(fig3)

    # Title contender scatter: championship odds vs regular-season wins (win rate * games proxy)
    team_id_to_wins: dict[int, float] = team_id_to_win_rate
    n_games = 82.0
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    odds_pct = [float(t["prediction"]["championship_odds"].rstrip("%")) for t in preds]
    wins_proxy = [team_id_to_wins.get(t["team_id"], 0.0) * n_games for t in preds]
    names = [t["team_name"] for t in preds]
    for i in range(len(preds)):
        ax4.scatter(wins_proxy[i], odds_pct[i], s=80, label=names[i], alpha=0.8)
    ax4.set_xlabel("Regular-season wins (proxy from standings-to-date win rate × 82)")
    ax4.set_ylabel("Championship odds (%)")
    ax4.set_title("Title contender: odds vs wins (top-left = sleeper)")
    ax4.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    ax4.grid(True, linestyle="--", alpha=0.7)
    fig4.tight_layout()
    fig4.savefig(out / "title_contender_scatter.png", bbox_inches="tight")
    plt.close(fig4)

    return pj


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
