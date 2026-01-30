"""Run SHAP (Model B) and attention ablation (Model A) on real data from DB."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    db_path = ROOT / config["paths"]["db"]
    if not db_path.exists():
        print("Database not found. Run scripts 1_download_raw and 2_build_db first.", file=sys.stderr)
        sys.exit(1)

    from src.data.db_loader import load_training_data
    from src.features.team_context import TEAM_CONTEXT_FEATURE_COLS, build_team_context_as_of_dates
    from src.training.build_lists import build_lists
    from src.training.data_model_a import build_batches_from_lists

    games, tgl, teams, pgl = load_training_data(db_path)
    if games.empty or tgl.empty:
        print("DB has no games/tgl. Run 2_build_db with raw data first.", file=sys.stderr)
        sys.exit(1)

    lists = build_lists(tgl, games, teams)
    if not lists:
        print("No lists from build_lists.", file=sys.stderr)
        sys.exit(1)
    rows = []
    for lst in lists:
        for tid, _ in zip(lst["team_ids"], lst["win_rates"]):
            rows.append({"team_id": int(tid), "as_of_date": lst["as_of_date"]})
    flat = pd.DataFrame(rows)
    team_dates = [(int(a), str(b)) for a, b in flat[["team_id", "as_of_date"]].drop_duplicates().values.tolist()]
    feat_df = build_team_context_as_of_dates(tgl, games, team_dates)
    feat_cols = [c for c in TEAM_CONTEXT_FEATURE_COLS if c in feat_df.columns]
    if not feat_cols:
        print("No feature columns for SHAP.", file=sys.stderr)
        sys.exit(1)
    X_real = feat_df[feat_cols].fillna(0.0).values.astype(np.float32)
    if X_real.shape[0] > 500:
        X_real = X_real[:500]

    # SHAP on real team-context X
    rf_path = out / "rf_model.joblib"
    if not rf_path.exists():
        print("RF model not found. Run script 4 first.", file=sys.stderr)
        sys.exit(1)
    try:
        import joblib
        from src.viz.shap_summary import shap_summary
        rf = joblib.load(rf_path)
        shap_summary(rf, X_real, feature_names=feat_cols, out_path=out / "shap_summary.png")
        print("Wrote", out / "shap_summary.png")
    except Exception as e:
        print("SHAP failed:", e, file=sys.stderr)
        sys.exit(1)

    # Attention ablation on real list batch
    model_a_path = out / "best_deep_set.pt"
    if not model_a_path.exists():
        print("Model A not found. Run script 3 first.", file=sys.stderr)
        sys.exit(1)
    try:
        from src.models.deep_set_rank import DeepSetRank
        from src.viz.integrated_gradients import attention_ablation
        valid_lists = [lst for lst in lists if len(lst["team_ids"]) >= 2]
        if not valid_lists:
            print("No valid lists for attention ablation.", file=sys.stderr)
        else:
            device = torch.device("cpu")
            batches_a, _ = build_batches_from_lists(valid_lists[:1], games, tgl, teams, pgl, config, device=device)
            if not batches_a:
                print("No batches for attention ablation.", file=sys.stderr)
            else:
                ck = torch.load(model_a_path, map_location="cpu", weights_only=False)
                ma = config.get("model_a", {})
                model = DeepSetRank(
                    ma.get("num_embeddings", 500), ma.get("embedding_dim", 32), 7,
                    ma.get("encoder_hidden", [128, 64]), ma.get("attention_heads", 4), ma.get("dropout", 0.2),
                )
                if "model_state" in ck:
                    model.load_state_dict(ck["model_state"])
                model.eval()
                batch = batches_a[0]
                emb = batch["embedding_indices"].to(device).reshape(-1, batch["embedding_indices"].shape[2])
                stats = batch["player_stats"].to(device).reshape(-1, batch["player_stats"].shape[2], batch["player_stats"].shape[3])
                minu = batch["minutes"].to(device).reshape(-1, batch["minutes"].shape[2])
                msk = batch["key_padding_mask"].to(device).reshape(-1, batch["key_padding_mask"].shape[2])
                with torch.no_grad():
                    _, _, attn = model(emb, stats, minu, msk)
                top_k = min(2, attn.shape[-1])
                v = attention_ablation(model, emb, stats, minu, msk, attn, top_k=top_k)
                if not math.isfinite(v):
                    print("Attention ablation (top-%d masked) score mean: NaN (masked forward produced non-finite scores)" % top_k)
                else:
                    print("Attention ablation (top-%d masked) score mean: %s" % (top_k, v))

                # Integrated Gradients for Model A (one team, optional)
                try:
                    from src.viz.integrated_gradients import ig_attr, _HAS_CAPTUM
                    if not _HAS_CAPTUM:
                        print("Integrated Gradients skipped (captum not installed).")
                    else:
                        n_steps = 50
                        emb_1 = emb[0:1]
                        stats_1 = stats[0:1]
                        minu_1 = minu[0:1]
                        msk_1 = msk[0:1]
                        attr, delta = ig_attr(model, emb_1, stats_1, minu_1, msk_1, n_steps=n_steps)
                        if attr is not None and attr.numel() > 0:
                            # attr (1, P, S); L2 norm per player
                            norms = torch.norm(attr[0].float(), dim=1)
                            k = min(5, norms.shape[0])
                            _, top_idx = norms.topk(k, largest=True)
                            lines = ["Integrated Gradients (Model A) top-%d player indices by attribution L2 norm:" % k]
                            for i, idx in enumerate(top_idx.tolist(), 1):
                                lines.append("  %d. player_idx=%d norm=%.4f" % (i, idx, norms[idx].item()))
                            summary = "\n".join(lines)
                            print(summary)
                            ig_path = out / "ig_model_a_attributions.txt"
                            ig_path.write_text(summary, encoding="utf-8")
                            print("Wrote", ig_path)
                        else:
                            print("Integrated Gradients: no attributions (empty result).")
                except ImportError:
                    print("Integrated Gradients skipped (captum not installed).")
                except Exception as e:
                    print("Integrated Gradients failed:", e, file=sys.stderr)
    except Exception as e:
        print("Attention ablation failed:", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
