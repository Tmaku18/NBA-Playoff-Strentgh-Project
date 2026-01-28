"""Run SHAP (Model B) and attention ablation (Model A) if deps available."""
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = Path(config["paths"]["outputs"])

    # SHAP: need a tree model
    try:
        import joblib
        from src.viz.shap_summary import shap_summary
        rf = joblib.load(out / "rf_model.joblib")
    except Exception:
        rf = None
    if rf is not None:
        X = np.random.randn(50, 5).astype(np.float32)
        try:
            exp = shap_summary(rf, X, feature_names=["eFG", "TOV_pct", "FT_rate", "ORB_pct", "pace"], out_path=out / "shap_summary.png")
            print("Wrote outputs/shap_summary.png")
        except Exception as e:
            print("SHAP skip:", e)

    # Attention ablation: need Model A
    try:
        from src.models.deep_set_rank import DeepSetRank
        from src.viz.integrated_gradients import attention_ablation
        ck = torch.load(out / "best_deep_set.pt", map_location="cpu", weights_only=False)
        ma = config.get("model_a", {})
        model = DeepSetRank(ma.get("num_embeddings", 500), ma.get("embedding_dim", 32), 7, ma.get("encoder_hidden", [128, 64]), ma.get("attention_heads", 4), ma.get("dropout", 0.2))
        if "model_state" in ck:
            model.load_state_dict(ck["model_state"])
        model.eval()
        emb = torch.randint(0, 100, (2, 15))
        stats = torch.randn(2, 15, 7) * 0.1
        minu = torch.rand(2, 15)
        msk = torch.zeros(2, 15, dtype=torch.bool)
        msk[:, 10:] = True
        with torch.no_grad():
            _, _, attn = model(emb, stats, minu, msk)
        v = attention_ablation(model, emb, stats, minu, msk, attn, top_k=2)
        print("Attention ablation (top-2 masked) score mean:", v)
    except Exception as e:
        print("Attention ablation skip:", e)


if __name__ == "__main__":
    main()
