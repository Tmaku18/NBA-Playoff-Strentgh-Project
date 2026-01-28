"""Train Model B (XGB + RF) on dummy team-context features."""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_model_b import train_model_b


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    np.random.seed(42)
    n = 200
    feat = ["eFG", "TOV_pct", "FT_rate", "ORB_pct", "pace"]
    X = np.random.randn(n, len(feat)).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)
    Xv = np.random.randn(50, len(feat)).astype(np.float32)
    yv = np.random.rand(50).astype(np.float32)
    out = Path(config["paths"]["outputs"])
    p1, p2 = train_model_b(X, y, Xv, yv, config, feat, out)
    print(f"Saved {p1}, {p2}")


if __name__ == "__main__":
    main()
