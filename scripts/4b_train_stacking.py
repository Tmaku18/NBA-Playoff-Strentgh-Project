"""Train RidgeCV meta-learner on pooled OOF (dummy OOF)."""
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.train_stacking import train_stacking


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    np.random.seed(42)
    n = 100
    oof_a = np.random.randn(n).astype(np.float32)
    oof_x = np.random.randn(n).astype(np.float32)
    oof_r = np.random.randn(n).astype(np.float32)
    y = (0.3 * oof_a + 0.4 * oof_x + 0.3 * oof_r + np.random.randn(n) * 0.1).astype(np.float32)
    out = Path(config["paths"]["outputs"])
    path = train_stacking(oof_a, oof_x, oof_r, y, config, out)
    print(f"Saved {path}, outputs/oof_pooled.parquet")


if __name__ == "__main__":
    main()
