"""Run inference pipeline: predictions.json and figures."""
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predict import run_inference


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = Path(config["paths"]["outputs"])
    run_id = "run_001"
    p = run_inference(out, config, run_id=run_id)
    print(f"Wrote {p} and outputs/{run_id}/pred_vs_actual.png")


if __name__ == "__main__":
    main()
