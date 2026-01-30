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
    if not out.is_absolute():
        out = ROOT / out
    run_id = "run_001"
    try:
        p = run_inference(out, config, run_id=run_id)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {p}, pred_vs_actual.png, pred_vs_playoff_rank.png, odds_top10.png, title_contender_scatter.png")


if __name__ == "__main__":
    main()
