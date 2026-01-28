"""Run evaluation on dummy data; write outputs/eval_report.json."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.evaluate import evaluate_ranking, evaluate_upset


def main():
    np.random.seed(42)
    n = 20
    y_true = np.random.rand(n)
    y_score = y_true + np.random.randn(n) * 0.2
    m = evaluate_ranking(y_true, y_score, k=10)
    y_bin = (np.random.rand(n) > 0.5).astype(float)
    m2 = evaluate_upset(y_bin, y_score)
    m.update(m2)
    out = Path(ROOT / "outputs" / "eval_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
