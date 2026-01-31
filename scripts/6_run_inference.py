"""Run inference pipeline: predictions.json and figures."""
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference.predict import run_inference


def _next_run_id(outputs_dir: Path) -> str:
    """Auto-increment: find existing run_NNN dirs, return run_{max+1:03d}. Starts at run_002 if run_001 exists."""
    outputs_dir = Path(outputs_dir)
    if not outputs_dir.exists():
        return "run_001"
    pattern = re.compile(r"^run_(\d+)$", re.I)
    numbers = []
    for p in outputs_dir.iterdir():
        if p.is_dir() and pattern.match(p.name):
            numbers.append(int(pattern.match(p.name).group(1)))
    next_n = max(numbers, default=0) + 1
    return f"run_{next_n:03d}"


def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    out = Path(config["paths"]["outputs"])
    if not out.is_absolute():
        out = ROOT / out
    run_id = config.get("inference", {}).get("run_id")
    if run_id is None or (isinstance(run_id, str) and run_id.strip().lower() in ("null", "")):
        run_id = _next_run_id(out)
    else:
        run_id = str(run_id).strip()
    try:
        p = run_inference(out, config, run_id=run_id)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    print(f"Wrote {out / run_id} (run_id={run_id})")


if __name__ == "__main__":
    main()
