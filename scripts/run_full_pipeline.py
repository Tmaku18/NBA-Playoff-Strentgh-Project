"""Run the full pipeline: download -> build_db -> leakage tests -> train A/B -> stacking -> inference -> evaluate -> explain."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(script: str) -> int:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script)],
        cwd=str(ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
    ).returncode

def main() -> int:
    steps = [
        "1_download_raw.py",
        "2_build_db.py",
        "run_leakage_tests.py",
        "3_train_model_a.py",
        "4_train_models_b_and_c.py",
        "4b_train_stacking.py",
        "6_run_inference.py",
        "5_evaluate.py",
        "5b_explain.py",
    ]
    for i, script in enumerate(steps, 1):
        print(f"\n--- Step {i}/{len(steps)}: {script} ---")
        code = run(script)
        if code != 0:
            print(f"Pipeline failed at {script} (exit {code})")
            return code
    print("\n--- Pipeline complete ---")
    return 0

if __name__ == "__main__":
    sys.exit(main())
