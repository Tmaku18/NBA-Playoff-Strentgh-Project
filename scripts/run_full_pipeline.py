"""Run the full pipeline: download -> build_db -> train A -> train B -> stacking -> evaluate -> explain -> inference."""
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
        "3_train_model_a.py",
        "4_train_model_b.py",
        "4b_train_stacking.py",
        "5_evaluate.py",
        "5b_explain.py",
        "6_run_inference.py",
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
