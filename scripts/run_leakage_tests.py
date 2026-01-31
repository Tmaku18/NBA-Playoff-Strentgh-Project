"""Run leakage tests before training."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.leakage_tests import run_all

if __name__ == "__main__":
    run_all()
    print("Leakage tests passed.")
