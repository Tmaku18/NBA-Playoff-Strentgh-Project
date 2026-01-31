"""Write run manifest: config snapshot, git hash, data manifest hash. No external services."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# project root (parent of scripts)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml


def _git_hash() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.stdout.strip() or None if r.returncode == 0 else None
    except Exception:
        return None


def _data_manifest_hash() -> str | None:
    p = ROOT / "data" / "manifest.json"
    if not p.exists():
        return None
    try:
        b = p.read_bytes()
        return hashlib.sha256(b).hexdigest()
    except Exception:
        return None


def _config_snapshot() -> dict:
    p = ROOT / "config" / "defaults.yaml"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config_snapshot": _config_snapshot(),
        "git_hash": _git_hash(),
        "data_manifest_hash": _data_manifest_hash(),
    }

    path = out_dir / "run_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Ensure we can import the project (repro, time_indexing)
    from src.utils import set_seeds, filter_as_of  # noqa: F401

    set_seeds(42)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
