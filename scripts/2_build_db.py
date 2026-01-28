"""Build DuckDB from raw logs; update data/manifest.json (processed, raw hashes)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    import sys
    sys.path.insert(0, str(ROOT))

    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["paths"]["raw"])
    db_path = Path(cfg["paths"]["db"])
    seasons = list(cfg.get("seasons", {}).keys())

    from src.data.db_loader import load_raw_into_db

    load_raw_into_db(raw_dir, db_path, seasons=seasons)

    manifest_path = ROOT / "data" / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    if db_path.exists():
        manifest["processed"] = hashlib.sha256(db_path.read_bytes()).hexdigest()
    # preserve raw hashes if present; if 1_download_raw wasn't run, hash raw files that exist
    if "raw" not in manifest or not manifest["raw"]:
        manifest["raw"] = {}
        for p in (raw_dir).glob("*.parquet"):
            manifest["raw"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (raw_dir).glob("*.csv"):
            manifest["raw"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest["db_path"] = str(db_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Built {db_path}, updated {manifest_path}")


if __name__ == "__main__":
    main()
