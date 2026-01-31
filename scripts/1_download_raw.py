"""Download player and team game logs via nba_api; write data/manifest.json (raw hashes, timestamps)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

# project root
ROOT = Path(__file__).resolve().parents[1]

def main():
    with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    raw_dir = Path(cfg["paths"]["raw"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(cfg.get("seasons", {}).keys())

    from src.data.nba_api_client import fetch_season_logs

    manifest = {"raw": {}, "timestamps": {}}
    for season in seasons:
        y1, y2 = season.split("-")[0], season.split("-")[1]
        for kind, ext in [("T", "parquet"), ("P", "parquet")]:
            try:
                df = fetch_season_logs(season, raw_dir, kind=kind, use_cache=True, cache_fmt=ext)
                stem = "team_logs" if kind == "T" else "player_logs"
                path = raw_dir / f"{stem}_{y1}_{y2}.{ext}"
                if path.exists():
                    h = hashlib.sha256(path.read_bytes()).hexdigest()
                    manifest["raw"][path.name] = h
            except Exception as e:
                print(f"Skip {season} {kind}: {e}")
    # Playoff logs (Playoffs only; Play-In excluded when computing playoff wins)
    for season in seasons:
        y1, y2 = season.split("-")[0], season.split("-")[1]
        for kind, ext in [("T", "parquet"), ("P", "parquet")]:
            try:
                df = fetch_season_logs(
                    season, raw_dir, kind=kind, use_cache=True, cache_fmt=ext,
                    season_type="Playoffs",
                )
                stem = "playoffs_team_logs" if kind == "T" else "playoffs_player_logs"
                path = raw_dir / f"{stem}_{y1}_{y2}.{ext}"
                if path.exists():
                    h = hashlib.sha256(path.read_bytes()).hexdigest()
                    manifest["raw"][path.name] = h
            except Exception as e:
                print(f"Skip playoffs {season} {kind}: {e}")

    manifest["timestamps"] = {"download": str(Path(__file__).stat().st_mtime)}
    out = ROOT / "data" / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
