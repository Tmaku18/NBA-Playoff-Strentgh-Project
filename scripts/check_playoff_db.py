"""Check if canonical DB exists and has playoff data. Run: python -m scripts.check_playoff_db"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

with open(ROOT / "config" / "defaults.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

db_path = Path(config["paths"]["db"])
if not db_path.is_absolute():
    db_path = ROOT / db_path

# NBA_DB_PATH override (same as inference)
import os
db_override = os.environ.get("NBA_DB_PATH")
if db_override and str(db_override).strip():
    db_path = Path(db_override.strip())

print(f"Canonical DB path: {db_path}")
print(f"DB exists: {db_path.exists()}")

if not db_path.exists():
    print("DB NOT FOUND. Run scripts 1_download_raw and 2_build_db first.")
    sys.exit(1)

import duckdb
con = duckdb.connect(str(db_path), read_only=True)
try:
    tables = [r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
    ).fetchall()]
    print(f"Tables: {tables}")

    for t in ["playoff_games", "playoff_team_game_logs"]:
        if t in tables:
            n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            print(f"  {t}: {n} rows")
            if n > 0 and t == "playoff_games":
                seasons = con.execute(
                    "SELECT season, COUNT(*) FROM playoff_games GROUP BY season ORDER BY season"
                ).fetchall()
                print(f"    Seasons: {seasons[:15]}")
        else:
            print(f"  {t}: MISSING")
finally:
    con.close()
