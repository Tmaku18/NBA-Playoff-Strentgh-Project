"""DuckDB schema: teams, players, games, player_game_logs, team_game_logs."""

SCHEMA_SQL = """
-- teams: from distinct team_game_logs / team info
CREATE TABLE IF NOT EXISTS teams (
  team_id BIGINT PRIMARY KEY,
  abbreviation VARCHAR(10),
  name VARCHAR(128),
  conference VARCHAR(8)
);

-- players: from distinct player_game_logs
CREATE TABLE IF NOT EXISTS players (
  player_id BIGINT PRIMARY KEY,
  player_name VARCHAR(128)
);

-- games: one row per game. home/away from MATCHUP (@ = away, vs = home)
CREATE TABLE IF NOT EXISTS games (
  game_id VARCHAR(32) PRIMARY KEY,
  game_date DATE,
  season VARCHAR(16),
  home_team_id BIGINT,
  away_team_id BIGINT,
  FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
  FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);

-- team_game_logs: one row per team per game (from LeagueGameLog PlayerOrTeam='T')
CREATE TABLE IF NOT EXISTS team_game_logs (
  game_id VARCHAR(32),
  team_id BIGINT,
  is_home INTEGER,
  wl VARCHAR(4),
  min INTEGER,
  pts INTEGER,
  oreb INTEGER,
  dreb INTEGER,
  reb INTEGER,
  ast INTEGER,
  stl INTEGER,
  blk INTEGER,
  tov INTEGER,
  fgm INTEGER,
  fga INTEGER,
  fg3m INTEGER,
  fg3a INTEGER,
  ftm INTEGER,
  fta INTEGER,
  plus_minus INTEGER,
  PRIMARY KEY (game_id, team_id),
  FOREIGN KEY (game_id) REFERENCES games(game_id),
  FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

-- player_game_logs: one row per player per game (from LeagueGameLog PlayerOrTeam='P')
CREATE TABLE IF NOT EXISTS player_game_logs (
  game_id VARCHAR(32),
  player_id BIGINT,
  team_id BIGINT,
  min INTEGER,
  pts INTEGER,
  reb INTEGER,
  ast INTEGER,
  stl INTEGER,
  blk INTEGER,
  tov INTEGER,
  fgm INTEGER,
  fga INTEGER,
  fg3m INTEGER,
  fg3a INTEGER,
  ftm INTEGER,
  fta INTEGER,
  oreb INTEGER,
  dreb INTEGER,
  plus_minus INTEGER,
  PRIMARY KEY (game_id, player_id),
  FOREIGN KEY (game_id) REFERENCES games(game_id),
  FOREIGN KEY (player_id) REFERENCES players(player_id),
  FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games(season);
CREATE INDEX IF NOT EXISTS idx_pgl_game ON player_game_logs(game_id);
CREATE INDEX IF NOT EXISTS idx_pgl_player ON player_game_logs(player_id);
CREATE INDEX IF NOT EXISTS idx_pgl_team ON player_game_logs(team_id);
CREATE INDEX IF NOT EXISTS idx_tgl_game ON team_game_logs(game_id);
CREATE INDEX IF NOT EXISTS idx_tgl_team ON team_game_logs(team_id);

-- Playoff tables (separate from regular season; Play-In excluded when computing wins)
CREATE TABLE IF NOT EXISTS playoff_games (
  game_id VARCHAR(32) PRIMARY KEY,
  game_date DATE,
  season VARCHAR(16),
  home_team_id BIGINT,
  away_team_id BIGINT,
  FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
  FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);
CREATE TABLE IF NOT EXISTS playoff_team_game_logs (
  game_id VARCHAR(32),
  team_id BIGINT,
  is_home INTEGER,
  wl VARCHAR(4),
  min INTEGER,
  pts INTEGER,
  oreb INTEGER,
  dreb INTEGER,
  reb INTEGER,
  ast INTEGER,
  stl INTEGER,
  blk INTEGER,
  tov INTEGER,
  fgm INTEGER,
  fga INTEGER,
  fg3m INTEGER,
  fg3a INTEGER,
  ftm INTEGER,
  fta INTEGER,
  plus_minus INTEGER,
  PRIMARY KEY (game_id, team_id),
  FOREIGN KEY (game_id) REFERENCES playoff_games(game_id),
  FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
CREATE TABLE IF NOT EXISTS playoff_player_game_logs (
  game_id VARCHAR(32),
  player_id BIGINT,
  team_id BIGINT,
  min INTEGER,
  pts INTEGER,
  reb INTEGER,
  ast INTEGER,
  stl INTEGER,
  blk INTEGER,
  tov INTEGER,
  fgm INTEGER,
  fga INTEGER,
  fg3m INTEGER,
  fg3a INTEGER,
  ftm INTEGER,
  fta INTEGER,
  oreb INTEGER,
  dreb INTEGER,
  plus_minus INTEGER,
  PRIMARY KEY (game_id, player_id),
  FOREIGN KEY (game_id) REFERENCES playoff_games(game_id),
  FOREIGN KEY (player_id) REFERENCES players(player_id),
  FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
CREATE INDEX IF NOT EXISTS idx_playoff_games_date ON playoff_games(game_date);
CREATE INDEX IF NOT EXISTS idx_playoff_games_season ON playoff_games(season);
CREATE INDEX IF NOT EXISTS idx_playoff_tgl_game ON playoff_team_game_logs(game_id);
CREATE INDEX IF NOT EXISTS idx_playoff_tgl_team ON playoff_team_game_logs(team_id);
CREATE INDEX IF NOT EXISTS idx_playoff_pgl_game ON playoff_player_game_logs(game_id);
CREATE INDEX IF NOT EXISTS idx_playoff_pgl_team ON playoff_player_game_logs(team_id);
"""


def create_schema(con) -> None:
    for raw in SCHEMA_SQL.strip().split(";"):
        s = "\n".join(
            line for line in raw.strip().split("\n")
            if line.strip() and not line.strip().startswith("--")
        ).strip()
        if s:
            con.execute(s)
