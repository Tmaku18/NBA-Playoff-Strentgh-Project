This plan details the development and implementation of the Multi-Modal Stacking Ensemble system for predicting NBA "True Team Strength" as specified in Plan.md.
Phase 1: Project Setup and Environment Configuration
1.1 Directory Structure
Create the following project layout:
nba-true-strength/
├── config/
│   └── defaults.yaml          # Hyperparams, paths, seeds
├── src/
│   ├── data/                   # Data acquisition & processing
│   │   ├── __init__.py
│   │   ├── nba_api_client.py   # nba_api wrapper
│   │   ├── kaggle_loader.py    # Kaggle SOS/SRS data
│   │   ├── db.py               # DuckDB connection manager
│   │   └── feature_eng.py      # Rolling windows, Four Factors
│   ├── models/
│   │   ├── __init__.py
│   │   ├── deep_set.py         # Model A: Deep Set network
│   │   ├── ensemble.py         # Model B: XGBoost + RF
│   │   ├── meta_learner.py     # Level-2 stacking
│   │   └── losses.py           # ListMLE implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Training loop
│   │   └── validation.py       # Walk-forward CV
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # NDCG, Spearman, MRR, Brier
│   │   └── baselines.py        # SRS/NR ranking baselines
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_model_b.py     # SHAP for ensemble
│   │   └── captum_model_a.py   # Integrated Gradients
│   └── viz/
│       ├── __init__.py
│       └── plots.py            # All visualizations
├── outputs/
│   ├── oof_deep_set.parquet    # Persisted OOF predictions
│   ├── oof_xgb.parquet
│   ├── oof_rf.parquet
│   └── models/                 # Checkpoints
├── data/
│   ├── raw/                    # Original data files
│   ├── processed/              # DuckDB database
│   └── hashes.json             # Data versioning
├── notebooks/                   # Exploration
├── tests/
├── requirements.txt
└── README.md

1.2 Dependencies (requirements.txt)
torch>=2.0.0
xgboost>=2.0.0
scikit-learn>=1.3.0
duckdb>=0.9.0
nba_api>=1.4.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
shap>=0.43.0
captum>=0.6.0
scipy>=1.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
tqdm>=4.65.0

1.3 Configuration File (config/defaults.yaml)
data:
  seasons: ["2015-16", "2016-17", ..., "2025-26"]
  db_path: "data/processed/nba.duckdb"
  
model_a:
  embedding_dim: 32
  encoder_hidden: [64, 32]
  attention_heads: 4
  dropout: 0.2
  
model_b:
  xgb:
    max_depth: 6
    learning_rate: 0.05
    n_estimators: 500
    early_stopping_rounds: 50
  rf:
    n_estimators: 300
    max_depth: 10
    
training:
  batch_size: 32
  lr: 0.001
  epochs: 100
  
validation:
  n_folds: 5
  
output:
  true_strength_scale: "percentile"  # or "softmax", "platt"
  
repro:
  seed: 42

  1.4 Reproducibility Setup
In src/__init__.py:
import torch
import numpy as np
import random

def set_seeds(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    Phase 2: Data Acquisition and Storage
2.1 DuckDB Schema Design
Create tables optimized for analytical queries with proper indexing:

-- Teams reference table
CREATE TABLE teams (
    team_id INTEGER PRIMARY KEY,
    abbreviation VARCHAR(3),
    name VARCHAR(50),
    conference VARCHAR(4)
);

-- Players reference table
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY,
    name VARCHAR(100)
);

-- Game-level data
CREATE TABLE games (
    game_id VARCHAR(10) PRIMARY KEY,
    game_date DATE,
    season VARCHAR(7),
    home_team_id INTEGER,
    away_team_id INTEGER,
    home_pts INTEGER,
    away_pts INTEGER
);
CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_games_season ON games(season);

-- Player game logs (core stats)
CREATE TABLE player_game_logs (
    game_id VARCHAR(10),
    player_id INTEGER,
    team_id INTEGER,
    minutes REAL,
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
    plus_minus REAL,
    PRIMARY KEY (game_id, player_id)
);
CREATE INDEX idx_pgl_player ON player_game_logs(player_id);
CREATE INDEX idx_pgl_team ON player_game_logs(team_id);
CREATE INDEX idx_pgl_game ON player_game_logs(game_id);

-- Team-level aggregates (Four Factors, SOS, SRS)
CREATE TABLE team_season_stats (
    team_id INTEGER,
    season VARCHAR(7),
    sos REAL,
    srs REAL,
    pace REAL,
    off_rtg REAL,
    def_rtg REAL,
    efg_pct REAL,
    tov_pct REAL,
    orb_pct REAL,
    ft_rate REAL,
    PRIMARY KEY (team_id, season)
);
2.2 nba_api Data Acquisition (src/data/nba_api_client.py)
Key endpoints to use (from Context7):
PlayerGameLog - Individual player stats per game
TeamGameLog - Team stats per game
LeagueGameLog - All games in a season

from nba_api.stats.endpoints import (
    PlayerGameLog, TeamGameLog, LeagueGameLog
)
import time

class NBADataClient:
    def __init__(self, rate_limit_delay: float = 0.6):
        self.delay = rate_limit_delay
    
    def get_player_game_logs(self, season: str) -> pd.DataFrame:
        """Fetch all player game logs for a season."""
        # Use LeagueGameLog with PlayerOrTeam='P'
        logs = LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="P",
            direction="ASC",
            sorter="DATE"
        )
        time.sleep(self.delay)
        return logs.get_data_frames()[0]
    
    def get_team_game_logs(self, season: str) -> pd.DataFrame:
        """Fetch all team game logs for a season."""
        logs = LeagueGameLog(
            season=season,
            season_type_all_star="Regular Season",
            player_or_team_abbreviation="T",
            direction="ASC",
            sorter="DATE"
        )
        time.sleep(self.delay)
        return logs.get_data_frames()[0]
2.3 Kaggle SOS/SRS Data (src/data/kaggle_loader.py)
Download and process SOS/SRS from Wyatt Walsh dataset (preferred over scraping).
2.4 Data Versioning
Store SHA-256 hashes of raw data files in data/hashes.json for reproducibility.
Phase 3: Feature Engineering Pipeline
3.1 Rolling Window Statistics (src/data/feature_eng.py)
Key requirement: per-game averages over games played only (not zeros for DNP).
def compute_rolling_stats(    player_logs: pd.DataFrame,    windows: list[int] = [10, 30],    as_of_date: str = None) -> pd.DataFrame:    """    Compute rolling stats strictly as of t-1.    DNP games are excluded from averages, not treated as zeros.    """    stats_cols = ['pts', 'reb', 'ast', 'stl', 'blk', 'fgm', 'fga', ...]        # Filter to games before as_of_date    if as_of_date:        player_logs = player_logs[player_logs['game_date'] < as_of_date]        # Only include games where player actually played    played = player_logs[player_logs['minutes'] > 0]        for window in windows:        for col in stats_cols:            # Per-game average over games played            player_logs[f'{col}_L{window}'] = (                played.groupby('player_id')[col]                .transform(lambda x: x.rolling(window, min_periods=1).mean())            )        # Availability feature: fraction of games played in window    player_logs['availability_L10'] = (        player_logs.groupby('player_id')['minutes']        .transform(lambda x: (x > 0).rolling(10).mean())    )        return player_logs
3.2 As-of-Date Roster Construction
Critical leakage prevention: roster selection based on minutes up to the prediction date only.
def get_roster_as_of_date(    player_logs: pd.DataFrame,    team_id: int,    as_of_date: str,    top_n: int = 15) -> list[int]:    """    Select top-N players by cumulative minutes as of date.    Never use full-season totals.    """    pre_date = player_logs[        (player_logs['team_id'] == team_id) &        (player_logs['game_date'] < as_of_date)    ]        minutes_to_date = pre_date.groupby('player_id')['minutes'].sum()    top_players = minutes_to_date.nlargest(top_n).index.tolist()        return top_players
3.3 Four Factors Computation
Calculate offensive and defensive Four Factors:
eFG% = (FGM + 0.5 * FG3M) / FGA
TOV% = TOV / (FGA + 0.44 * FTA + TOV)
ORB% = ORB / (ORB + Opp_DRB)
FT Rate = FTM / FGA
Phase 4: Model A - Deep Set Network
4.1 Architecture Overview
Output
Minutes-Weighted Attention
Player Encoder - Shared MLP
Input Layer
Team Strength Vector Z
Predicted Rank
Multi-Head Attention
Minutes Weights
MLP
MLP
MLP
Player 1 Stats
Player 2 Stats
Player N Stats
Player 1 Embedding
Player 2 Embedding
Player N Embedding
4.2 Player Embeddings (src/models/deep_set.py)
import torchimport torch.nn as nnclass PlayerEmbedding(nn.Module):    """Learnable player embeddings (R^32)."""        def __init__(self, num_players: int, embedding_dim: int = 32):        super().__init__()        self.embedding = nn.Embedding(num_players, embedding_dim)        def forward(self, player_ids: torch.Tensor) -> torch.Tensor:        return self.embedding(player_ids)
4.3 Player Encoder (Shared MLP)
class PlayerEncoder(nn.Module):    """Shared MLP that processes each player independently."""        def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float = 0.2):        super().__init__()        layers = []        prev_dim = input_dim        for dim in hidden_dims:            layers.extend([                nn.Linear(prev_dim, dim),                nn.ReLU(),                nn.Dropout(dropout)            ])            prev_dim = dim        self.encoder = nn.Sequential(*layers)        def forward(self, x: torch.Tensor) -> torch.Tensor:        # x: (batch, num_players, input_dim)        return self.encoder(x)
4.4 Minutes-Weighted Multi-Head Attention
Key design choice: use either explicit minutes-weighting or MP as encoder input, not both. Default: explicit minutes-weighting (no MP in encoder stats).
class MinutesWeightedAttention(nn.Module):    """Multi-head attention with minutes-based weighting."""        def __init__(self, embed_dim: int, num_heads: int = 4):        super().__init__()        self.attention = nn.MultiheadAttention(            embed_dim=embed_dim,            num_heads=num_heads,            batch_first=True        )        def forward(        self,        player_vectors: torch.Tensor,  # (batch, num_players, dim)        minutes: torch.Tensor,         # (batch, num_players)        mask: torch.Tensor = None      # (batch, num_players) - True for inactive    ) -> tuple[torch.Tensor, torch.Tensor]:        # Normalize minutes to attention bias        minutes_norm = minutes / (minutes.sum(dim=-1, keepdim=True) + 1e-8)                # Self-attention with minutes bias        attn_out, attn_weights = self.attention(            player_vectors, player_vectors, player_vectors,            key_padding_mask=mask        )                # Apply minutes weighting        weighted = attn_out * minutes_norm.unsqueeze(-1)        team_vector = weighted.sum(dim=1)  # (batch, dim)                return team_vector, attn_weights
4.5 Complete Deep Set Model
class DeepSetNetwork(nn.Module):    """Complete Model A: Deep Set with attention."""        def __init__(self, config: dict, num_players: int, num_stats: int):        super().__init__()        self.player_embed = PlayerEmbedding(num_players, config['embedding_dim'])                input_dim = num_stats + config['embedding_dim']        self.encoder = PlayerEncoder(input_dim, config['encoder_hidden'])                encoder_out_dim = config['encoder_hidden'][-1]        self.attention = MinutesWeightedAttention(encoder_out_dim, config['attention_heads'])                # Output head for ranking score        self.rank_head = nn.Linear(encoder_out_dim, 1)        def forward(        self,        player_ids: torch.Tensor,   # (batch, num_players)        player_stats: torch.Tensor, # (batch, num_players, num_stats)        minutes: torch.Tensor,      # (batch, num_players)        mask: torch.Tensor = None    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        # Get embeddings        embeddings = self.player_embed(player_ids)                # Concatenate stats + embeddings        x = torch.cat([player_stats, embeddings], dim=-1)                # Encode players        encoded = self.encoder(x)                # Attention aggregation        team_vector, attn_weights = self.attention(encoded, minutes, mask)                # Ranking score        score = self.rank_head(team_vector).squeeze(-1)                return score, team_vector, attn_weights
4.6 ListMLE Loss Implementation (src/models/losses.py)
def listMLE_loss(scores: torch.Tensor, relevance: torch.Tensor) -> torch.Tensor:    """    ListMLE loss for learning to rank.        Args:        scores: Model predicted scores (batch, list_size)        relevance: True relevance/ranking (batch, list_size)    """    # Sort by true relevance (descending)    _, indices = relevance.sort(dim=-1, descending=True)    sorted_scores = scores.gather(1, indices)        # Compute ListMLE    max_score = sorted_scores.max(dim=-1, keepdim=True)[0]    log_cumsum = torch.logcumsumexp(sorted_scores - max_score, dim=-1) + max_score        loss = (log_cumsum - sorted_scores).mean()    return loss
4.7 Game-Level List Construction
Critical: train on game-date lists (thousands of samples) instead of season-level (only ~40 lists).
def build_game_level_lists(    games_df: pd.DataFrame,    conference: str,    date: str) -> dict:    """    Build a ranking list for a specific conference-date.    Target = standings-to-date (win rate), not season-end.    """    # Get all teams in conference    teams = games_df[games_df['conference'] == conference]['team_id'].unique()        # Compute win rate as of this date    standings = []    for team_id in teams:        team_games = games_df[            (games_df['team_id'] == team_id) &            (games_df['game_date'] < date)        ]        if len(team_games) > 0:            win_rate = (team_games['wl'] == 'W').mean()            standings.append({'team_id': team_id, 'win_rate': win_rate})        # Convert to ranking target    standings_df = pd.DataFrame(standings).sort_values('win_rate', ascending=False)    standings_df['rank'] = range(1, len(standings_df) + 1)        return standings_df
Phase 5: Model B - Hybrid Tabular Ensemble
5.1 XGBoost Configuration (src/models/ensemble.py)
From Context7 documentation - use early stopping and proper CV:
import xgboost as xgbfrom sklearn.ensemble import RandomForestRegressorclass HybridEnsemble:    """Model B: XGBoost + Random Forest."""        def __init__(self, config: dict):        self.xgb_model = xgb.XGBRegressor(            max_depth=config['xgb']['max_depth'],            learning_rate=config['xgb']['learning_rate'],            n_estimators=config['xgb']['n_estimators'],            early_stopping_rounds=config['xgb']['early_stopping_rounds'],            eval_metric='ndcg',            random_state=config['seed']        )                self.rf_model = RandomForestRegressor(            n_estimators=config['rf']['n_estimators'],            max_depth=config['rf']['max_depth'],            random_state=config['seed'],            n_jobs=-1        )        def fit(self, X_train, y_train, X_val=None, y_val=None):        # XGBoost with early stopping        eval_set = [(X_val, y_val)] if X_val is not None else None        self.xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)                # Random Forest        self.rf_model.fit(X_train, y_train)        def predict(self, X) -> tuple[np.ndarray, np.ndarray]:        xgb_pred = self.xgb_model.predict(X)        rf_pred = self.rf_model.predict(X)        return xgb_pred, rf_pred
5.2 Feature Set for Model B
Inputs (no net_rating to avoid leakage):
Four Factors (eFG%, TOV%, ORB%, FT Rate)
SOS (Strength of Schedule)
SRS (Simple Rating System)
Pace
SOS-adjusted offensive/defensive ratings
Phase 6: Meta-Learner (True Stacking)
6.1 K-Fold OOF Predictions (src/models/meta_learner.py)
From Context7 scikit-learn docs - use cross_val_predict pattern:
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import pandas as pd

class MetaLearner:
    """Level-2 stacking with OOF predictions."""
    
    def __init__(self, n_folds: int = 5, seed: int = 42):
        self.n_folds = n_folds
        self.kfold = KFold(n_splits=n_folds, shuffle=False)  # Temporal order
        self.meta_model = LogisticRegression(random_state=seed)
        self.oof_predictions = {}
    
    def generate_oof_predictions(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str
    ) -> np.ndarray:
        """Generate out-of-fold predictions for stacking."""
        oof_preds = np.zeros(len(X))
        
        for fold, (train_idx, val_idx) in enumerate(self.kfold.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict(X_val)
        
        self.oof_predictions[model_name] = oof_preds
        return oof_preds
    
    def fit_meta_model(self, y: np.ndarray):
        """Fit level-2 model on stacked OOF predictions."""
        X_meta = np.column_stack(list(self.oof_predictions.values()))
        self.meta_model.fit(X_meta, y)
    
    def save_oof_predictions(self, output_dir: str):
        """Persist OOF predictions for diagnostics."""
        for name, preds in self.oof_predictions.items():
            pd.DataFrame({'oof_pred': preds}).to_parquet(
                f"{output_dir}/oof_{name}.parquet"
            )
Phase 7: Training Pipeline
7.1 Walk-Forward Temporal Validation (src/training/validation.py)
Critical: sports data is time-series, random splits are invalid.
def walk_forward_cv(    data: pd.DataFrame,    train_seasons: list[str],    val_season: str,    test_season: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:    """    Temporal train/val/test split.    Example:        Train: 2016-2022        Val: 2023        Test: 2024    """    train = data[data['season'].isin(train_seasons)]    val = data[data['season'] == val_season]    test = data[data['season'] == test_season]        return train, val, test
7.2 Training Loop (src/training/trainer.py)
def train_deep_set(    model: DeepSetNetwork,    train_loader: DataLoader,    val_loader: DataLoader,    config: dict,    device: str = 'cuda') -> dict:    """Complete training pipeline for Model A."""    model = model.to(device)    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(        optimizer, mode='min', patience=5    )        best_val_loss = float('inf')    history = {'train_loss': [], 'val_loss': []}        for epoch in range(config['epochs']):        # Training        model.train()        train_loss = 0        for batch in train_loader:            player_ids, stats, minutes, mask, targets = batch                        optimizer.zero_grad()            scores, _, _ = model(player_ids, stats, minutes, mask)            loss = listMLE_loss(scores, targets)            loss.backward()            optimizer.step()                        train_loss += loss.item()                # Validation        model.eval()        val_loss = 0        with torch.no_grad():            for batch in val_loader:                scores, _, _ = model(*batch[:-1])                val_loss += listMLE_loss(scores, batch[-1]).item()                scheduler.step(val_loss)                # Checkpointing        if val_loss < best_val_loss:            best_val_loss = val_loss            torch.save(model.state_dict(), 'outputs/models/best_deep_set.pth')                history['train_loss'].append(train_loss / len(train_loader))        history['val_loss'].append(val_loss / len(val_loader))        return history
Phase 8: Evaluation
8.1 Ranking Metrics (src/evaluation/metrics.py)
from scipy.stats import spearmanrfrom sklearn.metrics import ndcg_score, brier_score_loss, roc_auc_scoredef compute_ranking_metrics(    y_true: np.ndarray,    y_pred: np.ndarray) -> dict:    """Compute NDCG, Spearman, and MRR."""    # NDCG (top-heavy ranking quality)    ndcg = ndcg_score([y_true], [y_pred])        # Spearman correlation    spearman, _ = spearmanr(y_true, y_pred)        # MRR (mean reciprocal rank)    pred_ranks = np.argsort(np.argsort(-y_pred)) + 1    true_top = np.argmax(y_true)    mrr = 1.0 / pred_ranks[true_top]        return {'ndcg': ndcg, 'spearman': spearman, 'mrr': mrr}def compute_upset_metrics(    y_true_upset: np.ndarray,  # Binary: did upset occur?    y_pred_proba: np.ndarray   # Predicted probability) -> dict:    """ROC-AUC for sleeper/upset detection."""    roc_auc = roc_auc_score(y_true_upset, y_pred_proba)    brier = brier_score_loss(y_true_upset, y_pred_proba)    return {'roc_auc': roc_auc, 'brier': brier}
8.2 Baselines (src/evaluation/baselines.py)
def baseline_rank_by_srs(team_stats: pd.DataFrame) -> np.ndarray:    """Simple baseline: rank teams by SRS."""    return team_stats.sort_values('srs', ascending=False).index.valuesdef baseline_rank_by_net_rating(team_stats: pd.DataFrame) -> np.ndarray:    """Simple baseline: rank teams by net rating."""    team_stats['net_rtg'] = team_stats['off_rtg'] - team_stats['def_rtg']    return team_stats.sort_values('net_rtg', ascending=False).index.values
Phase 9: Explainability
9.1 SHAP for Model B (src/explainability/shap_model_b.py)
SHAP is restricted to Model B (RF/XGBoost) only:
import shapdef explain_model_b(    model,    X: np.ndarray,    feature_names: list[str]) -> shap.Explanation:    """SHAP TreeExplainer for Model B."""    explainer = shap.TreeExplainer(model)    shap_values = explainer.shap_values(X)        return shap.Explanation(        values=shap_values,        base_values=explainer.expected_value,        data=X,        feature_names=feature_names    )
9.2 Integrated Gradients for Model A (src/explainability/captum_model_a.py)
From Context7 Captum documentation:
from captum.attr import IntegratedGradientsdef explain_model_a(    model: DeepSetNetwork,    player_ids: torch.Tensor,    player_stats: torch.Tensor,    minutes: torch.Tensor,    baseline_stats: torch.Tensor = None) -> torch.Tensor:    """Integrated Gradients for Deep Set network."""    ig = IntegratedGradients(model)        if baseline_stats is None:        baseline_stats = torch.zeros_like(player_stats)        attributions, delta = ig.attribute(        inputs=player_stats,        baselines=baseline_stats,        target=0,        return_convergence_delta=True    )        return attributions
9.3 Attention Ablation Validation
Validate that attention weights correlate with importance:
def attention_ablation_test(    model: DeepSetNetwork,    batch: tuple,    top_k: int = 3) -> dict:    """    Compare accuracy drop when masking high-attention vs random players.    If drop is larger for high-attention, attention is meaningful.    """    player_ids, stats, minutes, mask, targets = batch        # Get attention weights    _, _, attn_weights = model(player_ids, stats, minutes, mask)        # Identify top-k attended players    top_attended = attn_weights.topk(top_k, dim=-1).indices        # Mask top-attended and measure accuracy drop    mask_top = mask.clone()    mask_top.scatter_(1, top_attended, True)    score_top_masked, _, _ = model(player_ids, stats, minutes, mask_top)        # Mask random players and measure accuracy drop    random_indices = torch.randint(0, mask.size(1), (mask.size(0), top_k))    mask_random = mask.clone()    mask_random.scatter_(1, random_indices, True)    score_random_masked, _, _ = model(player_ids, stats, minutes, mask_random)        return {        'drop_top_attended': (score_top_masked - targets).abs().mean().item(),        'drop_random': (score_random_masked - targets).abs().mean().item()    }
Phase 10: Visualization and Output
10.1 Visualization Suite (src/viz/plots.py)
import matplotlib.pyplot as pltimport seaborn as snsdef plot_accuracy(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):    """Predicted vs Actual rank scatter plot."""    plt.figure(figsize=(8, 8))    plt.scatter(y_true, y_pred, alpha=0.7)    plt.plot([1, 15], [1, 15], 'r--', label='Perfect prediction')    plt.xlabel('Actual Rank')    plt.ylabel('Predicted Rank')    plt.title('Predicted vs Actual Conference Rank')    plt.legend()    plt.savefig(save_path)    plt.close()def plot_sleeper_fraud_index(deltas: pd.DataFrame, save_path: str):    """Diverging bar chart of rank deltas."""    colors = ['green' if d > 0 else 'red' for d in deltas['delta']]        plt.figure(figsize=(12, 8))    plt.barh(deltas['team_name'], deltas['delta'], color=colors)    plt.axvline(x=0, color='black', linestyle='-')    plt.xlabel('Rank Delta (Actual - Predicted)')    plt.title('Fraud/Sleeper Index')    plt.savefig(save_path)    plt.close()def plot_shap_summary(shap_values, save_path: str):    """SHAP summary plot for Model B."""    shap.summary_plot(shap_values, show=False)    plt.savefig(save_path)    plt.close()def plot_attention_distribution(attn_weights: np.ndarray, team_names: list, save_path: str):    """Stacked area chart of attention weights per team."""    # Implementation for roster attention visualization    pass
10.2 JSON Output Generation
def generate_team_output(    team_name: str,    predicted_rank: int,    true_strength_score: float,    actual_rank: int,    ensemble_ranks: dict,    attention_weights: list) -> dict:    """Generate structured JSON output per team."""    delta = actual_rank - predicted_rank    classification = "Sleeper" if delta > 0 else "Paper Tiger" if delta < 0 else "Accurate"        return {        "team_name": team_name,        "prediction": {            "predicted_rank": predicted_rank,            "true_strength_score": round(true_strength_score, 3)        },        "analysis": {            "actual_rank": actual_rank,            "classification": f"{classification} ({abs(delta)} slots)",            "explanation": "Efficiency exceeds win-loss record due to difficult schedule."        },        "ensemble_diagnostics": {            "model_agreement": "High" if max(ensemble_ranks.values()) - min(ensemble_ranks.values()) <= 2 else "Low",            "deep_set_rank": ensemble_ranks['deep_set'],            "xgboost_rank": ensemble_ranks['xgboost'],            "random_forest_rank": ensemble_ranks['rf']        },        "roster_dependence": {            "star_reliance_score": "High" if attention_weights[0]['weight'] > 0.3 else "Balanced",            "primary_contributors": attention_weights[:3]        }    }
Implementation Sequence
The recommended order of implementation:
Phase 1: Project setup (1-2 days)
Phase 2: Data acquisition and DuckDB (3-5 days)
Phase 3: Feature engineering (3-4 days)
Phase 4: Model A Deep Set (5-7 days)
Phase 5: Model B Ensemble (2-3 days)
Phase 6: Meta-learner stacking (2-3 days)
Phase 7: Training pipeline (3-4 days)
Phase 8: Evaluation (2-3 days)
Phase 9: Explainability (2-3 days)
Phase 10: Visualization and output (2-3 days)
Key Technical References from Context7
PyTorch MultiheadAttention: Use batch_first=True, key_padding_mask for inactive players
XGBoost early stopping: Set early_stopping_rounds and pass eval_set to fit()
scikit-learn StackingRegressor: Alternative to manual OOF if simpler implementation needed
DuckDB Python: Use duckdb.read_parquet() for efficient data loading
Captum IntegratedGradients: Use return_convergence_delta=True to verify attribution quality
nba_api endpoints: LeagueGameLog with player_or_team_abbreviation='P' for player logs