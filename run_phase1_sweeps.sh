#!/usr/bin/env bash
# Run all 12 Phase I Optuna sweeps in sequence (WSL + venv).
# Usage: bash run_phase1_sweeps.sh   (or: ./run_phase1_sweeps.sh)

set -e
cd "/mnt/c/Users/tmaku/OneDrive/Documents/GSU/Advanced Machine Learning/NBA Playoff Strentgh Project"
if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi
export PYTHONPATH="$PWD"
cfg="config/outputs4_phase1.yaml"

python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective spearman --listmle-target final_rank --phase phase1 --batch-id phase1_spearman_final_rank --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective spearman --listmle-target playoff_outcome --phase phase1 --batch-id phase1_spearman_playoff_outcome --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective ndcg4 --listmle-target final_rank --phase phase1 --batch-id phase1_ndcg4_final_rank --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective ndcg4 --listmle-target playoff_outcome --phase phase1 --batch-id phase1_ndcg4_playoff_outcome --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective ndcg16 --listmle-target final_rank --phase phase1 --batch-id phase1_ndcg16_final_rank --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective ndcg16 --listmle-target playoff_outcome --phase phase1 --batch-id phase1_ndcg16_playoff_outcome --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective ndcg20 --listmle-target final_rank --phase phase1 --batch-id phase1_ndcg20_final_rank --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective ndcg20 --listmle-target playoff_outcome --phase phase1 --batch-id phase1_ndcg20_playoff_outcome --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective playoff_spearman --listmle-target final_rank --phase phase1 --batch-id phase1_playoff_spearman_final_rank --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective playoff_spearman --listmle-target playoff_outcome --phase phase1 --batch-id phase1_playoff_spearman_playoff_outcome --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective rank_rmse --listmle-target final_rank --phase phase1 --batch-id phase1_rank_rmse_final_rank --config "$cfg"
python -m scripts.sweep_hparams --method optuna --n-trials 12 --n-jobs 4 --no-run-explain --objective rank_rmse --listmle-target playoff_outcome --phase phase1 --batch-id phase1_rank_rmse_playoff_outcome --config "$cfg"

echo "All 12 Phase I sweeps finished."
