---
name: Sweep Rerun + Attention Check
overview: Rerun the sweep with a larger validation slice (medium profile) and a higher epoch range, then add a small attention diagnostic so we can answer whether Model A’s attention is producing non‑degenerate weights.
todos:
  - id: config-medium-profile
    content: Update defaults.yaml to medium data profile values
    status: pending
  - id: attention-diagnostic
    content: Add attention stats logging to sweep script
    status: pending
  - id: rerun-sweep
    content: Run sweep with epochs 15,20,25,30,35,40
    status: pending
  - id: report-results
    content: Summarize sweep metrics + attention diagnostics
    status: pending
isProject: false
---

# Sweep Rerun + Attention Check

## Scope and approach

- Use the sweep script with a higher epoch list and the medium data profile (larger val set), then rerun the batch.
- Add a lightweight attention diagnostic that inspects `attn_w` returned by `DeepSetRank` to confirm weights are finite and non‑zero.

## Proposed changes

- Update config values for the medium profile in `[C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\config\defaults.yaml](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\config\defaults.yaml)`:
  - `model_a.early_stopping_val_frac = 0.20`
  - `training.max_lists_oof = 30`
  - `training.model_a_history_days = 120`
- Add attention diagnostics in `[C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\scripts\sweep_hparams.py](C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\scripts\sweep_hparams.py)` using the existing `attn_w` output from `DeepSetRank`:

```
29:47:C:\Users\tmaku\.cursor\worktrees\NBA_Playoff_Strentgh_Project\hbf\src\models\deep_set_rank.py
    def forward(
        self,
        embedding_indices: torch.Tensor,
        player_stats: torch.Tensor,
        minutes: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ...
        pooled, attn_w = self.attn(z, key_padding_mask=key_padding_mask, minutes=minutes)
        # ...
        return score, Z, attn_w
```

## Execution steps

- Run the sweep with epochs `15,20,25,30,35,40`:
  - `python -u "scripts/sweep_hparams.py" --batch epochs_plus_model_b --epochs 15,20,25,30,35,40`
- Record results in `outputs/sweeps/batch_001` and summarize:
  - Best Model A epoch (NDCG/Spearman/MRR)
  - Best Model B combo (RMSE mean and Spearman mean)
  - Attention diagnostic stats (min/mean/max, % zeros, NaNs)

## What this answers

- Whether a larger validation slice changes the “best epoch” for Model A.
- Whether Model B hyperparameters shift with more data.
- Whether Model A attention weights are non‑degenerate on real data.

