---
name: Sweep foreground timing early exit
overview: Run the hyperparameter sweep in the foreground with wall-clock timing, optional early exit when models are not learning or when attention/players become masked again, light verbose/debug option, and document what is deactivated for faster sweeps.
todos: []
isProject: false
---

# Sweep: foreground, timing, early exit, and speed settings

## Goals

1. **Never run sweep in background** — always run in foreground so you see progress and get a clean exit with duration.
2. **Report how long the sweep took** — print elapsed time when the sweep finishes.
3. **Exit early** if (a) models are not learning, or (b) players become masked again (attention fallback / all-zero).
4. **Optional debug/verbose** — useful extra logging without slowing the sweep much; no heavy debugger in child processes.
5. **Document what is deactivated for faster sweeps** — list config/behavior that keeps sweep runtime lower.

---

## 1. Run in foreground and report duration

**Invocation**

- Run from project root: `python -m scripts.sweep_hparams` (no background). On PowerShell use `;` not `&&` if chaining with `cd`.
- No code change needed for "never background" — just never pass `is_background: true` when starting the sweep.

**Timing in [scripts/sweep_hparams.py**](scripts/sweep_hparams.py)

- At the start of `main()` (after parsing args and creating `batch_dir`): record `start_time = time.perf_counter()` (add `import time`).
- Just before `return 0` at the end of `main()`: compute `elapsed = time.perf_counter() - start_time`, convert to minutes (and hours if &gt; 60 min), and `print(f"Sweep finished in {elapsed_min:.1f} min (...)" , flush=True)`.

---

## 2. Exit early if models are not learning

**Definition**

- "Not learning" = many consecutive trials with very poor or invalid objective (e.g. `-inf` or below a small threshold).

**Implementation (Bayesian mode only)**

- In [scripts/sweep_hparams.py](scripts/sweep_hparams.py), inside the Optuna `objective(trial)` (or around it), maintain a running count of consecutive "poor" trials: e.g. objective is `float("-inf")` or `<= threshold` (e.g. `0.01`).
- After each trial, if the last `K` trials (e.g. 5) are all poor, call `study.stop()` and break out; print a message like `Stopping sweep: models not learning (K consecutive poor/invalid trials).`
- Make K and threshold configurable via sweep config (e.g. `sweep.early_stop_no_learning: true`, `sweep.early_stop_consecutive_poor: 5`, `sweep.early_stop_metric_threshold: 0.01`) or CLI flags so they can be tuned/disabled.

**Note:** Grid mode runs a fixed set of combos; early-stop there would mean "abort remaining combos" if the last K combos all fail or are below threshold — same idea, optional.

---

## 3. Exit early if players become masked again

**Definition**

- "Players masked" = attention unusable: many teams with **attention fallback** (`contributors_are_fallback`) or **all-zero attention** (`all_zero` in attn_debug). This was a past bug; we want to stop the sweep if it reappears.

**Data to use**

- Inference already computes `attn_debug` (e.g. in [src/inference/predict.py](src/inference/predict.py) around 359–447) and `attention_fallback_by_team`; these are used for logging and for `contributors_are_fallback` in predictions.

**Implementation**

1. **Write diagnostics from inference**
  - In [src/inference/predict.py](src/inference/predict.py), in `run_inference_from_db`, after running inference for the primary test spec(s), aggregate:
    - Total teams with `attention_fallback_by_team[tid] == True`.
    - From `attn_debug`: `all_zero`, `empty_roster`, and total teams.
  - Write a small JSON file under the run output dir (e.g. `out / "inference_diagnostics.json"`) with keys such as: `attention_fallback_count`, `all_zero_count`, `empty_roster_count`, `n_teams`. So the sweep can read it per trial.
2. **Sweep reads and reacts**
  - In [scripts/sweep_hparams.py](scripts/sweep_hparams.py), after `_run_pipeline` returns success, determine the run dir for that trial (e.g. `combo_out` is `trial_XXXX/outputs`; script 6 writes to `output_dir/run_id`, and sweep sets `paths.outputs` to `combo_out`, so the run dir is e.g. `combo_out / "run_001"` or whatever `run_id` is). Load `inference_diagnostics.json` from that run dir.
  - If `attention_fallback_count` or `all_zero_count` is above a threshold (e.g. &gt; half of `n_teams`), treat the trial as "masked again": return `float("-inf")` for the objective and increment a "masked_streak" counter. If `masked_streak >= N` (e.g. 3), call `study.stop()` and print `Stopping sweep: players masked again (N trials with high fallback/all_zero).`
  - Threshold and N configurable via sweep config or CLI (e.g. `sweep.early_stop_masked_ratio: 0.5`, `sweep.early_stop_masked_consecutive: 3`).

---

## 4. Debugger / verbose with sweep

**Requirements**

- Useful for debugging the sweep itself, without slowing child scripts too much.

**Approach**

- **Sweep only:** Add a `--verbose` flag to [scripts/sweep_hparams.py](scripts/sweep_hparams.py). When set:
  - Print per-script exit code after each step of `_run_pipeline` (e.g. "Model A: exit 0", "Model B: exit 0", ...).
  - Print trial start/end wall time for each trial (e.g. "Trial 5 started at ...", "Trial 5 finished in X.X min").
- **No debugger inside child processes** — running `python -m pdb` for scripts 3/4/6 would require changing those scripts or subprocess invocation and would slow every trial. So:
  - Document: to debug the **sweep loop** (e.g. between trials), run: `python -m pdb -m scripts.sweep_hparams [args]`. No code change; user runs this when they need a breakpoint in the sweep script.
- Optional: if `SWEEP_DEBUG=1` is set, sweep could print more (e.g. config path per trial, objective value right after collection). Keep it minimal so it does not add noticeable delay.

---

## 5. What is deactivated (or set) for faster sweeps

These are the main knobs that keep sweep runtime lower; document them (e.g. in [config/README.md](config/README.md) or a short "Sweep" section in [README.md](README.md)):


| Setting                          | Where                                                                                           | Effect                                                                                                                     |
| -------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **include_clone_classifier**     | `sweep.include_clone_classifier` (default true in [config/defaults.yaml](config/defaults.yaml)) | When **false**, sweep skips script 4c (clone classifier); fewer steps per trial, faster. Set to `false` for maximum speed. |
| **max_lists_oof**                | `training.max_lists_oof` (30)                                                                   | Caps how many lists are used for OOF in Model A; fewer lists = faster script 3.                                            |
| **max_final_batches**            | `training.max_final_batches` (50)                                                               | Caps lists for final Model A training; fewer = faster.                                                                     |
| **build_lists date subsampling** | [src/training/build_lists.py](src/training/build_lists.py) (e.g. 200 dates max)                 | Fewer dates = fewer lists = faster batch build and training.                                                               |
| **attention_debug**              | `model_a.attention_debug` (false)                                                               | When false, script 3 does not log attention stats every epoch; less I/O and logging.                                       |
| **walk_forward**                 | `training.walk_forward` (false)                                                                 | Pooled OOF instead of per-season walk-forward; one training path per trial.                                                |
| **batch_cache**                  | `training.batch_cache.enabled` (true)                                                           | **Enabled** to speed up: reuse batches when config/inputs unchanged across trials.                                         |


So for "what we deactivated for speed": we **cap** work (max_lists_oof, max_final_batches, date subsampling), **disable** extra logging (attention_debug), **skip** optional steps when not needed (include_clone_classifier false), and **avoid** walk-forward; we **enable** batch_cache to reuse work. Add a short subsection in the README or config README listing these.

---

## 6. Implementation order (suggested)

1. Add timing (start/end + print elapsed) and `--verbose` in [scripts/sweep_hparams.py](scripts/sweep_hparams.py).
2. Add writing of `inference_diagnostics.json` in [src/inference/predict.py](src/inference/predict.py) (run_inference_from_db).
3. In sweep: read `inference_diagnostics.json` after pipeline; implement "masked again" check and optional `study.stop()`.
4. In sweep: implement "models not learning" (consecutive poor trials) and optional `study.stop()` in Bayesian mode.
5. Document "what is deactivated for faster sweeps" in README or config README.
6. Document "run sweep in foreground" and "debug sweep with pdb" in README (invocation + `python -m pdb -m scripts.sweep_hparams`).

---

## 7. Files to touch


| File                                                           | Changes                                                                                                                                                                                                                                 |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [scripts/sweep_hparams.py](scripts/sweep_hparams.py)           | Import `time`; record start time; print elapsed at end; add `--verbose`; read inference_diagnostics.json after pipeline; early-stop logic for "not learning" and "masked again"; optional config keys for thresholds.                   |
| [src/inference/predict.py](src/inference/predict.py)           | After test_spec inference, aggregate attn_debug and fallback counts; write `inference_diagnostics.json` under run dir.                                                                                                                  |
| [config/defaults.yaml](config/defaults.yaml)                   | Optional: add `sweep.early_stop_no_learning`, `sweep.early_stop_consecutive_poor`, `sweep.early_stop_metric_threshold`, `sweep.early_stop_masked_ratio`, `sweep.early_stop_masked_consecutive` (or keep as code defaults and document). |
| [README.md](README.md) or [config/README.md](config/README.md) | Short subsection: run sweep in foreground; report duration; early exit conditions; what is deactivated for faster sweeps; how to debug sweep with pdb.                                                                                  |


No change to how the process is started (no background); timing and early-exit are inside the sweep script so the run "takes as long as it needs" until it finishes or stops early.