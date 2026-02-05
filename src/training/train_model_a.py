"""Train Model A (DeepSet + ListMLE). Checkpointing, walk-forward train/val seasons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.deep_set_rank import DeepSetRank
from src.models.listmle_loss import listmle_loss
from src.utils.repro import set_seeds


def _grad_norm_for_module(module: nn.Module) -> float:
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def _log_attention_debug_stats(model: nn.Module, batch: dict, device: torch.device) -> None:
    """Log one-shot attention diagnostics: encoder z, Z, scores, attn, grad norms, player_stats."""
    B, K, P, S = batch["embedding_indices"].shape[0], batch["embedding_indices"].shape[1], batch["embedding_indices"].shape[2], batch["player_stats"].shape[-1]
    embs = batch["embedding_indices"].to(device).reshape(B * K, P)
    stats = batch["player_stats"].to(device).reshape(B * K, P, S)
    minutes = batch["minutes"].to(device).reshape(B * K, P)
    mask = batch["key_padding_mask"].to(device).reshape(B * K, P)

    # Capture encoder output z via hook
    z_captured: list[torch.Tensor] = []

    def _hook(_module: nn.Module, _input: Any, output: torch.Tensor) -> None:
        z_captured.append(output.detach())

    handle = model.enc.register_forward_hook(_hook)
    model.train()
    model.zero_grad(set_to_none=True)
    score, Z_out, attn_w = model(embs, stats, minutes, mask)
    handle.remove()
    Z_out = Z_out.detach()
    score_detach = score.detach()

    loss = score.mean()
    if torch.isfinite(loss).all():
        loss.backward()

    grad_norm_global = _grad_norm_for_module(model)
    grad_norm_enc = _grad_norm_for_module(model.enc)
    grad_norm_attn = _grad_norm_for_module(model.attn)
    grad_norm_scorer = _grad_norm_for_module(model.scorer)
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        attn_w = attn_w.reshape(B, K, P)
        all_masked = mask.reshape(B, K, P).all(dim=-1)
        valid_mask = ~mask
        minutes_valid = minutes[valid_mask]
        attn_sum = attn_w.sum(dim=-1).reshape(-1)
        attn_max = attn_w.max(dim=-1).values.reshape(-1)
        attn_sum_mean = float(torch.nanmean(attn_sum).item()) if attn_sum.numel() else 0.0
        attn_max_mean = float(torch.nanmean(attn_max).item()) if attn_max.numel() else 0.0

        # Encoder z variance across teams (per-team vector -> variance over K)
        z_var_str = "n/a"
        if z_captured:
            z = z_captured[0].reshape(B, K, P, -1)
            z_per_team = z.mean(dim=2)
            z_var = z_per_team.var(dim=1).mean().item()
            z_var_str = f"{z_var:.6f}"

        # Z variance across teams
        Z_reshaped = Z_out.reshape(B, K, -1)
        Z_var = float(Z_reshaped.var(dim=1).mean().item()) if Z_reshaped.numel() > 0 else 0.0

        # Score stats (before nan_to_num in loss)
        score_flat = score_detach.reshape(B, K)
        score_min = float(score_flat.min().item())
        score_max = float(score_flat.max().item())
        score_mean = float(score_flat.mean().item())
        score_nan = int(torch.isnan(score_flat).sum().item())

        # player_stats: shape, min, max, mean
        ps_min = float(stats.min().item())
        ps_max = float(stats.max().item())
        ps_mean = float(stats.mean().item())

        print(
            "Attention debug (train):",
            f"teams={attn_sum.numel()}",
            f"all_masked={int(all_masked.sum().item())}",
            f"minutes_min={float(minutes_valid.min().item()) if minutes_valid.numel() else 0.0:.4f}",
            f"minutes_mean={float(minutes_valid.mean().item()) if minutes_valid.numel() else 0.0:.4f}",
            f"minutes_max={float(minutes_valid.max().item()) if minutes_valid.numel() else 0.0:.4f}",
            f"attn_sum_mean={attn_sum_mean:.4f}",
            f"attn_max_mean={attn_max_mean:.4f}",
            f"attn_grad_norm={grad_norm_attn:.4f}",
            f"grad_norm_global={grad_norm_global:.4f}",
            f"grad_norm_enc={grad_norm_enc:.4f}",
            f"grad_norm_scorer={grad_norm_scorer:.4f}",
            f"z_var={z_var_str}",
            f"Z_var={Z_var:.6f}",
            f"score_min={score_min:.4f}",
            f"score_max={score_max:.4f}",
            f"score_mean={score_mean:.4f}",
            f"score_nan={score_nan}",
            f"player_stats_shape={tuple(stats.shape)}",
            f"player_stats_min={ps_min:.4f}",
            f"player_stats_max={ps_max:.4f}",
            f"player_stats_mean={ps_mean:.4f}",
            flush=True,
        )


def get_dummy_batch(
    batch_size: int = 4,
    num_teams_per_list: int = 10,
    num_players: int = 15,
    stat_dim: int = 7,
    num_embeddings: int = 500,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Dummy batch for ListMLE: B lists, each with K teams; each team has P players."""
    B, K, P, S = batch_size, num_teams_per_list, num_players, stat_dim
    embs = torch.randint(0, num_embeddings, (B, K, P), device=device)
    stats = torch.randn(B, K, P, S, device=device) * 0.1
    minutes = torch.rand(B, K, P, device=device)
    mask = torch.zeros(B, K, P, dtype=torch.bool, device=device)
    mask[:, :, 10:] = True  # last 5 are padding
    rel = torch.rand(B, K, device=device)  # fake relevance

    return {
        "embedding_indices": embs,
        "player_stats": stats,
        "minutes": minutes,
        "key_padding_mask": mask,
        "rel": rel,
    }


def train_epoch(
    model: nn.Module,
    batches: list[dict],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip_max_norm: float = 1.0,
    attention_debug: bool = False,
    use_amp: bool = False,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> float:
    model.train()
    total = 0.0
    n = 0
    first_batch = True
    amp_enabled = use_amp and device.type == "cuda" and torch.cuda.is_available()
    use_scaler = amp_enabled and scaler is not None

    for batch in batches:
        B, K, P, S = batch["embedding_indices"].shape[0], batch["embedding_indices"].shape[1], batch["embedding_indices"].shape[2], batch["player_stats"].shape[-1]
        embs = batch["embedding_indices"].to(device).reshape(B * K, P)
        stats = batch["player_stats"].to(device).reshape(B * K, P, S)
        minutes = batch["minutes"].to(device).reshape(B * K, P)
        mask = batch["key_padding_mask"].to(device).reshape(B * K, P)
        rel = batch["rel"].to(device)

        if attention_debug and first_batch:
            with torch.no_grad():
                rel_ = rel.cpu()
                print(
                    "Rel (first batch):",
                    f"shape={tuple(rel_.shape)}",
                    flush=True,
                )
                for row in range(rel_.shape[0]):
                    r = rel_[row]
                    print(
                        f"  row_{row} rel_min={float(r.min().item()):.4f}",
                        f"rel_max={float(r.max().item()):.4f}",
                        f"rel_mean={float(r.mean().item()):.4f}",
                        flush=True,
                    )
                st = stats.cpu()
                print(
                    "Player_stats (first batch):",
                    f"shape={tuple(st.shape)}",
                    f"min={float(st.min().item()):.4f}",
                    f"max={float(st.max().item()):.4f}",
                    f"mean={float(st.mean().item()):.4f}",
                    flush=True,
                )
            first_batch = False

        optimizer.zero_grad(set_to_none=True)
        if amp_enabled:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                score, _, _ = model(embs, stats, minutes, mask)
            score = score.float().reshape(B, K)
        else:
            score, _, _ = model(embs, stats, minutes, mask)
            score = score.reshape(B, K)
        score = torch.nan_to_num(score, nan=0.0, posinf=10.0, neginf=-10.0)
        loss = listmle_loss(score, rel)
        if not torch.isfinite(loss).all():
            continue
        if use_scaler:
            scaler.scale(loss).backward()
            if attention_debug and n == 0:
                grad_norm_before = _grad_norm_for_module(model)
                print(
                    "Grad norm before clip (first batch):",
                    f"grad_norm={grad_norm_before:.4f}",
                    f"max_norm={grad_clip_max_norm}",
                    flush=True,
                )
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if attention_debug and n == 0:
                grad_norm_before = _grad_norm_for_module(model)
                print(
                    "Grad norm before clip (first batch):",
                    f"grad_norm={grad_norm_before:.4f}",
                    f"max_norm={grad_clip_max_norm}",
                    flush=True,
                )
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_max_norm)
            optimizer.step()
        total += loss.item()
        n += 1
    return total / n if n else 0.0


def eval_epoch(
    model: nn.Module,
    batches: list[dict],
    device: torch.device,
    *,
    use_amp: bool = False,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    amp_enabled = use_amp and device.type == "cuda" and torch.cuda.is_available()
    with torch.no_grad():
        for batch in batches:
            B, K, P, S = batch["embedding_indices"].shape[0], batch["embedding_indices"].shape[1], batch["embedding_indices"].shape[2], batch["player_stats"].shape[-1]
            embs = batch["embedding_indices"].to(device).reshape(B * K, P)
            stats = batch["player_stats"].to(device).reshape(B * K, P, S)
            minutes = batch["minutes"].to(device).reshape(B * K, P)
            mask = batch["key_padding_mask"].to(device).reshape(B * K, P)
            rel = batch["rel"].to(device)

            if amp_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    score, _, _ = model(embs, stats, minutes, mask)
                score = score.float().reshape(B, K)
            else:
                score, _, _ = model(embs, stats, minutes, mask)
                score = score.reshape(B, K)
            score = torch.nan_to_num(score, nan=0.0, posinf=10.0, neginf=-10.0)
            loss = listmle_loss(score, rel)
            if torch.isfinite(loss).all():
                total += loss.item()
                n += 1
    return total / n if n else 0.0


def _split_train_val(
    batches: list[dict],
    val_frac: float,
) -> tuple[list[dict], list[dict]]:
    if not batches or val_frac <= 0:
        return batches, []
    n = len(batches)
    if n < 5:
        return batches, []
    n_val = max(1, int(n * val_frac))
    return batches[:-n_val], batches[-n_val:]


def predict_batches(
    model: nn.Module,
    batches: list[dict],
    device: torch.device,
) -> list[torch.Tensor]:
    """Run model in eval mode on batches; return list of score tensors (1, K) per batch."""
    model.eval()
    scores_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches:
            B, K, P, S = batch["embedding_indices"].shape[0], batch["embedding_indices"].shape[1], batch["embedding_indices"].shape[2], batch["player_stats"].shape[-1]
            embs = batch["embedding_indices"].to(device).reshape(B * K, P)
            stats = batch["player_stats"].to(device).reshape(B * K, P, S)
            minutes = batch["minutes"].to(device).reshape(B * K, P)
            mask = batch["key_padding_mask"].to(device).reshape(B * K, P)
            score, _, _ = model(embs, stats, minutes, mask)
            score = score.reshape(B, K)
            scores_list.append(score.cpu())
    return scores_list


def predict_batches_with_attention(
    model: nn.Module,
    batches: list[dict],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run model in eval mode; return (scores_list, attn_weights_list). attn_weights_list[i] is (K, P)."""
    model.eval()
    scores_list: list[torch.Tensor] = []
    attn_list: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in batches:
            B, K, P, S = batch["embedding_indices"].shape[0], batch["embedding_indices"].shape[1], batch["embedding_indices"].shape[2], batch["player_stats"].shape[-1]
            embs = batch["embedding_indices"].to(device).reshape(B * K, P)
            stats = batch["player_stats"].to(device).reshape(B * K, P, S)
            minutes = batch["minutes"].to(device).reshape(B * K, P)
            mask = batch["key_padding_mask"].to(device).reshape(B * K, P)
            score, _, attn_w = model(embs, stats, minutes, mask)
            score = score.reshape(B, K)
            attn_w = attn_w.reshape(B, K, P)
            scores_list.append(score.cpu())
            attn_list.append(attn_w.cpu())
    return scores_list, attn_list


NOT_LEARNING_PATIENCE = 3  # stop when train loss does not improve for this many consecutive epochs

NOT_LEARNING_ANALYSIS = (
    "Model A is not learning: train loss did not improve. Typical cause (see docs/CHECKPOINT_PROJECT_REPORT.md, "
    "fix_attention plan): attention has collapsed (all-zero or uniform), so the pooled representation Z is "
    "constant across teams -> constant scores -> ListMLE loss is fixed for that list length. Fix: harden "
    "set_attention (minutes reweighting, uniform fallback when raw attention is zero) so gradients flow."
)


def _build_model(config: dict, device: torch.device, stat_dim_override: int | None = None) -> nn.Module:
    ma = config.get("model_a", {})
    num_emb = ma.get("num_embeddings", 500)
    emb_dim = ma.get("embedding_dim", 32)
    stat_dim = int(stat_dim_override) if stat_dim_override is not None else int(ma.get("stat_dim", 14))
    enc_h = ma.get("encoder_hidden", [128, 64])
    heads = ma.get("attention_heads", 4)
    drop = ma.get("dropout", 0.2)
    minutes_bias_weight = float(ma.get("minutes_bias_weight", 0.3))
    minutes_sum_min = float(ma.get("minutes_sum_min", 1e-6))
    fallback_strategy = str(ma.get("attention_fallback_strategy", "minutes"))
    return DeepSetRank(
        num_emb,
        emb_dim,
        stat_dim,
        enc_h,
        heads,
        drop,
        minutes_bias_weight=minutes_bias_weight,
        minutes_sum_min=minutes_sum_min,
        fallback_strategy=fallback_strategy,
    ).to(device)


def train_model_a_on_batches(
    config: dict,
    batches: list[dict],
    device: torch.device,
    max_epochs: int | None = None,
    *,
    val_batches: list[dict] | None = None,
) -> nn.Module:
    """Train Model A on given batches; return the model (do not save). For OOF fold training."""
    ma = config.get("model_a", {})
    num_emb = ma.get("num_embeddings", 500)
    stat_dim_override = int(batches[0]["player_stats"].shape[-1]) if batches else None
    if not batches:
        model = _build_model(config, device)
        return model
    model = _build_model(config, device, stat_dim_override=stat_dim_override)
    lr = float(ma.get("learning_rate", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    grad_clip_max_norm = float(ma.get("grad_clip_max_norm", 1.0))
    attention_debug = bool(ma.get("attention_debug", False))
    use_amp = bool(ma.get("use_amp", False)) and device.type == "cuda" and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    epochs = int(max_epochs) if max_epochs is not None else int(ma.get("epochs", 20))
    patience = int(ma.get("early_stopping_patience", 3))
    min_delta = float(ma.get("early_stopping_min_delta", 0.0))
    use_early = bool(val_batches) and patience > 0
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    best_train_loss = float("inf")
    train_no_improve = 0
    for epoch in range(epochs):
        loss = train_epoch(
            model,
            batches,
            optimizer,
            device,
            grad_clip_max_norm=grad_clip_max_norm,
            attention_debug=attention_debug,
            use_amp=use_amp,
            scaler=scaler,
        )
        print(f"epoch {epoch+1} loss={loss:.4f}", flush=True)
        if loss < best_train_loss:
            best_train_loss = loss
            train_no_improve = 0
        else:
            train_no_improve += 1
        if train_no_improve >= NOT_LEARNING_PATIENCE:
            print("Stopping: train loss did not improve for {} epoch(s).".format(NOT_LEARNING_PATIENCE), flush=True)
            print(NOT_LEARNING_ANALYSIS, flush=True)
            if batches and bool(ma.get("attention_debug", False)):
                try:
                    _log_attention_debug_stats(model, batches[0], device)
                except Exception:
                    pass
            break
        if use_early:
            val_loss = eval_epoch(model, val_batches or [], device, use_amp=use_amp)
            if val_loss + min_delta < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_model_a(
    config: dict,
    output_dir: str | Path,
    device: str | torch.device | None = None,
    batches: list[dict] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(config.get("repro", {}).get("seed", 42))

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device) if isinstance(device, str) else device

    ma = config.get("model_a", {})
    stat_dim = int(ma.get("stat_dim", 14))
    num_emb = ma.get("num_embeddings", 500)
    stat_dim_override = int(batches[0]["player_stats"].shape[-1]) if batches else None
    model = _build_model(config, device, stat_dim_override=stat_dim_override)
    lr = float(ma.get("learning_rate", 1e-3))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    grad_clip_max_norm = float(ma.get("grad_clip_max_norm", 1.0))
    attention_debug = bool(ma.get("attention_debug", False))
    use_amp = bool(ma.get("use_amp", False)) and device.type == "cuda" and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    if batches is None:
        batches = [get_dummy_batch(4, 10, 15, stat_dim, num_emb, device) for _ in range(5)]
    if not batches:
        batches = [get_dummy_batch(4, 10, 15, stat_dim, num_emb, device) for _ in range(5)]
    epochs = int(ma.get("epochs", 20))
    val_frac = float(ma.get("early_stopping_val_frac", 0.1))
    patience = int(ma.get("early_stopping_patience", 3))
    min_delta = float(ma.get("early_stopping_min_delta", 0.0))
    train_batches, val_batches = _split_train_val(batches, val_frac)
    use_early = bool(val_batches) and patience > 0
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    best_train_loss = float("inf")
    train_no_improve = 0
    for epoch in range(epochs):
        loss = train_epoch(
            model,
            train_batches,
            optimizer,
            device,
            grad_clip_max_norm=grad_clip_max_norm,
            attention_debug=attention_debug,
        )
        print(f"epoch {epoch+1} loss={loss:.4f}", flush=True)
        if loss < best_train_loss:
            best_train_loss = loss
            train_no_improve = 0
        else:
            train_no_improve += 1
        if train_no_improve >= NOT_LEARNING_PATIENCE:
            print("Stopping: train loss did not improve for {} epoch(s).".format(NOT_LEARNING_PATIENCE), flush=True)
            print(NOT_LEARNING_ANALYSIS, flush=True)
            if train_batches and bool(ma.get("attention_debug", False)):
                try:
                    _log_attention_debug_stats(model, train_batches[0], device)
                except Exception:
                    pass
            break
        if use_early:
            val_loss = eval_epoch(model, val_batches, device, use_amp=use_amp)
            print(f"val_loss {epoch+1}: {val_loss:.4f}", flush=True)
            if val_loss + min_delta < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    if batches and bool(ma.get("attention_debug", False)):
        try:
            _log_attention_debug_stats(model, batches[0], device)
        except Exception:
            pass

    path = output_dir / "best_deep_set.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, path)
    return path
