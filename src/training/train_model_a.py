"""Train Model A (DeepSet + ListMLE). Checkpointing, walk-forward train/val seasons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.deep_set_rank import DeepSetRank
from src.models.listmle_loss import listmle_loss
from src.utils.repro import set_seeds


def _log_attention_debug_stats(model: nn.Module, batch: dict, device: torch.device) -> None:
    """Log one-shot attention diagnostics: mask/minutes/attn sums + grad norm."""
    B, K, P, S = batch["embedding_indices"].shape[0], batch["embedding_indices"].shape[1], batch["embedding_indices"].shape[2], batch["player_stats"].shape[-1]
    embs = batch["embedding_indices"].to(device).reshape(B * K, P)
    stats = batch["player_stats"].to(device).reshape(B * K, P, S)
    minutes = batch["minutes"].to(device).reshape(B * K, P)
    mask = batch["key_padding_mask"].to(device).reshape(B * K, P)

    model.train()
    model.zero_grad(set_to_none=True)
    score, _, attn_w = model(embs, stats, minutes, mask)
    loss = score.mean()
    if torch.isfinite(loss).all():
        loss.backward()
    grad_norm = 0.0
    for p in model.attn.parameters():
        if p.grad is not None:
            grad_norm += p.grad.data.norm(2).item()
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        attn_w = attn_w.reshape(B, K, P)
        all_masked = mask.reshape(B, K, P).all(dim=-1)
        valid_mask = ~mask
        minutes_valid = minutes[valid_mask]
        attn_sum = attn_w.sum(dim=-1).reshape(-1)
        attn_max = attn_w.max(dim=-1).values.reshape(-1)
        print(
            "Attention debug (train):",
            f"teams={attn_sum.numel()}",
            f"all_masked={int(all_masked.sum().item())}",
            f"minutes_min={float(minutes_valid.min().item()) if minutes_valid.numel() else 0.0:.4f}",
            f"minutes_mean={float(minutes_valid.mean().item()) if minutes_valid.numel() else 0.0:.4f}",
            f"minutes_max={float(minutes_valid.max().item()) if minutes_valid.numel() else 0.0:.4f}",
            f"attn_sum_mean={float(attn_sum.mean().item()) if attn_sum.numel() else 0.0:.4f}",
            f"attn_max_mean={float(attn_max.mean().item()) if attn_max.numel() else 0.0:.4f}",
            f"attn_grad_norm={grad_norm:.4f}",
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
) -> float:
    model.train()
    total = 0.0
    n = 0
    for batch in batches:
        B, K, P, S = batch["embedding_indices"].shape[0], batch["embedding_indices"].shape[1], batch["embedding_indices"].shape[2], batch["player_stats"].shape[-1]
        embs = batch["embedding_indices"].to(device).reshape(B * K, P)
        stats = batch["player_stats"].to(device).reshape(B * K, P, S)
        minutes = batch["minutes"].to(device).reshape(B * K, P)
        mask = batch["key_padding_mask"].to(device).reshape(B * K, P)
        rel = batch["rel"].to(device)

        score, _, _ = model(embs, stats, minutes, mask)
        score = score.reshape(B, K)
        score = torch.nan_to_num(score, nan=0.0, posinf=10.0, neginf=-10.0)
        loss = listmle_loss(score, rel)
        optimizer.zero_grad()
        if torch.isfinite(loss).all():
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
            n += 1
    return total / n if n else 0.0


def eval_epoch(
    model: nn.Module,
    batches: list[dict],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in batches:
            B, K, P, S = batch["embedding_indices"].shape[0], batch["embedding_indices"].shape[1], batch["embedding_indices"].shape[2], batch["player_stats"].shape[-1]
            embs = batch["embedding_indices"].to(device).reshape(B * K, P)
            stats = batch["player_stats"].to(device).reshape(B * K, P, S)
            minutes = batch["minutes"].to(device).reshape(B * K, P)
            mask = batch["key_padding_mask"].to(device).reshape(B * K, P)
            rel = batch["rel"].to(device)

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


def _build_model(config: dict, device: torch.device) -> nn.Module:
    ma = config.get("model_a", {})
    num_emb = ma.get("num_embeddings", 500)
    emb_dim = ma.get("embedding_dim", 32)
    stat_dim = 7
    enc_h = ma.get("encoder_hidden", [128, 64])
    heads = ma.get("attention_heads", 4)
    drop = ma.get("dropout", 0.2)
    return DeepSetRank(num_emb, emb_dim, stat_dim, enc_h, heads, drop).to(device)


def train_model_a_on_batches(
    config: dict,
    batches: list[dict],
    device: torch.device,
    max_epochs: int | None = None,
    *,
    val_batches: list[dict] | None = None,
) -> nn.Module:
    """Train Model A on given batches; return the model (do not save). For OOF fold training."""
    stat_dim = 7
    ma = config.get("model_a", {})
    num_emb = ma.get("num_embeddings", 500)
    if not batches:
        model = _build_model(config, device)
        return model
    model = _build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = int(max_epochs) if max_epochs is not None else int(ma.get("epochs", 20))
    patience = int(ma.get("early_stopping_patience", 3))
    min_delta = float(ma.get("early_stopping_min_delta", 0.0))
    use_early = bool(val_batches) and patience > 0
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(epochs):
        train_epoch(model, batches, optimizer, device)
        if use_early:
            val_loss = eval_epoch(model, val_batches or [], device)
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

    stat_dim = 7
    ma = config.get("model_a", {})
    num_emb = ma.get("num_embeddings", 500)
    model = _build_model(config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
    for epoch in range(epochs):
        loss = train_epoch(model, train_batches, optimizer, device)
        print(f"epoch {epoch+1} loss={loss:.4f}")
        if use_early:
            val_loss = eval_epoch(model, val_batches, device)
            print(f"val_loss {epoch+1}: {val_loss:.4f}")
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

    if batches:
        try:
            _log_attention_debug_stats(model, batches[0], device)
        except Exception:
            pass

    path = output_dir / "best_deep_set.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, path)
    return path
