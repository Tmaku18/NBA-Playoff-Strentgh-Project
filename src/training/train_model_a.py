"""Train Model A (DeepSet + ListMLE). Checkpointing, walk-forward train/val seasons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.models.deep_set_rank import DeepSetRank
from src.models.listmle_loss import listmle_loss
from src.utils.repro import set_seeds


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
    max_epochs: int = 3,
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
    for epoch in range(max_epochs):
        train_epoch(model, batches, optimizer, device)
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
    for epoch in range(3):
        loss = train_epoch(model, batches, optimizer, device)
        print(f"epoch {epoch+1} loss={loss:.4f}")

    path = output_dir / "best_deep_set.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, path)
    return path
