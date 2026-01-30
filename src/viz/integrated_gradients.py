"""Captum Integrated Gradients for Model A. Baselines; additional_forward_args for minutes, mask."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from captum.attr import IntegratedGradients
    _HAS_CAPTUM = True
except ImportError:
    _HAS_CAPTUM = False


def ig_attr(
    model: torch.nn.Module,
    emb_indices: torch.Tensor,
    player_stats: torch.Tensor,
    minutes: torch.Tensor,
    key_padding_mask: torch.Tensor,
    *,
    n_steps: int = 100,
    baselines: tuple[torch.Tensor, torch.Tensor] | None = None,
    target: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    IG on player_stats (continuous). emb_indices, minutes, mask as additional_forward_args.
    Returns (attributions on player_stats, convergence_delta if return_convergence_delta).
    """
    if not _HAS_CAPTUM:
        raise ImportError("captum is required")

    def fn(stats: torch.Tensor) -> torch.Tensor:
        o, _, _ = model(emb_indices, stats, minutes, key_padding_mask)
        return o if target is None else o[..., target]

    ig = IntegratedGradients(fn)
    if baselines is None:
        baselines = (player_stats * 0,)
    attr = ig.attribute(
        player_stats,
        baselines=baselines[0] if isinstance(baselines[0], torch.Tensor) else player_stats * 0,
        additional_forward_args=(),
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    if isinstance(attr, tuple):
        return attr[0], attr[1]
    return attr, None


def attention_ablation(
    model: torch.nn.Module,
    emb: torch.Tensor,
    stats: torch.Tensor,
    minutes: torch.Tensor,
    mask: torch.Tensor,
    attn_weights: torch.Tensor,
    *,
    top_k: int = 3,
    metric_fn: Any = None,
) -> float:
    """Mask top-attention players (set stats to 0 and mask=True), run model, return metric. Skips padded positions (mask=True). Returns nan if forward yields non-finite scores."""
    B, P, S = stats.shape
    stats_masked = stats.clone()
    mask_masked = mask.clone()
    for b in range(B):
        # Only consider non-padded positions (mask[b, i] is False)
        valid = (mask[b] == False).nonzero(as_tuple=True)[0]
        if valid.numel() == 0:
            continue
        k_use = min(top_k, valid.numel())
        attn_b = attn_weights[b]
        attn_valid = attn_b[valid]
        _, order = attn_valid.topk(k_use, largest=True)
        top_indices = valid[order]
        for idx in top_indices.tolist():
            if idx < P:
                stats_masked[b, idx, :] = 0
                mask_masked[b, idx] = True
    with torch.no_grad():
        s_ablate, _, _ = model(emb, stats_masked, minutes, mask_masked)
    if not torch.isfinite(s_ablate).all():
        return float("nan")
    if metric_fn is not None:
        return float(metric_fn(s_ablate))
    return float(s_ablate.mean().item())
