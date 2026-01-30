"""Set attention: MultiheadAttention with batch_first=True and key_padding_mask. Minutes-weighting."""

from __future__ import annotations

import torch
import torch.nn as nn


class SetAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        minutes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, P, D). key_padding_mask: (B, P) bool, True = ignore.
        minutes: (B, P) optional weights. Returns (out, attn_weights).
        """
        # query from a single learned vector or mean; use masked mean of x as query for set-pooling
        if key_padding_mask is not None and key_padding_mask.shape[:2] == x.shape[:2]:
            valid = (~key_padding_mask).unsqueeze(-1).float()
            denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            q = (x * valid).sum(dim=1, keepdim=True) / denom
        else:
            q = x.mean(dim=1, keepdim=True)  # (B, 1, D)
        out, w = self.attn(q, x, x, key_padding_mask=key_padding_mask, need_weights=True)
        # out (B, 1, D), w (B, 1, P)
        if minutes is not None and w.shape[-1] == minutes.shape[-1]:
            mins = minutes
            if key_padding_mask is not None and key_padding_mask.shape == minutes.shape:
                mins = mins.masked_fill(key_padding_mask, 0.0)
            mins = mins.clamp(min=0.0)
            mins = mins / (mins.sum(dim=-1, keepdim=True).clamp(min=1e-8))
            w = w * (0.5 + 0.5 * mins.unsqueeze(1))
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
        return out.squeeze(1), w.squeeze(1)
