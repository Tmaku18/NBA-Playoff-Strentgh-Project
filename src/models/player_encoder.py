"""Shared MLP for player stat vector."""

from __future__ import annotations

import torch
import torch.nn as nn


class PlayerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], dropout: float = 0.2):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.mlp = nn.Sequential(*layers)
        self.output_dim = d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, S)
        return self.mlp(x)
