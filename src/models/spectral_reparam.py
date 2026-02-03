"""Spectral reparameterization (σReparam) for attention stability.

Reference: Zhai et al., "Stabilizing Transformer Training by Preventing Attention
Entropy Collapse", arXiv:2303.06296. We reparameterize W as W_hat = (gamma / sigma(W)) * W
with one power-iteration estimate of sigma(W) per forward. This bounds the spectral
norm of attention logits and helps prevent entropy collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralReparamLinear(nn.Module):
    """Linear layer with σReparam: effective weight = (gamma / sigma(W)) * W.
    sigma(W) is approximated by one power iteration per forward (training only).
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.gamma = nn.Parameter(torch.ones(1))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self._init_weights()
        self._register_buffers()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _register_buffers(self) -> None:
        # u: left singular vector (out_features,), v: right singular vector (in_features,)
        u = torch.randn(self.out_features)
        u = u / (u.norm() + 1e-12)
        v = torch.randn(self.in_features)
        v = v / (v.norm() + 1e-12)
        self.register_buffer("_u", u)
        self.register_buffer("_v", v)

    def _power_iteration(self, eps: float = 1e-12) -> torch.Tensor:
        """One step of power iteration in no_grad; returns sigma. Buffers updated in no_grad so no graph."""
        with torch.no_grad():
            u = F.linear(self._v.unsqueeze(0), self.weight).squeeze(0)
            u = u / (u.norm() + eps)
            v = F.linear(u.unsqueeze(0), self.weight.t()).squeeze(0)
            v = v / (v.norm() + eps)
            sigma = (u.unsqueeze(0) @ self.weight @ v.unsqueeze(1)).squeeze().clamp(min=eps)
            self._u.copy_(u)
            self._v.copy_(v)
        return sigma.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.weight.requires_grad:
            sigma = self._power_iteration()
            effective_weight = (self.gamma / sigma) * self.weight
        else:
            with torch.no_grad():
                sigma = (
                    (self._u.unsqueeze(0) @ self.weight @ self._v.unsqueeze(1))
                    .squeeze()
                    .clamp(min=1e-12)
                )
            effective_weight = (self.gamma / sigma) * self.weight
        return F.linear(x, effective_weight, self.bias)
