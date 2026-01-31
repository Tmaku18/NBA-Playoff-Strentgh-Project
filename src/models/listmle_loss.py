"""Numerically stable ListMLE using torch.logsumexp / logcumsumexp."""

from __future__ import annotations

import torch


def listmle_loss(scores: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
    """
    scores: (B, L) model scores. rel: (B, L) relevance/rank (higher = better). List is ordered by rel descending.
    Loss = -log P(permutation | scores). Stable: use logsumexp. P = prod_i exp(s_i) / sum_{j in remain} exp(s_j).
    -log P = sum_i [ log(sum_{j>=i} exp(s_j)) - s_i ].
    For stability: subtract max_s; then logsumexp = log(sum exp(s-m)) + m. Guards against NaN/inf.
    """
    B, L = scores.shape
    if L <= 1:
        return scores.new_zeros(B).mean()

    # Guard against NaN/inf so logsumexp and gradients stay finite
    scores = torch.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=-50.0)
    scores = scores.clamp(-50.0, 50.0)

    # Order indices by rel descending (best first)
    _, order = rel.sort(dim=1, descending=True)
    s = torch.gather(scores, 1, order)  # (B, L)

    # Per-row max for numerical stability (logsumexp)
    max_s = s.max(dim=1, keepdim=True)[0]
    s_stable = s - max_s
    log_denom = torch.stack(
        [torch.logsumexp(s_stable[:, i:], dim=1) for i in range(L)],
        dim=1,
    )
    nll = (log_denom - s).sum(dim=1)
    out = nll.mean()
    return torch.nan_to_num(out, nan=0.0, posinf=1e2, neginf=1e2)
