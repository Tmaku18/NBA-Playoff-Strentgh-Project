"""Player embedding table + hash-trick for unseen players."""

from __future__ import annotations

import torch
import torch.nn as nn


def hash_trick_index(player_id: int | str, num_embeddings: int) -> int:
    return hash(str(player_id)) % num_embeddings


class PlayerEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # indices 0..num_embeddings-1 = hash; num_embeddings = padding
        if padding_idx is None:
            padding_idx = num_embeddings
        self.emb = nn.Embedding(num_embeddings + 1, embedding_dim, padding_idx=padding_idx)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # indices: (B, P) int; num_embeddings = padding
        return self.emb(indices.clamp(0, self.num_embeddings))
