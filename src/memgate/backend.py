"""Pluggable memory backends for :class:`memgate.RetrievalMemoryGate`.

A backend supplies the gate with memory cells to attend over and (optionally)
records usage feedback. Any object implementing the :class:`MemoryBackend`
protocol works — the bundled :class:`InMemoryBackend`, a FAISS index, a
disk-backed store, etc.

Copyright 2026 Hew Carroll
Licensed under MIT OR Apache-2.0
"""

from __future__ import annotations

from typing import List, Protocol, Tuple, runtime_checkable

import torch
import torch.nn.functional as F


@runtime_checkable
class MemoryBackend(Protocol):
    """Interface a :class:`RetrievalMemoryGate` uses to fetch/score memory cells."""

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        """Return ``(embeddings [K, d_mem], indices, scores [K])`` for a query.

        Args:
            query: ``[B, S, d_mem]`` projected hidden states. The implementation
                decides how to pool these into a routing signal.
        """
        ...

    def update_cell_strengths(
        self, attn_weights: torch.Tensor, indices: List[int], ema_decay: float = 0.99
    ) -> None:
        """Record usage feedback for the retrieved cells. May be a no-op."""
        ...


class InMemoryBackend:
    """Reference backend: top-k cosine retrieval over a fixed ``[N, d_mem]`` bank.

    Returns the ``top_k`` cells most similar to the mean-pooled query and tracks
    an EMA "strength" per cell from attention feedback. Good as a default and for
    tests; swap in a FAISS/disk implementation for scale.

    Args:
        embeddings: ``[N, d_mem]`` cell embeddings.
        top_k: maximum number of cells to return per query.
    """

    def __init__(self, embeddings: torch.Tensor, top_k: int = 64):
        if embeddings.dim() != 2:
            raise ValueError("embeddings must be a 2-D tensor of shape [N, d_mem]")
        self.embeddings = embeddings
        self._normed = F.normalize(embeddings, dim=-1)
        self.top_k = top_k
        self.strengths = torch.ones(embeddings.shape[0])

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        n, d = self.embeddings.shape
        if n == 0:
            empty = self.embeddings.new_zeros(0, d).to(query.device)
            return empty, [], query.new_zeros(0)
        # Mean-pool the query [B, S, d_mem] into one routing vector.
        q = query.reshape(-1, query.shape[-1]).mean(dim=0)
        q = F.normalize(q, dim=-1).to(self._normed.device)
        sims = self._normed @ q  # [N]
        k = min(self.top_k, n)
        scores, idx = torch.topk(sims, k)
        embs = self.embeddings.index_select(0, idx).to(query.device)
        return embs, idx.tolist(), scores.to(query.device)

    def update_cell_strengths(
        self, attn_weights: torch.Tensor, indices: List[int], ema_decay: float = 0.99
    ) -> None:
        if not indices:
            return
        # attn_weights [B, S, K] -> per-cell mean importance [K]
        per_cell = attn_weights.reshape(-1, attn_weights.shape[-1]).mean(dim=0).detach()
        per_cell = per_cell.to(self.strengths.device)
        for i, cell in enumerate(indices):
            if i < per_cell.numel():
                self.strengths[cell] = (
                    ema_decay * self.strengths[cell] + (1.0 - ema_decay) * per_cell[i]
                )
