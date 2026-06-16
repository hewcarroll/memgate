"""
RetrievalMemoryGate — gated cross-attention over a retrieved subset of memory.

A drop-in variant of :class:`MemoryGate` that, instead of attending over an
in-VRAM memory tensor passed to ``forward``, retrieves a small set of cells from
a pluggable :class:`~memgate.backend.MemoryBackend` (e.g. top-k from a large
external bank) and attends over those. Useful for kNN-LM / RETRO / memorizing-
transformer style augmentation where the full bank is too large to hold in VRAM.

The interface matches MemoryGate: ``forward(hidden_states, memory_states,
memory_mask) -> (output, attn_weights)``, but ``memory_states`` and
``memory_mask`` are ignored — retrieval is performed internally via the backend.

Copyright 2026 Hew Carroll
Licensed under MIT OR Apache-2.0
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backend import MemoryBackend


class RetrievalMemoryGate(nn.Module):
    """Gated cross-attention over cells retrieved from a pluggable backend.

    At selected decoder layers, the transformer's hidden states are used to
    retrieve relevant cells from the backend; those cells are attended to via
    standard cross-attention, then gated back into the hidden states.

    Interchangeable with :class:`MemoryGate` (same ``forward`` signature and
    return shape; ``memory_states``/``memory_mask`` are accepted but ignored).

    Args:
        d_model: Transformer hidden size (e.g., 2048).
        d_mem: Memory embedding dimension (e.g., 256).
        n_heads: Number of cross-attention heads.
        memory_backend: Any object implementing :class:`MemoryBackend`. If None,
            the layer is a pass-through.
        dropout: Dropout rate for attention weights.
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_mem: int = 256,
        n_heads: int = 4,
        memory_backend: Optional[MemoryBackend] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.head_dim = d_mem // n_heads
        self.backend = memory_backend

        assert d_mem % n_heads == 0, "d_mem must be divisible by n_heads"

        # Project transformer hidden states down to memory dimension
        self.proj_down = nn.Linear(d_model, d_mem)

        # Cross-attention projections
        self.q_proj = nn.Linear(d_mem, d_mem)
        self.k_proj = nn.Linear(d_mem, d_mem)
        self.v_proj = nn.Linear(d_mem, d_mem)
        self.attn_dropout = nn.Dropout(dropout)

        # Project memory context back to transformer dimension
        self.proj_up = nn.Linear(d_mem, d_model)

        # Gating mechanism
        self.gate_proj = nn.Linear(d_model * 2, d_model)

        # Layer norm for output
        self.layer_norm = nn.LayerNorm(d_model)

        # Initialize gate near zero for gradual memory integration
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)  # sigmoid(-2) ~ 0.12

        # Track last retrieval for strength updates
        self._last_retrieved_indices = []
        self._last_attn_weights = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply gated cross-attention over retrieved memory cells.

        ``memory_states`` and ``memory_mask`` are accepted for interface
        compatibility with :class:`MemoryGate` but ignored — retrieval is
        internal.

        Returns:
            output: [B, S, d_model] hidden states enriched with memory.
            attn_weights: [B, S, K] cross-attention weights for tracking.
        """
        if self.backend is None:
            # No backend configured -- pass through.
            dummy_attn = torch.zeros(
                hidden_states.shape[0], hidden_states.shape[1], 1,
                device=hidden_states.device,
            )
            return hidden_states, dummy_attn

        B, S, _ = hidden_states.shape
        device = hidden_states.device

        # Project to memory space for routing
        query_mem = self.proj_down(hidden_states)  # [B, S, d_mem]

        # Retrieve relevant cells from the backend
        with torch.no_grad():
            retrieved_embs, retrieved_indices, _ = self.backend.retrieve(query_mem)

        self._last_retrieved_indices = retrieved_indices
        K = retrieved_embs.shape[0]

        if K == 0:
            # No cells retrieved -- pass through.
            dummy_attn = torch.zeros(B, S, 1, device=device)
            return hidden_states, dummy_attn

        # Retrieved embeddings are treated as constants (cells are not updated by
        # the transformer's gradients).
        memory = retrieved_embs.detach()
        memory = memory.unsqueeze(0).expand(B, -1, -1)  # [B, K, d_mem]

        # Multi-head cross-attention
        Q = self.q_proj(query_mem).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K_proj = self.k_proj(memory).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        V_proj = self.v_proj(memory).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K_proj.transpose(-2, -1)) / scale  # [B, heads, S, K]

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, heads, S, K]
        attn_weights = self.attn_dropout(attn_weights)

        # Attention output
        attn_output = torch.matmul(attn_weights, V_proj)  # [B, heads, S, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_mem)

        # Project back to transformer dimension
        memory_context = self.proj_up(attn_output)  # [B, S, d_model]

        # Gated residual connection
        gate_input = torch.cat([hidden_states, memory_context], dim=-1)
        gate_value = torch.sigmoid(self.gate_proj(gate_input))
        output = hidden_states + gate_value * memory_context

        output = self.layer_norm(output)

        # Mean attention weights across heads for strength tracking
        mean_attn = attn_weights.mean(dim=1)  # [B, S, K]
        self._last_attn_weights = mean_attn.detach()

        return output, mean_attn

    def update_strengths(self, ema_decay: float = 0.99) -> None:
        """Push attention-based usage feedback for the last retrieved cells to
        the backend (no-op if the backend doesn't track strengths).

        Call after a forward pass (or after each batch).
        """
        if (
            self.backend is not None
            and self._last_attn_weights is not None
            and self._last_retrieved_indices
        ):
            self.backend.update_cell_strengths(
                self._last_attn_weights,
                self._last_retrieved_indices,
                ema_decay=ema_decay,
            )
