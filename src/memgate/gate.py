"""
MemoryGate — gated cross-attention between transformer hidden states and an
external memory bank.

Projects transformer hidden states into memory space, performs cross-attention
to query the memory bank, then projects back and applies a learned gate to
control how much memory context influences the output.

The gate is initialized near zero so memory integration is gradual during early
training, preventing the memory from destabilizing the base model.

Copyright 2026 Hew Carroll
Licensed under MIT OR Apache-2.0
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryGate(nn.Module):
    """Gated cross-attention between transformer hidden states and a memory bank.

    At selected decoder layers, the transformer's hidden states query the memory
    bank via cross-attention. A learned gate controls how much memory context
    feeds back into the transformer.

    Args:
        d_model: Transformer hidden size (e.g., 2048).
        d_mem: Memory embedding dimension (e.g., 256).
        n_heads: Number of cross-attention heads.
        dropout: Dropout rate for attention weights.
    """

    def __init__(
        self,
        d_model: int = 2048,
        d_mem: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_heads = n_heads
        self.head_dim = d_mem // n_heads

        assert d_mem % n_heads == 0, "d_mem must be divisible by n_heads"

        # Project transformer hidden states down to memory dimension
        self.proj_down = nn.Linear(d_model, d_mem)

        # Cross-attention projections (query from transformer, key/value from memory)
        self.q_proj = nn.Linear(d_mem, d_mem)
        self.k_proj = nn.Linear(d_mem, d_mem)
        self.v_proj = nn.Linear(d_mem, d_mem)
        self.attn_dropout = nn.Dropout(dropout)

        # Project memory context back to transformer dimension
        self.proj_up = nn.Linear(d_mem, d_model)

        # Gating mechanism: controls how much memory influences the output.
        # Input is the concatenation of hidden_states and memory_context.
        self.gate_proj = nn.Linear(d_model * 2, d_model)

        # Layer norm for output
        self.layer_norm = nn.LayerNorm(d_model)

        # Initialize gate near zero for gradual memory integration
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)  # sigmoid(-2) ~ 0.12

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply gated cross-attention to query the memory bank.

        Args:
            hidden_states: [B, S, d_model] transformer hidden states.
            memory_states: [1, M, d_mem] memory bank representations
                (M = number of active memory cells).
            memory_mask: [1, M] True for pruned/inactive memory cells.

        Returns:
            output: [B, S, d_model] hidden states enriched with memory.
            attn_weights: [B, S, M] cross-attention weights for strength tracking.
        """
        B, S, _ = hidden_states.shape
        M = memory_states.shape[1]

        # Expand memory to batch size
        memory = memory_states.expand(B, -1, -1)  # [B, M, d_mem]

        # Project hidden states to memory space
        query = self.proj_down(hidden_states)  # [B, S, d_mem]

        # Multi-head cross-attention
        Q = self.q_proj(query).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(memory).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(memory).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        # Q: [B, n_heads, S, head_dim], K/V: [B, n_heads, M, head_dim]

        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, n_heads, S, M]

        # Apply memory mask (mask out pruned cells)
        if memory_mask is not None:
            mask = memory_mask.unsqueeze(1).unsqueeze(2)  # [1, 1, 1, M]
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, n_heads, S, M]
        attn_weights = self.attn_dropout(attn_weights)

        # Compute attention output
        attn_output = torch.matmul(attn_weights, V)  # [B, n_heads, S, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, self.d_mem)

        # Project back to transformer dimension
        memory_context = self.proj_up(attn_output)  # [B, S, d_model]

        # Gated residual connection
        gate_input = torch.cat([hidden_states, memory_context], dim=-1)  # [B, S, 2*d_model]
        gate_value = torch.sigmoid(self.gate_proj(gate_input))  # [B, S, d_model]
        output = hidden_states + gate_value * memory_context

        output = self.layer_norm(output)

        # Mean attention weights across heads for strength tracking
        mean_attn = attn_weights.mean(dim=1)  # [B, S, M]

        return output, mean_attn
