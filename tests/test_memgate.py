import torch

from memgate import InMemoryBackend, MemoryBackend, MemoryGate, RetrievalMemoryGate

B, S, D_MODEL, D_MEM, N_HEADS = 2, 3, 16, 8, 2


def test_memory_gate_shapes_and_finite():
    gate = MemoryGate(d_model=D_MODEL, d_mem=D_MEM, n_heads=N_HEADS)
    gate.eval()  # disable attention dropout so weights sum to 1
    hidden = torch.randn(B, S, D_MODEL)
    memory = torch.randn(1, 5, D_MEM)
    out, attn = gate(hidden, memory)
    assert out.shape == (B, S, D_MODEL)
    assert attn.shape == (B, S, 5)
    assert torch.isfinite(out).all()
    # softmax rows over the M axis sum to 1 (no dropout in eval mode)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(B, S), atol=1e-4)


def test_memory_mask_excludes_cells():
    gate = MemoryGate(d_model=D_MODEL, d_mem=D_MEM, n_heads=N_HEADS, dropout=0.0)
    hidden = torch.randn(B, S, D_MODEL)
    memory = torch.randn(1, 4, D_MEM)
    mask = torch.tensor([[False, False, True, True]])  # last two pruned
    _, attn = gate(hidden, memory, memory_mask=mask)
    assert torch.allclose(attn[..., 2:], torch.zeros_like(attn[..., 2:]), atol=1e-6)


def test_retrieval_gate_passthrough_without_backend():
    gate = RetrievalMemoryGate(d_model=D_MODEL, d_mem=D_MEM, n_heads=N_HEADS)
    hidden = torch.randn(B, S, D_MODEL)
    out, attn = gate(hidden)
    assert torch.equal(out, hidden)          # pure pass-through
    assert attn.shape == (B, S, 1)


def test_retrieval_gate_with_in_memory_backend():
    torch.manual_seed(0)
    bank = torch.randn(10, D_MEM)
    backend = InMemoryBackend(bank, top_k=4)
    gate = RetrievalMemoryGate(
        d_model=D_MODEL, d_mem=D_MEM, n_heads=N_HEADS, memory_backend=backend
    )
    hidden = torch.randn(B, S, D_MODEL)
    out, attn = gate(hidden)
    assert out.shape == (B, S, D_MODEL)
    assert attn.shape == (B, S, 4)           # K == top_k

    before = backend.strengths.clone()
    gate.update_strengths()
    assert not torch.equal(before, backend.strengths)  # feedback recorded


def test_in_memory_backend_retrieve_and_empty():
    bank = torch.randn(6, D_MEM)
    backend = InMemoryBackend(bank, top_k=3)
    q = torch.randn(B, S, D_MEM)
    embs, idx, scores = backend.retrieve(q)
    assert embs.shape == (3, D_MEM)
    assert len(idx) == 3 and scores.shape == (3,)

    empty = InMemoryBackend(torch.zeros(0, D_MEM), top_k=3)
    e_embs, e_idx, _ = empty.retrieve(q)
    assert e_embs.shape[0] == 0 and e_idx == []


def test_in_memory_backend_satisfies_protocol():
    backend = InMemoryBackend(torch.randn(3, D_MEM))
    assert isinstance(backend, MemoryBackend)  # runtime_checkable Protocol
