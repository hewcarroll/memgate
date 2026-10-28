# memgate

Gated cross-attention **memory layers** for transformers — the kind used in
kNN-LM / RETRO / memorizing-transformer style augmentation. Drop a gate into
selected decoder layers and let hidden states attend over an external memory
bank, with a learned gate (initialized near zero) controlling how much memory
context feeds back in.

Two layers, one interface:

- **`MemoryGate`** — attends over an in-VRAM memory tensor you pass to `forward`.
- **`RetrievalMemoryGate`** — retrieves a top-k subset of cells from a pluggable
  **`MemoryBackend`** and attends over those (for banks too large to hold in VRAM).

> Scope note: this is a mechanism (layers + a backend protocol), not a trained
> model or a benchmark claim. Wire it into your own architecture and training.

## Install

```sh
pip install git+https://github.com/hewcarroll/memgate
```

Install from this repository, not PyPI: the `memgate` package on PyPI is an
unrelated project.

## Usage

### In-VRAM memory

```python
import torch
from memgate import MemoryGate

gate = MemoryGate(d_model=2048, d_mem=256, n_heads=4)
hidden = torch.randn(2, 128, 2048)        # [B, S, d_model]
memory = torch.randn(1, 4096, 256)        # [1, M, d_mem] memory bank
out, attn = gate(hidden, memory)          # out: [B, S, d_model], attn: [B, S, M]
```

### Retrieval from a backend

```python
import torch
from memgate import RetrievalMemoryGate, InMemoryBackend

bank = torch.randn(100_000, 256)                       # your [N, d_mem] cells
backend = InMemoryBackend(bank, top_k=64)              # top-k cosine retrieval
gate = RetrievalMemoryGate(d_model=2048, d_mem=256, n_heads=4, memory_backend=backend)

out, attn = gate(hidden)                               # retrieval is internal
gate.update_strengths()                                # optional usage feedback
```

`RetrievalMemoryGate` shares `MemoryGate`'s signature, so the two are
interchangeable (its `memory_states`/`memory_mask` args are accepted but ignored).
With no backend it's a pass-through.

### Custom backend

`MemoryBackend` is a small `typing.Protocol` — implement two methods to back the
gate with FAISS, a disk store, a vector DB, etc.:

```python
from typing import List, Tuple
import torch

class MyBackend:
    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, List[int], torch.Tensor]:
        ...  # return (embeddings [K, d_mem], indices, scores [K])
    def update_cell_strengths(self, attn_weights, indices, ema_decay=0.99) -> None:
        ...  # optional; may be a no-op
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
