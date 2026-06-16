"""memgate — gated cross-attention memory layers for transformers."""

from .backend import InMemoryBackend, MemoryBackend
from .gate import MemoryGate
from .retrieval_gate import RetrievalMemoryGate

__version__ = "0.1.0"

__all__ = [
    "MemoryGate",
    "RetrievalMemoryGate",
    "MemoryBackend",
    "InMemoryBackend",
    "__version__",
]
