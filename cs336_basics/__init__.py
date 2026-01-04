import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from cs336_basics._utils import configure_logging, read_in_chunks
from cs336_basics.bpe import BPETokenizer, train_bpe
from cs336_basics.layers import Embedding, Linear, RmsNorm, RoPE, SwiGLU
from cs336_basics.pretokenization_example import find_chunk_boundaries

__all__ = [
    "configure_logging",
    "BPETokenizer",
    "train_bpe",
    "find_chunk_boundaries",
    "read_in_chunks",
    "Linear",
    "Embedding",
    "RmsNorm",
    "SwiGLU",
    "RoPE",
]
