import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

from cs336_basics._utils import configure_logging, read_in_chunks
from cs336_basics.bpe import BPETokenizer, train_bpe
from cs336_basics.pretokenization_example import find_chunk_boundaries

__all__ = ["configure_logging", "BPETokenizer", "train_bpe", "find_chunk_boundaries", "read_in_chunks"]
