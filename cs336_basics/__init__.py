import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from cs336_basics._utils import configure_logging
from cs336_basics.bpe import BPETokenizer

__all__ = ["configure_logging", "BPETokenizer"]
