from collections.abc import Iterable

from cs336_basics._utils import encode
from cs336_basics.bpe._types import Token

_MAX_BYTE_VALUE = 256


class Vocabulary(dict[int, Token]):
    def __init__(self, dict: dict[int, Token] | None = None, **kwargs) -> None:
        super().__init__()

        self._reverse_vocab = {}
        if dict is not None:
            for k, v in dict.items():
                self[k] = v

    def __setitem__(self, key: int, item: Token) -> None:
        super().__setitem__(key, item)
        self._reverse_vocab[item] = key

    @property
    def reversed(self) -> dict[Token, int]:
        return self._reverse_vocab


def init_vocab(special_tokens: Iterable[str] = ()) -> Vocabulary:
    vocab = {}
    for b in range(_MAX_BYTE_VALUE):
        vocab[int(b)] = bytes([b])

    for i, token in enumerate(special_tokens):
        vocab[_MAX_BYTE_VALUE + i] = encode(token)

    return Vocabulary(vocab)
