import base64
import logging
import os
from collections.abc import Iterable, Iterator

from cs336_basics._utils import load_from_json, save_as_json
from cs336_basics.bpe._types import Merges, Token
from cs336_basics.bpe._vocab import Vocabulary
from cs336_basics.bpe.tokenize import tokenize

logger = logging.getLogger(__name__)


class BPETokenizer:
    def __init__(
        self,
        vocab: Vocabulary | dict[int, Token],
        merges: Merges,
        special_tokens: list[str] | None = None,
    ) -> None:
        self._special_tokens = special_tokens

        if not isinstance(vocab, Vocabulary):
            self._vocab = Vocabulary(vocab)
        else:
            self._vocab = vocab

        self._merges = merges

    @property
    def vocab(self) -> dict[int, Token]:
        return self._vocab

    @property
    def merges(self) -> Merges:
        return self._merges

    def encode(self, text: str) -> list[int]:
        return tokenize(text, vocab=self.vocab, merges=self.merges, special_tokens=self._special_tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        res = b""
        for id in ids:
            res += self.vocab[id]

        return res.decode("utf-8", errors="replace")

    def save_to_files(
        self,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens_path: str | None = None,
    ) -> None:
        ser_vocab = {k: _encode_as_b64(v) for k, v in self._vocab.items()}
        ser_merges = [(_encode_as_b64(p[0]), _encode_as_b64(p[1])) for p in self._merges]

        save_as_json(ser_vocab, vocab_filepath)
        save_as_json(ser_merges, merges_filepath)

        if special_tokens_path is not None:
            save_as_json(self._special_tokens, special_tokens_path)

    def save_to_folder(self, folder_path: str) -> None:
        self.save_to_files(
            vocab_filepath=os.path.join(folder_path, "vocab.json"),
            merges_filepath=os.path.join(folder_path, "merges.json"),
            special_tokens_path=os.path.join(folder_path, "special.json"),
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | str | None = None,
    ) -> "BPETokenizer":
        vocab = load_from_json(vocab_filepath)
        vocab = {k: _decode_from_b64(v) for k, v in vocab.items()}

        merges = load_from_json(merges_filepath)
        merges = [(_decode_from_b64(p[0]), _decode_from_b64(p[1])) for p in merges]

        if isinstance(special_tokens, str):
            special_tokens = load_from_json(special_tokens)

        assert special_tokens is None or isinstance(special_tokens, list)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @classmethod
    def from_folder(cls, folder_path: str) -> "BPETokenizer":
        return cls.from_files(
            vocab_filepath=os.path.join(folder_path, "vocab.json"),
            merges_filepath=os.path.join(folder_path, "merges.json"),
            special_tokens=os.path.join(folder_path, "special.json"),
        )


def _decode_from_b64(text: str) -> bytes:
    return base64.b64decode(text)


def _encode_as_b64(bytes: bytes) -> str:
    return base64.b64encode(bytes).decode("utf-8")
