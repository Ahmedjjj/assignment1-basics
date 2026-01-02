import base64
import logging
import os
from collections.abc import Iterable, Iterator

from cs336_basics._utils import encode, load_from_json, log_time, save_as_json
from cs336_basics.bpe._pre_tokens import PToken, TokenPairFactory, find_pre_tokens, split_on_special_tokens
from cs336_basics.bpe._types import Merges, Token
from cs336_basics.bpe._vocab import Vocabulary

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

        self._merges = {m: i for i, m in enumerate(merges)}

    @property
    def vocab(self) -> dict[int, Token]:
        return self._vocab

    @property
    def merges(self) -> Merges:
        return list(self._merges.keys())

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    @log_time("Encoding token", logger=logger)
    def _encode_token(self, token: str) -> list[int]:
        pair_factory = TokenPairFactory()
        ptoken = PToken(text=token, pair_factory=pair_factory)

        next_merge = min(
            (pair for pair in ptoken.pairs if (pair.left, pair.right) in self._merges),
            key=lambda x: self._merges[(x.left, x.right)],
            default=None,
        )

        while next_merge is not None:
            ptoken.merge(next_merge)
            next_merge = min(
                (pair for pair in ptoken.pairs if (pair.left, pair.right) in self._merges),
                key=lambda x: self._merges[(x.left, x.right)],
                default=None,
            )

        return [self._vocab.reversed[t] for t in ptoken.tokens]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        cache = {}
        for chunk in iterable:
            if self._special_tokens:
                splitted_chunks = split_on_special_tokens(
                    chunk, special_tokens=self._special_tokens, return_special_tokens=True
                )
            else:
                splitted_chunks = [chunk]

            for chunk in splitted_chunks:
                if self._special_tokens and chunk in self._special_tokens:
                    yield self._vocab.reversed[encode(chunk)]
                else:
                    for ptoken in find_pre_tokens(chunk):
                        if ptoken in cache:
                            yield from cache[ptoken]
                        else:
                            cache[ptoken] = list(self._encode_token(ptoken))
                            yield from cache[ptoken]

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
        vocab = {int(k): _decode_from_b64(v) for k, v in vocab.items()}

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
