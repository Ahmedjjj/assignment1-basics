import base64
import logging
import os
from collections.abc import Iterable, Iterator

import tqdm

from cs336_basics._utils import encode, load_from_json, log_time, save_as_json
from cs336_basics.bpe._pre_tokens import (
    par_count_unique_ptokens,
    prepare_indices,
)
from cs336_basics.bpe._types import Merges, Token, TokenPair, Vocabulary

logger = logging.getLogger(__name__)


MAX_BYTE_VALUE = 256


def _init_vocab(special_tokens: Iterable[str] = ()) -> dict[int, Token]:
    vocab = {}
    for b in range(MAX_BYTE_VALUE):
        vocab[int(b)] = bytes([b])

    for i, token in enumerate(special_tokens):
        vocab[MAX_BYTE_VALUE + i] = encode(token)

    return vocab


def _get_next_merge_from_counts(counts: dict[TokenPair, int]) -> TokenPair:
    assert len(counts) > 0

    return max(counts, key=lambda x: (counts[x], x))


class BPETokenizer:
    _vocab: Vocabulary
    _special_tokens: tuple[str, ...]
    _trained: bool
    _merges: Merges

    def __init__(
        self,
        vocab: dict[int, Token] | None = None,
        merges: list[TokenPair] | None = None,
        special_tokens: list[str] | None = None,
    ) -> None:
        self._special_tokens = tuple(special_tokens or [])
        self._vocab = vocab or _init_vocab(self._special_tokens)
        self._merges = merges or []
        self._trained = False

    @property
    def vocab(self) -> dict[int, Token]:
        return dict(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def merges(self) -> tuple[TokenPair, ...]:
        return tuple(self._merges)

    @log_time("Finding next merge", logger=logger)
    def _get_next_merge(self) -> TokenPair:
        return self._ptokens_for_pair.most_freq_pair()
        # counts = {}
        # for pair, tokens in self._ptokens_for_pair.items():
        #     counts[pair] = sum(self._counts[t] * ptoken_count for t, ptoken_count in tokens.items())

        # return _get_next_merge_from_counts(counts)

    def _update_ptoken_pairs(self, ptoken: int, pairs: list[TokenPair]) -> None:
        self._pairs_for_ptoken[ptoken] = pairs

        for pair in pairs:
            if pair in self._ptokens_for_pair:
                self._ptokens_for_pair.inc(pair, ptoken)
            else:
                self._ptokens_for_pair.inc(pair, ptoken)

    @log_time("Merging token pair", logger=logger)
    def _merge(self, pair: TokenPair) -> None:
        # Add to vocab
        new_token = pair[0] + pair[1]
        self._vocab[self.vocab_size] = new_token

        for ptoken in self._ptokens_for_pair[pair]:
            ptoken_pairs = self._pairs_for_ptoken[ptoken]

            ptoken_new_pairs = []
            for i, p in enumerate(ptoken_pairs):
                if p == pair:
                    continue
                elif i > 0 and ptoken_pairs[i - 1] == pair:
                    self._ptokens_for_pair.dec(pair=p, ptoken=ptoken)
                    ptoken_new_pairs.append((new_token, p[1]))
                elif i < len(ptoken_pairs) - 1 and ptoken_pairs[i + 1] == pair:
                    self._ptokens_for_pair.dec(pair=p, ptoken=ptoken)
                    ptoken_new_pairs.append((p[0], new_token))
                else:
                    self._ptokens_for_pair.dec(pair=p, ptoken=ptoken)
                    ptoken_new_pairs.append(p)

            self._update_ptoken_pairs(ptoken, ptoken_new_pairs)

        # Remove old pair
        self._ptokens_for_pair.pop(pair)

    @log_time("Merge Step", logger=logger)
    def _run_merge(self) -> None:
        merge = self._get_next_merge()
        self._merges.append(merge)
        self._merge(merge)

    @log_time("Running merges", logger=logger)
    def _run_merges(self, num_merges) -> None:
        for _ in tqdm.tqdm(range(num_merges)):
            self._run_merge()

    @log_time("Training", logger=logger)
    def train(self, texts: Iterable[str], vocab_size: int, ptoken_count_n_jobs: int = 1) -> None:
        if self._trained:
            raise ValueError("Called `train` on an already trained tokenizer")

        num_merges = vocab_size - self.vocab_size

        if num_merges <= 0:
            return

        self._trained = True

        counts = par_count_unique_ptokens(texts=texts, special_tokens=self._special_tokens, n_jobs=ptoken_count_n_jobs)
        self._counts, self._pairs_for_ptoken, self._ptokens_for_pair = prepare_indices(counts)
        self._merges = []

        self._run_merges(num_merges)

    def encode(self, text: str) -> list[int]: ...

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]: ...

    def decode(self, ids: list[int]) -> str: ...

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


def _decode_from_b64(text: str) -> bytes:
    return base64.b64decode(text)


def _encode_as_b64(bytes: bytes) -> str:
    return base64.b64encode(bytes).decode("utf-8")
