import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import NamedTuple

import regex as re
import sortedcontainers
from joblib import Parallel, delayed

from cs336_basics._utils import encode, log_time
from cs336_basics.bpe._types import PToken, Token, TokenPair

logger = logging.getLogger(__name__)


PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_pre_tokens(texts: Iterable[str]) -> Iterable[str]:
    for text in texts:
        for m in PAT.finditer(text):
            yield m.group(0)


def split_on_special_tokens(special_tokens: Iterable[str], texts: Iterable[str]) -> Iterable[str]:
    split_re = re.compile("|".join(re.escape(token) for token in special_tokens))

    for text in texts:
        yield from split_re.split(text)


@log_time("Executing parallel counts")
def _par_count(texts: Iterable[str], special_tokens: tuple[str, ...], n_jobs: int) -> Iterable[Counter[str]]:
    return Parallel(n_jobs=n_jobs)(
        delayed(count_unique_ptokens)(text=text, special_tokens=special_tokens) for text in texts
    )  #  type: ignore (https://github.com/joblib/joblib/issues/1176)


@log_time("Aggregating counts from sub-processes", logger=logger)
def _sum_counters(*counters: Counter[str]) -> Counter[str]:
    res = Counter[str]()

    for counter in counters:
        res += counter
    return res


@log_time("Counting unique ptokens in parallel", logger=logger)
def par_count_unique_ptokens(texts: Iterable[str], special_tokens: tuple[str, ...], n_jobs: int) -> dict[str, int]:
    return _sum_counters(*_par_count(texts=texts, special_tokens=special_tokens, n_jobs=n_jobs))


def count_unique_ptokens(text: str, special_tokens: tuple[str, ...]) -> Counter[str]:
    texts = split_on_special_tokens(texts=[text], special_tokens=special_tokens)
    ptokens = find_pre_tokens(texts)
    return Counter(ptokens)


def get_ptoken_pairs(tokens: Iterator[Token]) -> Iterator[TokenPair]:
    try:
        prev_token = next(tokens)
    except StopIteration:
        return

    for token in tokens:
        yield (prev_token, token)
        prev_token = token


class _PTokenIndex(dict[TokenPair, dict[PToken, int]]):
    def __init__(self, ptoken_counts: dict[PToken, int]):
        self._ptoken_counts = ptoken_counts
        self._sorted_pairs = sortedcontainers.SortedKeyList(key=self._pair_key)
        self._pair_total_count = {}

    def _pair_key(self, pair: TokenPair):
        return (self._pair_total_count[pair], pair)

    def add_count(self, pair: TokenPair, ptoken: PToken, count: int = 1) -> None:
        if pair not in self:
            self[pair] = {}

        pair_counts = self[pair]
        if ptoken not in pair_counts:
            pair_counts[ptoken] = 0

        new_ptoken_count = pair_counts[ptoken] + count

        if new_ptoken_count < 0:
            raise ValueError("Updating count for PToken would results in a negative value")
        elif new_ptoken_count == 0:
            pair_counts.pop(ptoken)
        else:
            pair_counts[ptoken] = new_ptoken_count

        if pair not in self._pair_total_count:
            self._pair_total_count[pair] = count * self._ptoken_counts[ptoken]
            self._sorted_pairs.add(pair)
        else:
            self._sorted_pairs.remove(pair)
            self._pair_total_count[pair] += count * self._ptoken_counts[ptoken]
            self._sorted_pairs.add(pair)

    def inc(self, pair: TokenPair, ptoken: PToken) -> None:
        self.add_count(pair=pair, ptoken=ptoken, count=1)

    def dec(self, pair: TokenPair, ptoken: PToken) -> None:
        self.add_count(pair=pair, ptoken=ptoken, count=-1)

    def pop(self, key: TokenPair, default=None) -> dict[PToken, int]:
        self._sorted_pairs.remove(key)
        self._pair_total_count.pop(key)

        return super().pop(key, default)

    def most_freq_pair(self) -> TokenPair:
        return self._sorted_pairs[-1]


class PTokenIndices(NamedTuple):
    counts: list[int]
    pairs_for_ptoken: list[list[TokenPair]]
    ptokens_for_pair: _PTokenIndex


@log_time("Preparing indices", logger=logger)
def prepare_indices(counts: dict[str, int]) -> PTokenIndices:
    pairs_for_ptoken: list[list[TokenPair]] = []

    ptoken_counts_by_idx = {i: v for i, v in enumerate(counts.values())}
    ptokens_for_pair: _PTokenIndex = _PTokenIndex(ptoken_counts_by_idx)

    for idx, ptoken_s in enumerate(counts):
        token_bytes = encode(ptoken_s)
        pairs = []
        for pair in get_ptoken_pairs(bytes([byte]) for byte in token_bytes):
            pairs.append(pair)
            ptokens_for_pair.inc(pair=pair, ptoken=idx)

        pairs_for_ptoken.append(pairs)

    return PTokenIndices(
        counts=list(counts.values()), pairs_for_ptoken=pairs_for_ptoken, ptokens_for_pair=ptokens_for_pair
    )
