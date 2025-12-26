import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import NamedTuple

import regex as re
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
    def add_count(self, pair: TokenPair, ptoken: PToken, count: int = 1) -> None:
        if pair not in self:
            self[pair] = {}

        counts = self[pair]
        if ptoken not in counts:
            counts[ptoken] = 0

        if counts[ptoken] + count < 0:
            raise ValueError("PToken count is < 0")

        counts[ptoken] += count

    def inc(self, pair: TokenPair, ptoken: PToken) -> None:
        self.add_count(pair=pair, ptoken=ptoken, count=1)

    def dec(self, pair: TokenPair, ptoken: PToken) -> None:
        self.add_count(pair=pair, ptoken=ptoken, count=-1)


class PTokenIndices(NamedTuple):
    counts: list[int]
    pairs_for_ptoken: list[list[TokenPair]]
    ptokens_for_pair: _PTokenIndex


@log_time("Preparing indices", logger=logger)
def prepare_indices(counts: dict[str, int]) -> PTokenIndices:
    pairs_for_ptoken: list[list[TokenPair]] = []
    ptokens_for_pair: _PTokenIndex = _PTokenIndex()

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
