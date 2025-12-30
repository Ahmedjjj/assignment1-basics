import logging
from collections import Counter
from collections.abc import Iterable, Iterator

import regex as re
from joblib import Parallel, delayed

from cs336_basics._utils import log_time

logger = logging.getLogger(__name__)


PRETOKEN_PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def find_pre_tokens(texts: str | Iterable[str], /) -> Iterator[str]:
    if isinstance(texts, str):
        texts = (texts,)

    for text in texts:
        for m in PRETOKEN_PAT.finditer(text):
            yield m.group(0)


def split_on_special_tokens(
    texts: str | Iterable[str], /, special_tokens: list[str], return_special_tokens: bool = False
) -> Iterator[str]:
    if isinstance(texts, str):
        texts = (texts,)

    if len(special_tokens) == 0:
        yield from texts
        return

    split_re = "|".join(re.escape(token) for token in sorted(special_tokens, key=lambda x: -len(x)))
    if return_special_tokens:
        split_re = rf"({split_re})"

    split_re = re.compile(split_re)

    for text in texts:
        yield from split_re.split(text)


@log_time("Counting unique ptokens in parallel", logger=logger)
def par_count_unique_ptokens(
    texts: str | Iterable[str], /, special_tokens: Iterable[str], n_jobs: int = 1
) -> dict[str, int]:
    return _sum_counters(_par_count(texts, special_tokens=special_tokens, n_jobs=n_jobs))


def _count_unique_ptokens(text: str, special_tokens: list[str]) -> Counter[str]:
    texts = split_on_special_tokens(text, special_tokens=special_tokens)
    ptokens = find_pre_tokens(texts)
    return Counter(ptokens)


@log_time("Executing parallel counts", logger=logger)
def _par_count(texts: str | Iterable[str], /, special_tokens: Iterable[str], n_jobs: int = 1) -> Iterator[Counter[str]]:
    if isinstance(texts, str):
        texts = [texts]
    parallel = Parallel(n_jobs=n_jobs, return_as="generator")
    return parallel(delayed(_count_unique_ptokens)(text=text, special_tokens=special_tokens) for text in texts)  #  type: ignore (https://github.com/joblib/joblib/issues/1176)


@log_time("Aggregating counts from sub-processes", logger=logger)
def _sum_counters(counters: Iterator[Counter[str]]) -> Counter[str]:
    res = Counter[str]()

    for counter in counters:
        res += counter
        del counter
    return res
