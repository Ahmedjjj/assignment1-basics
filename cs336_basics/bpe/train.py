import logging
from collections.abc import Iterable

import tqdm

from cs336_basics._utils import log_time
from cs336_basics.bpe._pre_tokens import (
    OrderedTokenPairIndex,
    PToken,
    TokenPairFactory,
    par_count_unique_ptokens,
)
from cs336_basics.bpe._types import Merges
from cs336_basics.bpe._vocab import Vocabulary, init_vocab
from cs336_basics.bpe.bpe import BPETokenizer

logger = logging.getLogger(__name__)


@log_time("Training BPE", logger=logger)
def train_bpe(
    texts: str | Iterable[str], special_tokens: list[str], vocab_size: int, ptoken_count_n_jobs: int = 1
) -> BPETokenizer:
    vocab = init_vocab(special_tokens=special_tokens)

    num_merges = vocab_size - len(vocab)

    if num_merges <= 0:
        return BPETokenizer(vocab=vocab, merges=[], special_tokens=special_tokens)

    counts = par_count_unique_ptokens(texts, special_tokens=special_tokens, n_jobs=ptoken_count_n_jobs)
    index = _prepare_index(counts)
    merges = _run_merges(num_merges=num_merges, index=index, vocab=vocab)

    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


@log_time("Preparing index", logger=logger)
def _prepare_index(counts: dict[str, int]) -> OrderedTokenPairIndex:
    pair_factory = TokenPairFactory()
    ptokens: list[PToken] = []
    for text, count in counts.items():
        ptokens.append(PToken(text=text, pair_factory=pair_factory, count=count))

    index = OrderedTokenPairIndex(pair_factory.pairs)

    return index


@log_time("Running merges", logger=logger)
def _run_merges(num_merges: int, index: OrderedTokenPairIndex, vocab: Vocabulary) -> Merges:
    merges = []
    for _ in tqdm.tqdm(range(num_merges)):
        pair = index.most_frequent_pair
        merges.append((pair.left, pair.right))
        vocab[len(vocab)] = pair.to_token()
        affected_pairs = pair.merge()
        index.update(*affected_pairs)

    return merges
