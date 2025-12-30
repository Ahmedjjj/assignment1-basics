from cs336_basics._utils import encode
from cs336_basics.bpe._pre_tokens import PToken, TokenPairFactory, find_pre_tokens, split_on_special_tokens
from cs336_basics.bpe._types import Merges, Token
from cs336_basics.bpe._vocab import Vocabulary


def tokenize(text: str, vocab: dict[int, Token], merges: Merges, special_tokens: list[str] | None = None) -> list[int]:
    special_tokens = special_tokens or []
    chunks = split_on_special_tokens(text, special_tokens=special_tokens, return_special_tokens=True)

    vocab = Vocabulary(vocab)

    all_ptokens = []
    ptokens_to_encode = set()
    for chunk in chunks:
        if chunk in special_tokens:
            all_ptokens.append(chunk)
        else:
            ptokens = list(filter(lambda x: len(x) > 0, find_pre_tokens(chunk)))
            ptokens_to_encode.update(ptokens)
            all_ptokens.extend(ptokens)

    pair_factory = TokenPairFactory()
    ptokens = {}

    for ptoken_s in ptokens_to_encode:
        ptokens[ptoken_s] = PToken(text=ptoken_s, pair_factory=pair_factory)

    for left, right in merges:
        pair = pair_factory.get_pair(left, right)
        if pair is None:
            continue
        pair.merge()

    encoding = []
    for ptoken in all_ptokens:
        if ptoken in special_tokens:
            encoding.append(vocab.reversed[encode(ptoken)])
        else:
            encoding.extend(vocab.reversed[b] for b in ptokens[ptoken].tokens)

    return encoding
