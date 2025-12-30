from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

import sortedcollections

from cs336_basics._utils import encode
from cs336_basics.bpe._types import Token


@dataclass(slots=True)
class TokenPair:
    left: Token
    right: Token
    _ptokens: dict["PToken", int] = field(
        default_factory=lambda: defaultdict["PToken", int](lambda: 0), init=False, repr=False, hash=False, compare=False
    )
    _total_count: int = field(default=0, init=False, hash=False, compare=False)

    def add_count(self, ptoken: "PToken", count: int = 1) -> None:
        new_count = self._ptokens[ptoken] + count
        if new_count < 0:
            raise ValueError("Update of count would result in negative count")
        elif new_count == 0:
            self._ptokens.pop(ptoken)
        else:
            self._ptokens[ptoken] = new_count

        self._total_count += (ptoken.count or 0) * count

    def inc(self, ptoken: "PToken") -> None:
        self.add_count(ptoken, count=1)

    def dec(self, ptoken: "PToken") -> None:
        self.add_count(ptoken, count=-1)

    def to_token(self) -> Token:
        return self.left + self.right

    def ptokens(self) -> Iterable["PToken"]:
        return self._ptokens.keys()

    def merge(self) -> set["TokenPair"]:
        affected = set()
        for ptoken in tuple(self._ptokens):
            affected.update(ptoken.merge(self))

        return affected

    @property
    def total_count(self) -> int:
        return self._total_count

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, TokenPair):
            return False
        return self.left == value.left and self.right == value.right

    def __hash__(self) -> int:
        return hash((self.left, self.right))


@dataclass(slots=True)
class PToken:
    text: str
    pair_factory: Callable[[Token, Token], TokenPair] = field(hash=False, compare=False, repr=False)
    count: int | None = field(default=None, hash=False, compare=False)

    _pairs: list[TokenPair] = field(init=False, hash=False, compare=False, repr=False)
    _token: Token | None = field(init=False, hash=False, compare=False)

    def __post_init__(
        self,
    ) -> None:
        if len(self.text) == 0:
            raise ValueError("Cannot create PToken with empty string")

        bytes = encode(self.text)

        if len(bytes) == 1:
            self._pairs = []
            self._token = bytes
        else:
            self._pairs = list(self.pair_factory(left, right) for left, right in _get_ptoken_pairs_from_bytes(bytes))
            for pair in self._pairs:
                pair.inc(self)
            self._token = None

    @property
    def tokens(self) -> list[Token]:
        if self._token is not None:
            return [self._token]

        return list(_reconstruct_ptoken_subtokens_from_pairs(self._pairs))

    @property
    def pairs(self) -> list[TokenPair]:
        return self._pairs

    def merge(self, pair: TokenPair) -> set[TokenPair]:
        affected = set()

        for s_pair in self._pairs:
            affected.add(s_pair)
            s_pair.dec(self)

        tokens = _get_new_tokens_after_merge(pair, self._pairs)

        assert b"".join(tokens) == b"".join(self.tokens)

        if len(tokens) == 1:
            self._token = tokens[0]
            self._pairs = []
        else:
            self._pairs = list(self.pair_factory(left, right) for left, right in _get_ptoken_pairs(tokens))
            for pair in self._pairs:
                affected.add(pair)
                pair.inc(self)

        return affected

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, PToken):
            return False
        return self.text == value.text

    def __hash__(self) -> int:
        return hash(self.text)


def _get_new_tokens_after_merge(merge: TokenPair, pairs: list[TokenPair]) -> list[Token]:
    new_tokens = []
    last_merged = False

    new_token = merge.to_token()

    for i, pair in enumerate(pairs):
        merged_pair = False

        if not last_merged:
            if pair != merge:
                new_tokens.append(pair.left)
            else:
                new_tokens.append(new_token)
                merged_pair = last_merged = True
        else:
            last_merged = False

        if i == len(pairs) - 1 and not merged_pair:  # Handle last case seperately
            new_tokens.append(pair.right)

    return new_tokens


def _reconstruct_ptoken_subtokens_from_pairs(pairs: Iterable[TokenPair]) -> Iterator[Token]:
    last_token = None

    for pair in pairs:
        last_token = pair.right
        yield pair.left

    assert last_token is not None
    yield last_token


def _get_ptoken_pairs(ptoken_subtokens: Iterable[Token]) -> Iterator[tuple[Token, Token]]:
    prev_token = None

    for token in ptoken_subtokens:
        if prev_token is None:
            prev_token = token
            continue
        yield (prev_token, token)
        prev_token = token


def _get_ptoken_pairs_from_bytes(s_bytes: bytes) -> Iterator[tuple[Token, Token]]:
    return _get_ptoken_pairs(bytes([b]) for b in s_bytes)


class TokenPairFactory:
    def __init__(self) -> None:
        self._pairs = {}

    def __call__(self, left: Token, right: Token) -> TokenPair:
        if (left, right) in self._pairs:
            return self._pairs[(left, right)]

        pair = TokenPair(left, right)
        self._pairs[(left, right)] = pair
        return pair

    def get_pair(self, left: Token, right: Token) -> TokenPair | None:
        return self._pairs.get((left, right), None)


class OrderedTokenPairIndex:
    def __init__(self):
        self._index = sortedcollections.ValueSortedDict()

    def _pair_in_index(self, pair: TokenPair) -> bool:
        if pair not in self._index:
            return False

        total_count, (left, right) = self._index[pair]

        return pair.total_count == total_count and pair.left == left and pair.right == right

    def remove(self, pair: TokenPair) -> None:
        if pair not in self._index:
            return

        self._index.pop(pair)

    def update(self, pair: TokenPair) -> None:
        if self._pair_in_index(pair):
            return

        self._index[pair] = (pair.total_count, (pair.left, pair.right))

    @property
    def most_frequent_pair(self) -> TokenPair:
        return self._index.keys()[-1]
