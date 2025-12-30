from collections.abc import Iterable, Iterator

from hypothesis import given
from hypothesis import strategies as st

from cs336_basics._utils import encode
from cs336_basics.bpe._pre_tokens.index import (
    _get_ptoken_pairs,
    _get_ptoken_pairs_from_bytes,
)
from cs336_basics.bpe._types import Token


def _get_ptoken_pairs_from_str(s: str) -> Iterator[tuple[Token, Token]]:
    return _get_ptoken_pairs_from_bytes(encode(s))


def _reconstruct_ptoken_subtokens_from_pairs(pairs: Iterable[tuple[Token, Token]]) -> Iterator[Token]:
    last_token = None

    for pair in pairs:
        last_token = pair[1]
        yield pair[0]

    assert last_token is not None
    yield last_token


def test_get_ptoken_pairs_empty_string():
    assert list(_get_ptoken_pairs_from_str("")) == []


def test_get_ptoken_pairs_empty_for_single_byte():
    for i in range(256):
        assert list(_get_ptoken_pairs([bytes([i])])) == []


@given(st.binary(min_size=2))
def test_get_ptoken_pairs_is_reversible_for_min_two_bytes(b: bytes):
    assert b"".join(_reconstruct_ptoken_subtokens_from_pairs(_get_ptoken_pairs_from_bytes(b))) == b


@given(st.binary())
def test_get_ptoken_pairs_has_correct_length(b: bytes):
    pairs = list(_get_ptoken_pairs_from_bytes(b))
    if len(b) == 0:
        assert len(pairs) == 0
    else:
        assert len(pairs) == len(b) - 1
