from cs336_basics.bpe._vocab import Vocabulary


def test_vocab_init():
    data = {0: b"\x00", 1: b"\x11"}
    v = Vocabulary(data)

    assert 0 in v
    assert 1 in v
    assert b"\x00" in v.reversed
    assert b"\x11" in v.reversed
    assert v[0] == b"\x00"
    assert v.reversed[b"\x00"] == 0
    assert v[1] == b"\x11"
    assert v.reversed[b"\x11"] == 1
    assert len(v) == 2
    assert len(v.reversed) == 2


def test_vocab_setitem():
    v = Vocabulary()
    v[0] = b"\x00"
    v[1] = b"\x11"
    assert 0 in v
    assert 1 in v
    assert b"\x00" in v.reversed
    assert b"\x11" in v.reversed
