from cs336_basics.bpe._pre_tokens import split_on_special_tokens


def test_split_on_special_token_discards_special_tokens():
    test_s = "Hello <|endoftext|> World"
    texts = split_on_special_tokens(test_s, special_tokens=["<|endoftext|>"])
    assert tuple(texts) == ("Hello ", " World")


def test_split_on_special_token_returns_special_tokens():
    test_s = "Hello <|endoftext|> World"
    texts = split_on_special_tokens(test_s, special_tokens=["<|endoftext|>"], return_special_tokens=True)
    assert tuple(texts) == ("Hello ", "<|endoftext|>", " World")
