from app.words import tokenize


def test_identifiers_stay_whole():
    assert tokenize("Error KST-2044: allow 203.0.113.10.") == ["error", "kst-2044", "allow", "203.0.113.10"]


def test_lowercase_and_apostrophes():
    assert tokenize("The Captain’s desk, can't LOG in") == ["the", "captain’s", "desk", "can't", "log", "in"]


def test_punctuation_and_underscores_split_words():
    assert tokenize("—Wait_and hope!— (Fac et spera)") == ["wait", "and", "hope", "fac", "et", "spera"]
