"""Deterministic cleaning and training-only vocabulary, with no downloads."""
import html
import re
from collections import Counter

PAD, UNK = "<PAD>", "<UNK>"
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)


def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    return " ".join(html.unescape(text).split())


def tokenize(text):
    return TOKEN_PATTERN.findall(clean_text(text).lower())


def build_vocab(texts, max_size=30000):
    if max_size < 2:
        raise ValueError("Vocabulary must include PAD and UNK")
    counts = Counter(token for text in texts for token in tokenize(text))
    ordered = sorted(counts, key=lambda token: (-counts[token], token))
    return {token: i for i, token in enumerate([PAD, UNK] + ordered[:max_size-2])}


def encode(text, vocab, max_length=400):
    if max_length < 5:
        raise ValueError("max_length must be at least the largest CNN kernel (5)")
    ids = [vocab.get(token, 1) for token in tokenize(text)[:max_length]]
    return ids + [0] * (max_length-len(ids))
