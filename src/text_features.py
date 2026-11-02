"""Reproducible preprocessing and sparse features for PTIT comments.

Profiles are fixed rules, never learned from validation or test labels.
Saved optimized pipelines accept raw strings and include this preprocessing.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.preprocessing import clean_text, segment_vietnamese

PROFILES = ("preserve", "slang")
PROFILE_VERSION = "steps23_v1"
# Whole-token replacements only. In particular, k, r, đk, and numeric units
# remain unchanged because they have multiple common meanings.
ALIASES = {
    "ko": "không", "k0": "không", "khum": "không", "hok": "không",
    "khong": "không", "đc": "được", "thik": "thích", "cx": "cũng",
    "rùi": "rồi", "nma": "nhưng mà",
}
_ALIASES_RE = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, ALIASES)) + r")(?!\w)")
_URL = re.compile(r"(?:https?://|www\.)[^\s<>]+", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"\[\s*(MENTION|URL)\s*\]", re.IGNORECASE)
INPUT_COLUMNS = ["text_clean", "text_tokenized"]


def normalize_for_features(text: str, profile: str = "preserve") -> str:
    if profile not in PROFILES:
        raise ValueError(f"Unknown preprocessing profile: {profile}")
    normalized = clean_text(text)
    normalized = _URL.sub(lambda match: "[URL]" + match[0][len(match[0].rstrip(".,!?;:)")):], normalized)
    normalized = _PLACEHOLDER.sub(lambda match: f"[{match[1].upper()}]", normalized)
    if profile == "slang":
        # Case-sensitive rules avoid rewriting uppercase acronyms and names.
        normalized = _ALIASES_RE.sub(lambda match: ALIASES[match[0]], normalized)
    return clean_text(normalized)


def _segment_features(text: str) -> str:
    # Tokenizers can return MENTION_] or MENTION_]; isolate placeholders before
    # segmentation rather than trying to repair tokenizer-specific output.
    pieces = []
    start = 0
    for match in _PLACEHOLDER.finditer(text):
        prefix = text[start:match.start()].strip()
        if prefix:
            pieces.append(segment_vietnamese(prefix))
        pieces.append(f"[{match[1].upper()}]")
        start = match.end()
    suffix = text[start:].strip()
    if suffix:
        pieces.append(segment_vietnamese(suffix))
    return " ".join(pieces)


class CommentPreprocessor(TransformerMixin, BaseEstimator):
    """A stateless sklearn transformer; rejects accidental metadata input."""

    def __init__(self, profile: str = "preserve"):
        self.profile = profile

    def fit(self, X, y=None):
        if self.profile not in PROFILES:
            raise ValueError(f"Unknown preprocessing profile: {self.profile}")
        return self

    def transform(self, X: Iterable[str]) -> pd.DataFrame:
        if isinstance(X, (str, bytes, pd.DataFrame)):
            raise TypeError("Expected an iterable of raw comment strings")
        texts = [normalize_for_features(text, self.profile) for text in X]
        tokenized = [_segment_features(text) for text in texts]
        return pd.DataFrame({"text_clean": texts, "text_tokenized": tokenized})


def build_features(
    kind: str, *, ngram_range: tuple[int, int] = (1, 1), min_df: int = 1,
    char_range: tuple[int, int] = (3, 5), char_weight: float = 1.0,
    word_stop_words: list[str] | None = None,
) -> ColumnTransformer:
    """Use identical word settings for BoW/TF-IDF controlled comparisons."""
    if kind not in ("bow", "tfidf", "char", "word_char"):
        raise ValueError(f"Unknown feature kind: {kind}")
    branches = []
    if kind != "char":
        vectorizer = CountVectorizer if kind == "bow" else TfidfVectorizer
        options = dict(tokenizer=str.split, token_pattern=None, lowercase=True,
                       ngram_range=ngram_range, min_df=min_df, stop_words=word_stop_words)
        if kind != "bow":
            options.update(sublinear_tf=True, smooth_idf=True, norm="l2")
        branches.append(("word", vectorizer(**options), "text_tokenized"))
    if kind in ("char", "word_char"):
        branches.append(("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=char_range, min_df=min_df,
            lowercase=True, sublinear_tf=True, smooth_idf=True, norm="l2",
        ), "text_clean"))
    weights = {"word": 1.0, "char": char_weight} if kind == "word_char" else None
    return ColumnTransformer(branches, remainder="drop", sparse_threshold=1.0,
                             transformer_weights=weights)
