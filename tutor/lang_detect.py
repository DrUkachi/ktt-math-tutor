"""Lightweight language detector for KIN / EN / FR / mix.

A full langid model would blow the footprint budget for what is essentially
a 4-way classifier on short utterances. This module uses a hand-rolled
character-n-gram lookup over closed-class words (numbers 1-20, polite words,
common verbs) which is enough for the tutor's reply-routing decision.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Literal

Lang = Literal["kin", "en", "fr", "mix"]


# Number words 1-10 — the most common tokens the tutor will see.
_KIN_NUMBERS = {
    "rimwe", "kabiri", "gatatu", "kane", "gatanu",
    "gatandatu", "karindwi", "umunani", "icyenda", "icumi",
}
_EN_NUMBERS = {
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
}
_FR_NUMBERS = {
    "un", "deux", "trois", "quatre", "cinq",
    "six", "sept", "huit", "neuf", "dix",
}

# Tiny closed-class anchor sets — words a child is likely to use.
_KIN_ANCHORS = _KIN_NUMBERS | {"yego", "oya", "muraho", "ndi", "ni", "ntabwo"}
_EN_ANCHORS = _EN_NUMBERS | {"yes", "no", "the", "is", "and", "i"}
_FR_ANCHORS = _FR_NUMBERS | {"oui", "non", "le", "la", "et", "je"}


_WORD_RE = re.compile(r"[a-zA-Z\u00c0-\u017f]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text)]


def detect(text: str) -> Lang:
    """Return the dominant language of ``text``, or ``'mix'`` if 2+ tie."""
    toks = _tokens(text)
    if not toks:
        return "en"
    scores = Counter()
    for tok in toks:
        if tok in _KIN_ANCHORS:
            scores["kin"] += 1
        if tok in _EN_ANCHORS:
            scores["en"] += 1
        if tok in _FR_ANCHORS:
            scores["fr"] += 1
    if not scores:
        # Fall back to crude heuristic: Kinyarwanda has frequent 'rw', 'nk'.
        if any(b in text.lower() for b in ("rw", "nk", "ny")):
            return "kin"
        return "en"
    top = scores.most_common()
    if len(top) >= 2 and top[0][1] == top[1][1]:
        return "mix"
    return top[0][0]  # type: ignore[return-value]


def number_words(text: str) -> dict[Lang, list[str]]:
    """Return the number-words found in ``text``, grouped by language.

    Used by the inference loop to mirror the dominant language while embedding
    the second language for any number words the child actually used.
    """
    toks = _tokens(text)
    return {
        "kin": [t for t in toks if t in _KIN_NUMBERS],
        "en": [t for t in toks if t in _EN_NUMBERS],
        "fr": [t for t in toks if t in _FR_NUMBERS],
        "mix": [],
    }
