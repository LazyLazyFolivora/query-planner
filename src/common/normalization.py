"""Deterministic text normalization shared by parsing and linking."""
from __future__ import annotations

import re
import unicodedata


_PUNCT_TRANSLATION = str.maketrans({
    "，": ",",
    "。": ".",
    "？": "?",
    "！": "!",
    "；": ";",
    "：": ":",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
})


def normalize_text(text: str) -> str:
    """Normalize surface form without changing the question semantics."""

    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.translate(_PUNCT_TRANSLATION)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_key(text: str) -> str:
    """Normalize a mention for exact dictionary lookup."""

    return normalize_text(text).casefold()