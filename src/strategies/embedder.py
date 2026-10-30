"""兼容门面：``src.strategies.embedder`` -> :mod:`src.infrastructure.embedder`。"""

from __future__ import annotations

from ..infrastructure.embedder import *  # noqa: F401,F403
from ..infrastructure.embedder import (  # noqa: F401
    DEFAULT_EMBED_MODEL,
    Embedder,
    HashEmbedder,
    SentenceEmbedder,
)