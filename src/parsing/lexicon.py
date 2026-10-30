"""Lexicon: a vocabulary-only view of a knowledge graph.

The semantic parsing layer must not read KG evidence. All it may consult is
the *vocabulary* (entity surface forms and relation names), which is what a
Lexicon exposes. This keeps ``parsing`` free of any KG implementation.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Lexicon(Protocol):
    """Vocabulary view over a knowledge base."""

    def entity_surfaces(self) -> tuple[str, ...]:
        """All surface forms that may appear in a question (names + aliases)."""
        ...

    def relation_names(self) -> tuple[str, ...]:
        """All relation names available in the knowledge base."""
        ...


class EmptyLexicon:
    """A lexicon without vocabulary, used when no knowledge base is bound."""

    name = "empty"

    def entity_surfaces(self) -> tuple[str, ...]:
        return ()

    def relation_names(self) -> tuple[str, ...]:
        return ()