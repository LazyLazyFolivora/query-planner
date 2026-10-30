"""KG vocabulary: built once at load time, then reused.

The vocabulary (entity surfaces + relation names + alias/relation indexes) is
static for a given knowledge base. Building it per query is wasted work, so it
is constructed once and cached per KG instance.
"""
from __future__ import annotations

from ..common.normalization import normalize_key
from ..parsing.lexicon import Lexicon


class KGVocabulary(Lexicon):
    """Immutable, precomputed vocabulary view of a knowledge graph."""

    __slots__ = ("name", "_surfaces", "_relations", "_alias_index", "_relation_index", "_canonical")

    def __init__(self, kg) -> None:
        self.name = "kg"
        self._canonical = _canonical_lookup(kg)
        self._surfaces = _build_surfaces(kg)
        self._relations = _build_relations(kg)

        alias_index: dict[str, str] = {}
        for surface, canonical in self._canonical.items():
            alias_index.setdefault(normalize_key(surface), canonical)
            alias_index.setdefault(normalize_key(canonical), canonical)
        self._alias_index = alias_index
        self._relation_index = {normalize_key(r): r for r in self._relations}

    # ---- Lexicon protocol (O(1), already computed) ----
    def entity_surfaces(self) -> tuple[str, ...]:
        return self._surfaces

    def relation_names(self) -> tuple[str, ...]:
        return self._relations

    # ---- grounding helpers ----
    def link_entity(self, mention: str, fuzzy_threshold: float = 0.72, kg=None):
        from .linker import _fuzzy_entity

        key = normalize_key(mention)
        if not key or key.startswith("?"):
            return mention or None

        if kg is not None:
            exact = getattr(kg, "canonical", None)
            if callable(exact):
                hit = exact(mention)
                if hit:
                    return str(hit)

        hit = self._alias_index.get(key)
        if hit:
            return hit
        return _fuzzy_entity(key, self._alias_index, fuzzy_threshold)

    def link_relation(self, hint: str, fuzzy_threshold: float = 0.72):
        from .linker import _fuzzy_relation, _RELATION_SYNONYMS

        key = normalize_key(hint)
        if not key:
            return None

        hit = self._relation_index.get(key)
        if hit:
            return hit

        for relation in self._relations:
            if normalize_key(relation) in key:
                return relation

        for synonym, candidates in _RELATION_SYNONYMS.items():
            if normalize_key(synonym) not in key:
                continue
            for candidate in candidates:
                mapped = self._relation_index.get(normalize_key(candidate))
                if mapped:
                    return mapped

        return _fuzzy_relation(key, self._relation_index, fuzzy_threshold)

    def has_entity(self, name: str) -> bool:
        return self.link_entity(name) is not None

    def has_relation(self, name: str) -> bool:
        return self.link_relation(name) is not None


def _canonical_lookup(kg) -> dict[str, str]:
    aliases = getattr(kg, "aliases", {}) or {}
    if callable(aliases):
        aliases = aliases()
    return {str(k): str(v) for k, v in aliases.items()}


def _build_surfaces(kg) -> tuple[str, ...]:
    surfaces: list[str] = [str(s) for s in _canonical_lookup(kg).keys()]
    names = getattr(kg, "entity_names", None)
    if callable(names):
        surfaces.extend(str(n) for n in names())
    ordered: list[str] = []
    for surface in surfaces:
        if surface and surface not in ordered:
            ordered.append(surface)
    return tuple(ordered)


def _build_relations(kg) -> tuple[str, ...]:
    names = getattr(kg, "relation_names", None)
    values = names() if callable(names) else (names or ())
    ordered: list[str] = []
    for name in values:
        text = str(name)
        if text and text not in ordered:
            ordered.append(text)
    return tuple(ordered)


_CACHE: dict[int, tuple[object, KGVocabulary]] = {}


def get_vocabulary(kg) -> KGVocabulary:
    """Return the cached vocabulary for ``kg``, building it once per instance."""
    key = id(kg)
    hit = _CACHE.get(key)
    if hit is not None and hit[0] is kg:
        return hit[1]
    vocab = KGVocabulary(kg)
    _CACHE[key] = (kg, vocab)
    return vocab