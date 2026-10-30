"""Deterministic KG grounding for entities and relations.

All indices come from ``KGVocabulary``, which is built once per knowledge base
at load time. The linker itself is stateless and cheap to create.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from ..common.normalization import normalize_key, normalize_text


_RELATION_SYNONYMS = {
    "为何导致": ("导致", "引起", "加重"),
    "为什么导致": ("导致", "引起", "加重"),
    "引致": ("导致", "引起"),
    "引起": ("引起", "导致"),
    "造成": ("导致", "引起"),
    "使得": ("导致", "引起"),
    "作用机制": ("药理作用", "机制"),
    "药理机制": ("药理作用", "机制"),
    "副作用": ("不良反应",),
    "不良反应": ("不良反应",),
    "相互作用": ("相互作用",),
    "增加": ("加重", "导致", "引起"),
    "增加出血风险": ("加重", "导致", "引起"),
    "表现": ("表现为",),
    "禁忌": ("禁忌症", "禁忌"),
    "适应证": ("适用范围", "适应症"),
}


def _fuzzy_entity(key: str, alias_index: dict[str, str], threshold: float) -> str | None:
    best_name: str | None = None
    best_score = 0.0
    for surface_key, candidate in alias_index.items():
        score = SequenceMatcher(None, key, surface_key).ratio()
        if score > best_score:
            best_name, best_score = candidate, score
    return best_name if best_score >= threshold else None


def _fuzzy_relation(key: str, relation_index: dict[str, str], threshold: float) -> str | None:
    best_name: str | None = None
    best_score = 0.0
    for relation_key, relation in relation_index.items():
        score = SequenceMatcher(None, key, relation_key).ratio()
        if score > best_score:
            best_name, best_score = relation, score
    return best_name if best_score >= threshold else None


class KGLinker:
    """Resolve mentions against a precomputed KG vocabulary."""

    def __init__(self, kg, fuzzy_threshold: float = 0.72) -> None:
        from .vocabulary import get_vocabulary

        self.kg = kg
        self.fuzzy_threshold = fuzzy_threshold
        self.vocabulary = get_vocabulary(kg)

    def link_entity(self, mention: str) -> str | None:
        if not mention or mention.startswith("?"):
            return mention or None
        return self.vocabulary.link_entity(mention, self.fuzzy_threshold, kg=self.kg)

    def link_relation(self, hint: str) -> str | None:
        if not hint:
            return None
        return self.vocabulary.link_relation(hint, self.fuzzy_threshold)

    def has_entity(self, name: str) -> bool:
        return self.link_entity(name) is not None

    def has_relation(self, name: str) -> bool:
        return self.link_relation(name) is not None