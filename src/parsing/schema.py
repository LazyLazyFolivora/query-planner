"""Intermediate objects produced before KG grounding."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EntityDraft:
    role: str                 # start | target | answer
    mention: str
    type_hint: str | None = None


@dataclass(frozen=True)
class AnswerDraft:
    variable: str | None = None
    type_hint: str | None = None
    cardinality: str = "single"


@dataclass(frozen=True)
class RelationDraft:
    source: str
    predicate_hint: str
    target: str
    direction: str = "forward"
    strength: str = "required"
    hop: int | None = None


@dataclass(frozen=True)
class IntentDraft:
    template_id: str
    canonical_question: str
    entities: tuple[EntityDraft, ...] = ()
    answer: AnswerDraft = field(default_factory=AnswerDraft)
    relations: tuple[RelationDraft, ...] = ()
    path_shape: tuple[tuple[str, ...], ...] = ()
    exclusions: tuple[str, ...] = ()
    raw_text: str = ""