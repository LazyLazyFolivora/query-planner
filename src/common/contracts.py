"""Shared cross-stage contracts.

These objects are intentionally independent of any specific parser,
retriever, model, or storage backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field


PARSE_LINKED = "linked"
PARSE_PARTIAL = "partial"
PARSE_UNLINKED = "unlinked"
PARSE_FALLBACK = "fallback"

RELATION_REQUIRED = "required"
RELATION_PREFERRED = "preferred"

DIRECTION_FORWARD = "forward"
DIRECTION_BACKWARD = "backward"
DIRECTION_BIDIRECTIONAL = "bidirectional"
DIRECTION_UNKNOWN = "unknown"

EVIDENCE_SHAPE_PATH = "path"
EVIDENCE_SHAPE_SUBGRAPH = "subgraph"
EVIDENCE_SHAPE_RELATION_SET = "relation_set"
EVIDENCE_SHAPE_SINGLE_FACT = "single_fact"
EVIDENCE_SHAPE_COUNT = "count"


@dataclass(frozen=True)
class AnswerSlot:
    """A logical variable that must be found or counted."""

    variable: str
    type: str | None = None
    cardinality: str = "single"  # single | set | count | boolean


@dataclass(frozen=True)
class RelationConstraint:
    """A relation requirement over canonical entities or logical variables."""

    source: str
    relation: str
    target: str
    direction: str = DIRECTION_FORWARD
    strength: str = RELATION_REQUIRED
    hop: int | None = None
    raw_hint: str = ""


@dataclass(frozen=True)
class EvidenceCheck:
    """A machine-verifiable check derived from a question template."""

    check_id: str
    required: bool = True
    params: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class EvidenceSpec:
    """Backend-independent description of what evidence must be collected."""

    shape: str
    hard_constraints: tuple[RelationConstraint, ...] = ()
    soft_constraints: tuple[RelationConstraint, ...] = ()
    checks: tuple[EvidenceCheck, ...] = ()
    answer_slots: tuple[AnswerSlot, ...] = ()


@dataclass(frozen=True)
class QueryIntent:
    """Normalized, backend-independent interpretation of a user question."""

    raw_question: str
    canonical_question: str
    template_id: str
    start_entities: tuple[str, ...] = ()
    target_entities: tuple[str, ...] = ()
    answer_slots: tuple[AnswerSlot, ...] = ()
    target_types: tuple[str, ...] = ()
    relation_constraints: tuple[RelationConstraint, ...] = ()
    path_template: tuple[tuple[str, ...], ...] = ()
    reasoning_mode: str = "chain"
    answer_mode: str = "entity"
    exclusions: tuple[str, ...] = ()
    unlinked_mentions: tuple[str, ...] = ()
    parse_status: str = PARSE_LINKED
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ParsingResult:
    """Result returned by the parsing pipeline."""

    intent: QueryIntent
    evidence_spec: EvidenceSpec
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        return self.intent.parse_status


def is_variable(name: str) -> bool:
    return name.startswith("?")