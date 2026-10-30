"""Grounding: ungrounded IntentDraft + vocabulary -> QueryIntent.

This is where mentions are resolved against the knowledge base vocabulary and
where the backend-independent evidence requirements are produced.

The linker is derived from the precomputed vocabulary, so grounding a question
does not rebuild any index.
"""
from __future__ import annotations

from ..common.contracts import (
    PARSE_LINKED,
    PARSE_PARTIAL,
    PARSE_UNLINKED,
    AnswerSlot,
    QueryIntent,
    RelationConstraint,
    is_variable,
)
from ..parsing.schema import IntentDraft
from .linker import KGLinker
from .validator import IntentValidator


_REASONING_BY_TEMPLATE = {
    "mechanism_path": "mechanism",
    "chain_path": "chain",
    "interaction_set": "intersection",
    "attribute_lookup": "chain",
    "comparison": "intersection",
    "counting": "counting",
    "yes_no": "chain",
    "aggregation": "aggregation",
}


class IntentGrounder:
    """Resolve an intent draft against a knowledge base vocabulary."""

    def __init__(
        self,
        *,
        fuzzy_threshold: float = 0.72,
        linker_factory=KGLinker,
        validator: IntentValidator | None = None,
    ) -> None:
        self.fuzzy_threshold = fuzzy_threshold
        self.linker_factory = linker_factory
        self.validator = validator or IntentValidator()

    def ground(self, raw_question: str, normalized: str, draft: IntentDraft, kg):
        """Ground a draft.

        ``raw_question`` is kept verbatim in ``QueryIntent.raw_question``;
        ``normalized`` is the already-normalized text used for matching.
        """
        linker = self.linker_factory(kg, fuzzy_threshold=self.fuzzy_threshold)
        diagnostics: list[str] = []
        intent = self._resolve(raw_question, normalized, draft, linker, diagnostics)
        report = self.validator.validate(intent)
        diagnostics.extend(f"validation_error:{item}" for item in report.errors)
        diagnostics.extend(f"validation_warning:{item}" for item in report.warnings)
        return intent, tuple(diagnostics)

    def _resolve(
        self,
        raw_question: str,
        normalized: str,
        draft: IntentDraft,
        linker: KGLinker,
        diagnostics: list[str],
    ) -> QueryIntent:
        unlinked: list[str] = []
        start_entities: list[str] = []
        target_entities: list[str] = []

        for entity in draft.entities:
            canonical = linker.link_entity(entity.mention)
            if canonical is None:
                unlinked.append(entity.mention)
                continue
            bucket = start_entities if entity.role == "start" else target_entities
            if canonical not in bucket:
                bucket.append(canonical)

        relation_constraints: list[RelationConstraint] = []
        for relation in draft.relations:
            source = self._resolve_node(relation.source, linker, diagnostics)
            target = self._resolve_node(relation.target, linker, diagnostics)
            relation_name = linker.link_relation(relation.predicate_hint)
            if relation_name is None:
                relation_name = relation.predicate_hint
                diagnostics.append(f"unlinked_relation:{relation.predicate_hint}")
            relation_constraints.append(RelationConstraint(
                source=source,
                relation=relation_name,
                target=target,
                direction=relation.direction or "unknown",
                strength=relation.strength or "required",
                hop=relation.hop,
                raw_hint=relation.predicate_hint,
            ))

        answer_slots: list[AnswerSlot] = []
        if draft.answer.variable:
            answer_slots.append(AnswerSlot(
                variable=draft.answer.variable,
                type=draft.answer.type_hint,
                cardinality=draft.answer.cardinality,
            ))

        target_types: list[str] = []
        for entity in draft.entities:
            if entity.role == "target" and entity.type_hint and entity.type_hint not in target_types:
                target_types.append(entity.type_hint)
        if draft.answer.type_hint and draft.answer.type_hint not in target_types:
            target_types.append(draft.answer.type_hint)

        unlinked = list(dict.fromkeys(unlinked))
        if not start_entities:
            parse_status = PARSE_UNLINKED
        elif unlinked or any("unlinked_relation:" in item for item in diagnostics):
            parse_status = PARSE_PARTIAL
        else:
            parse_status = PARSE_LINKED

        return QueryIntent(
            raw_question=raw_question,
            canonical_question=draft.canonical_question or normalized,
            template_id=draft.template_id,
            start_entities=tuple(start_entities),
            target_entities=tuple(target_entities),
            answer_slots=tuple(answer_slots),
            target_types=tuple(target_types),
            relation_constraints=tuple(relation_constraints),
            path_template=draft.path_shape,
            reasoning_mode=_REASONING_BY_TEMPLATE.get(draft.template_id, "chain"),
            answer_mode=_answer_mode(draft),
            exclusions=draft.exclusions,
            unlinked_mentions=tuple(unlinked),
            parse_status=parse_status,
            metadata=(("raw_draft", draft.raw_text),),
        )

    @staticmethod
    def _resolve_node(value: str, linker: KGLinker, diagnostics: list[str]) -> str:
        if is_variable(value):
            return value
        canonical = linker.link_entity(value)
        if canonical is None:
            diagnostics.append(f"unlinked_entity:{value}")
            return value
        return canonical


def _answer_mode(draft: IntentDraft) -> str:
    cardinality = (draft.answer.cardinality or "single").lower()
    if cardinality in {"set", "count", "boolean"}:
        return cardinality
    return "entity"