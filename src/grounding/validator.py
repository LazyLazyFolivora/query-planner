"""Validation for resolved query intents."""
from __future__ import annotations

from dataclasses import dataclass

from ..common.contracts import QueryIntent
from ..parsing.llm_intent import TEMPLATE_IDS


@dataclass(frozen=True)
class ValidationReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


class IntentValidator:
    def validate(self, intent: QueryIntent) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []

        if not intent.canonical_question:
            errors.append("canonical_question is empty")
        if intent.template_id not in TEMPLATE_IDS:
            errors.append(f"unknown template_id: {intent.template_id}")
        if not intent.start_entities:
            errors.append("no linked start entity")
        if not intent.relation_constraints and intent.template_id not in {"yes_no"}:
            warnings.append("no relation constraints")
        if intent.answer_slots and not any(intent.answer_slots):
            warnings.append("answer slots are empty")

        for constraint in intent.relation_constraints:
            if not constraint.source or not constraint.relation or not constraint.target:
                errors.append("relation constraint has an empty field")
            if constraint.strength not in {"required", "preferred"}:
                errors.append(f"invalid constraint strength: {constraint.strength}")
            if constraint.direction not in {"forward", "backward", "bidirectional", "unknown"}:
                errors.append(f"invalid relation direction: {constraint.direction}")

        return ValidationReport(tuple(errors), tuple(warnings))