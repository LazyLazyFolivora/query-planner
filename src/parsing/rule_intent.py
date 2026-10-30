"""Deterministic fallback intent rewriter driven by surface cues.

Used when no LLM rewriter is configured. It only needs the user question,
plus an optional Lexicon to spot entity mentions. It emits *surface mentions*
(not canonical KG names) -- resolving mentions to canonical nodes is the job
of the grounding layer.
"""
from __future__ import annotations

from ..common.normalization import normalize_key
from .lexicon import EmptyLexicon, Lexicon
from .schema import AnswerDraft, EntityDraft, IntentDraft, RelationDraft

_MECHANISM_CUES = ("为什么", "为何", "机制", "原因", "怎么导致", "如何导致", "如何引起")
_INTERACTION_CUES = ("相互作用", "联合用药", "合用", "配伍")
_COUNT_CUES = ("多少", "几个", "几种", "数量")
_YES_NO_CUES = ("是否", "会不会", "能否", "能不能")
_COMPARE_CUES = ("比较", "区别", "差异")
_AGGREGATION_CUES = ("总结", "综合", "概述")


class RuleIntentRewriter:
    """A safe fallback when no LLM rewriter is configured."""

    name = "rule"

    def __init__(self, lexicon: Lexicon | None = None) -> None:
        self.lexicon = lexicon or EmptyLexicon()

    def rewrite(self, question: str) -> IntentDraft | None:
        """The question must already be normalized by ``SemanticParser``."""
        normalized = question
        if not normalized:
            return None

        mentions = self._mentions(normalized)
        template = self._template(normalized)
        answer_var = "?answer" if any(cue in normalized for cue in ("哪些", "什么药物", "哪些药物")) else None
        cardinality = (
            "set"
            if answer_var
            else "count"
            if any(cue in normalized for cue in _COUNT_CUES)
            else "boolean"
            if template == "yes_no"
            else "single"
        )
        cardinality = "count" if template == "counting" else cardinality
        answer = AnswerDraft(variable=answer_var, type_hint=self._type_hint(normalized), cardinality=cardinality)

        relation_hints = self._relation_hints(normalized)
        start = mentions[0] if mentions else ""
        target = mentions[-1] if len(mentions) > 1 else ""
        relation_drafts: list[RelationDraft] = []

        if answer_var and start:
            relation_hints = relation_hints or tuple(
                hint for hint in ("相互作用", "导致", "加重", "表现为") if hint in normalized
            )
            first_hint = relation_hints[0] if relation_hints else "相互作用"
            relation_drafts.append(RelationDraft(start, first_hint, answer_var))
            if target:
                second_hint = "导致"
                for candidate in ("增加", "加重", "导致", "引起", "表现为"):
                    if candidate in normalized:
                        second_hint = candidate
                        break
                relation_drafts.append(RelationDraft(answer_var, second_hint, target))
        elif start and target:
            for hint in relation_hints or ("导致",):
                relation_drafts.append(RelationDraft(start, hint, target))
        elif start and template == "attribute_lookup":
            relation_drafts.append(RelationDraft(start, "属性", "?answer"))

        role_entities = tuple(
            EntityDraft("start" if index == 0 else "target", mention)
            for index, mention in enumerate(mentions)
        )
        return IntentDraft(
            template_id=template,
            canonical_question=normalized,
            entities=role_entities,
            answer=answer,
            relations=tuple(relation_drafts),
            raw_text="rule-fallback",
        )

    def _mentions(self, question: str) -> list[str]:
        """Longest-match, non-overlapping surface mentions found in the question."""
        surfaces = sorted(
            (str(s) for s in self.lexicon.entity_surfaces() if str(s)),
            key=len,
            reverse=True,
        )
        lowered = normalize_key(question)
        claimed: list[tuple[int, int]] = []
        found: list[tuple[int, str]] = []
        for surface in surfaces:
            key = normalize_key(surface)
            if not key:
                continue
            pos = lowered.find(key)
            while pos >= 0:
                end = pos + len(key)
                if not any(pos < c_end and c_start < end for c_start, c_end in claimed):
                    claimed.append((pos, end))
                    found.append((pos, surface))
                    break
                pos = lowered.find(key, pos + 1)
        found.sort(key=lambda item: item[0])
        return [surface for _, surface in found]

    @staticmethod
    def _template(question: str) -> str:
        if any(cue in question for cue in _MECHANISM_CUES):
            return "mechanism_path"
        if any(cue in question for cue in _INTERACTION_CUES):
            return "interaction_set" if any(cue in question for cue in ("哪些", "什么药物")) else "chain_path"
        if any(cue in question for cue in _COUNT_CUES):
            return "counting"
        if any(cue in question for cue in _YES_NO_CUES):
            return "yes_no"
        if any(cue in question for cue in _COMPARE_CUES):
            return "comparison"
        if any(cue in question for cue in _AGGREGATION_CUES):
            return "aggregation"
        return "chain_path"

    @staticmethod
    def _type_hint(question: str) -> str | None:
        if "药物" in question or "药" in question:
            return "drug"
        if "疾病" in question or "病" in question:
            return "disease"
        if "基因" in question or "蛋白" in question:
            return "gene"
        return None

    def _relation_hints(self, question: str) -> tuple[str, ...]:
        hits: list[str] = []
        for name in self.lexicon.relation_names():
            text = str(name)
            if text and text in question and text not in hits:
                hits.append(text)
        return tuple(hits)