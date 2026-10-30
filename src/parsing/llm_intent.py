"""Optional LLM-backed question rewriting and slot extraction.

This module belongs to the semantic layer: it takes a question and returns an
ungrounded ``IntentDraft``. It never touches a knowledge graph.
"""
from __future__ import annotations

import json
import re
from typing import Any, Protocol, runtime_checkable

from .schema import AnswerDraft, EntityDraft, IntentDraft, RelationDraft


TEMPLATE_IDS = (
    "mechanism_path",
    "chain_path",
    "interaction_set",
    "attribute_lookup",
    "comparison",
    "counting",
    "yes_no",
    "aggregation",
)


@runtime_checkable
class IntentRewriter(Protocol):
    name: str

    def rewrite(self, question: str) -> IntentDraft | None: ...

    def is_available(self) -> bool: ...


class NullIntentRewriter:
    name = "null"

    def rewrite(self, question: str) -> IntentDraft | None:
        return None

    def is_available(self) -> bool:
        return False


class LLMIntentRewriter:
    """Use a chat-completion-like client with a strict JSON contract."""

    name = "llm"

    def __init__(self, llm) -> None:
        self.llm = llm

    def is_available(self) -> bool:
        """委托给底层客户端判断。

        客户端没配 base_url / model / key 时返回 False，
        这样 SemanticParser 能把"没配置"和"调用失败"区分开。
        """
        check = getattr(self.llm, "is_available", None)
        return bool(check()) if callable(check) else True

    def rewrite(self, question: str) -> IntentDraft | None:
        prompt = build_intent_prompt(question)
        raw = (self.llm.complete(prompt) or "").strip()
        payload = _load_json_object(raw)
        if not payload:
            return None
        return draft_from_payload(payload, raw_text=raw)


def build_intent_prompt(question: str) -> str:
    templates = ", ".join(TEMPLATE_IDS)
    return f"""你是知识图谱问答系统的查询意图解析器。
只做语义规范化，不回答问题，不生成证据。

允许的 template_id：
{templates}

请把用户问题改写成严格 JSON，字段如下：
{{
  "template_id": "...",
  "canonical_question": "...",
  "entities": [{{"role": "start|target", "mention": "...", "type_hint": null}}],
  "answer": {{"variable": "?name|null", "type_hint": null, "cardinality": "single|set|count|boolean"}},
  "relation_constraints": [
    {{
      "source": "...",
      "predicate_hint": "...",
      "target": "...",
      "direction": "forward|backward|bidirectional|unknown",
      "strength": "required|preferred",
      "hop": null
    }}
  ],
  "path_shape": [],
  "exclusions": []
}}

要求：
1. 实体和关系可以输出原始提及或语义提示，不要强行编造知识图谱规范名。
2. 未知答案必须使用以 ? 开头的变量，例如 ?drug。
3. 只输出 JSON，不要输出 Markdown 或解释。
4. 不得输出 gold answer。

用户问题：
{question}
"""


def _strip_code_fence(raw: str) -> str:
    value = raw.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
    return value.strip()


def _load_json_object(raw: str) -> dict[str, Any] | None:
    value = _strip_code_fence(raw)
    if not value:
        return None
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _text(value: Any, default: str = "") -> str:
    return str(value).strip() if value is not None else default


def draft_from_payload(payload: dict[str, Any], *, raw_text: str = "") -> IntentDraft:
    entities: list[EntityDraft] = []
    for item in payload.get("entities") or []:
        if not isinstance(item, dict):
            continue
        mention = _text(item.get("mention"))
        if mention:
            entities.append(EntityDraft(
                role=_text(item.get("role"), "target"),
                mention=mention,
                type_hint=_text(item.get("type_hint")) or None,
            ))

    answer_payload = payload.get("answer") or {}
    if not isinstance(answer_payload, dict):
        answer_payload = {}
    answer = AnswerDraft(
        variable=_text(answer_payload.get("variable")) or None,
        type_hint=_text(answer_payload.get("type_hint")) or None,
        cardinality=_text(answer_payload.get("cardinality"), "single") or "single",
    )

    relations: list[RelationDraft] = []
    for item in payload.get("relation_constraints") or []:
        if not isinstance(item, dict):
            continue
        source = _text(item.get("source"))
        predicate = _text(item.get("predicate_hint"))
        target = _text(item.get("target"))
        if not source or not predicate or not target:
            continue
        hop_raw = item.get("hop")
        try:
            hop = int(hop_raw) if hop_raw is not None else None
        except (TypeError, ValueError):
            hop = None
        relations.append(RelationDraft(
            source=source,
            predicate_hint=predicate,
            target=target,
            direction=_text(item.get("direction"), "forward") or "forward",
            strength=_text(item.get("strength"), "required") or "required",
            hop=hop,
        ))

    path_shape: list[tuple[str, ...]] = []
    for step in payload.get("path_shape") or []:
        if isinstance(step, str):
            path_shape.append((step,))
        elif isinstance(step, (list, tuple)):
            path_shape.append(tuple(_text(x) for x in step if _text(x)))

    return IntentDraft(
        template_id=_text(payload.get("template_id"), "chain_path") or "chain_path",
        canonical_question=_text(payload.get("canonical_question")),
        entities=tuple(entities),
        answer=answer,
        relations=tuple(relations),
        path_shape=tuple(path_shape),
        exclusions=tuple(_text(x) for x in (payload.get("exclusions") or []) if _text(x)),
        raw_text=raw_text,
    )
