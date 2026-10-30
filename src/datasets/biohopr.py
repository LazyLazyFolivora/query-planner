"""BioHopR 数据加载器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import Question

DEFAULT_BUDGET = {
    "relation_access": 40,
    "tool_calls": 25,
    "tokens": 12000,
}


@dataclass(frozen=True)
class BioHopRRecord:
    """BioHopR 原始记录的最小结构化视图。"""

    qid: str
    question: str
    answer: tuple[str, ...]
    hop1: str
    hop2: str
    relation_hop1: str
    relation_hop2: str
    hop1_type: str
    hop2_type: str
    target_type: str

    @classmethod
    def from_raw(cls, raw: dict[str, Any], index: int) -> "BioHopRRecord":
        answers = raw.get("answer", [])
        if isinstance(answers, str):
            answers = [answers]
        question = (
            raw.get("hop2_question_multi")
            or raw.get("hop2_question")
            or raw.get("prompt")
            or ""
        ).strip()
        return cls(
            qid=f"biohopr_{index:05d}",
            question=question,
            answer=tuple(str(item) for item in answers),
            hop1=str(raw.get("hop1", "")),
            hop2=str(raw.get("hop2", "")),
            relation_hop1=str(raw.get("relation_hop1", "")),
            relation_hop2=str(raw.get("relation_hop2", "")),
            hop1_type=str(raw.get("hop1_type", "")),
            hop2_type=str(raw.get("hop2_type", "")),
            target_type=str(raw.get("target_type", "")),
        )

    def to_question(
        self,
        *,
        budget: dict[str, int] | None = None,
        max_steps: int = 10,
        sufficiency_threshold: float = 1.0,
    ) -> Question:
        """转成主流程使用的 Question；gold answer 不进入 PolicyState。"""
        return Question(
            qid=self.qid,
            question=self.question,
            qtype="chain",
            budget=dict(budget or DEFAULT_BUDGET),
            requirement_groups=[],
            max_steps=max_steps,
            sufficiency_threshold=sufficiency_threshold,
            note=(
                f"{self.relation_hop2} | "
                f"{self.hop2_type}->{self.hop1_type}->{self.target_type}"
            ),
        )


def load_biohopr(path: str | Path, limit: int | None = None) -> list[BioHopRRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("BioHopR.json 顶层应为 list")
    records = [BioHopRRecord.from_raw(raw, i) for i, raw in enumerate(payload)]
    return records[:limit] if limit is not None else records
