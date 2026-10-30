"""训练轨迹数据结构和 JSONL 序列化。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import RunTrace


@dataclass(frozen=True)
class Transition:
    qid: str
    step: int
    state_vector: tuple[float, ...] = ()
    decision: str = "continue"
    stop_reason: str = ""
    stop_probabilities: tuple[float, ...] = ()
    action: str | None = None
    action_kind: str = ""
    action_args: tuple[str, ...] = ()
    action_features: tuple[float, ...] = ()
    estimated_cost: dict[str, int] = field(default_factory=dict)
    actual_cost: dict[str, int] = field(default_factory=dict)
    new_evidence_ids: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "step": self.step,
            "state_vector": list(self.state_vector),
            "decision": self.decision,
            "stop_reason": self.stop_reason,
            "stop_probabilities": list(self.stop_probabilities),
            "action": self.action,
            "action_kind": self.action_kind,
            "action_args": list(self.action_args),
            "action_features": list(self.action_features),
            "estimated_cost": self.estimated_cost,
            "actual_cost": self.actual_cost,
            "new_evidence_ids": list(self.new_evidence_ids),
            "note": self.note,
        }


@dataclass
class Trajectory:
    qid: str
    question: str
    policy: str
    transitions: list[Transition] = field(default_factory=list)
    final_decision: str = ""
    final_reason: str = ""
    spent: dict[str, int] = field(default_factory=dict)
    evidence_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_run(cls, trace: RunTrace, metadata: dict[str, Any] | None = None) -> "Trajectory":
        meta = dict(metadata or {})
        meta["final_evidence"] = [e.label() for e in trace.final_evidence]
        transitions: list[Transition] = []
        for item in trace.transitions:
            transitions.append(Transition(
                qid=str(item.get("qid", trace.qid)),
                step=int(item.get("step", 0)),
                state_vector=tuple(float(x) for x in item.get("state_vector", [])),
                decision=str(item.get("decision", "continue")),
                stop_reason=str(item.get("stop_reason", "")),
                stop_probabilities=tuple(float(x) for x in (item.get("stop_probabilities") or [])),
                action=item.get("action"),
                action_kind=str(item.get("action_kind", "")),
                action_args=tuple(str(x) for x in item.get("action_args", [])),
                action_features=tuple(float(x) for x in item.get("action_features", [])),
                estimated_cost=dict(item.get("estimated_cost", {}) or {}),
                actual_cost=dict(item.get("actual_cost", {}) or {}),
                new_evidence_ids=tuple(str(x) for x in item.get("new_evidence_ids", [])),
                note=str(item.get("note", "")),
            ))
        return cls(
            qid=trace.qid,
            question=trace.question,
            policy=trace.policy,
            transitions=transitions,
            final_decision=trace.final_decision.value,
            final_reason=trace.final_reason,
            spent=trace.spent.as_dict(),
            evidence_count=trace.evidence_count,
            metadata=meta,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "question": self.question,
            "policy": self.policy,
            "final_decision": self.final_decision,
            "final_reason": self.final_reason,
            "spent": self.spent,
            "evidence_count": self.evidence_count,
            "metadata": self.metadata,
            "transitions": [t.to_dict() for t in self.transitions],
        }


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
