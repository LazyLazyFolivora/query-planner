"""数据模型（第一版占位设计，对应 D 技术路线 4.1 节的契约）。

重要约定：
  * PolicyState 是"策略可见"的信息，不含 gold；
  * Question.requirement_groups 是"评测可见"的 gold，只能用于打分；
  * ActionCost 是异质成本：主预算 = relation_access，辅助 = tool_calls / tokens。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Decision(str, Enum):
    CONTINUE = "continue"
    COMPLETE = "complete"
    INSUFFICIENT = "insufficient"


class StopReason(str, Enum):
    SUFFICIENT = "sufficient_evidence"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_LEGAL_ACTION = "no_legal_action"
    MAX_STEPS = "max_steps_reached"
    NO_PROGRESS = "no_progress"
    LOW_CONTINUATION_VALUE = "low_continuation_value"
    TARGET_FOUND = "target_found_naive"
    NONE = "-"


@dataclass(frozen=True)
class Triple:
    """原子 KG 证据单元。字段设计参考 SSE-Bio 的 KnowledgeTriplet。

    身份由 (head, relation, tail) 决定；`metadata` 不参与相等性与哈希
    （compare=False），因此两个只有元数据不同的三元组被视为同一条证据。
    """

    head: str
    relation: str
    tail: str
    source: str = ""                       # 来源库/文档（如 "药品说明书" / "DrugBank"）
    confidence: float = 1.0                # 置信度
    source_id: str = ""                    # 来源内的记录 ID
    head_type: str = ""                    # head 的实体类型（来自图谱）
    tail_type: str = ""                    # tail 的实体类型
    metadata: dict[str, str] = field(default_factory=dict, compare=False)

    def key(self) -> tuple[str, str, str]:
        return (self.head, self.relation, self.tail)

    def text(self) -> str:
        """可读形式（日志与轨迹用）。"""
        return f"{self.head} -[{self.relation}]-> {self.tail}"

    def linearize(self) -> str:
        """线性化成文本，供嵌入模型编码（照 SSE-Bio 的 to_text 思路）。"""
        return f"<{self.head}; {self.relation}; {self.tail}>"

    def meta(self, key: str, default: str = "") -> str:
        return self.metadata.get(key, default)


@dataclass(frozen=True)
class Passage:
    pid: str
    text: str
    supports: tuple[Triple, ...] = ()
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class Evidence:
    eid: str
    kind: str                 # relation_list | triple | passage
    action_id: str
    step: int
    entity: str = ""
    relations: tuple[str, ...] = ()
    triple: Triple | None = None
    passage: Passage | None = None

    def label(self) -> str:
        if self.kind == "relation_list":
            return f"relations({self.entity}) = {list(self.relations)}"
        if self.triple is not None:
            return self.triple.text()
        if self.passage is not None:
            return f"passage({self.passage.pid})"
        return self.eid


@dataclass(frozen=True)
class ActionCost:
    relation_access: int = 0
    tool_calls: int = 0
    tokens: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "relation_access": self.relation_access,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
        }

    def __add__(self, other: "ActionCost") -> "ActionCost":
        return ActionCost(
            self.relation_access + other.relation_access,
            self.tool_calls + other.tool_calls,
            self.tokens + other.tokens,
        )

    def text(self) -> str:
        return f"ra={self.relation_access} calls={self.tool_calls} tok={self.tokens}"


@dataclass(frozen=True)
class HistoryItem:
    """一步执行记录，用于状态编码的历史块（RareAgent 的 stagnation 思路）。"""
    kind: str              # entity_lookup | relation_expand | bounded_path | passage_retrieve
    yielded: int           # 本次新增的"新"三元组条数（去重后）
    cost_ra: int           # 本次消耗的关系访问数


ZERO_COST = ActionCost()


@dataclass(frozen=True)
class Action:
    aid: str
    kind: str                 # entity_lookup | relation_expand | bounded_path | passage_retrieve
    args: tuple[str, ...]
    cost: ActionCost

    def describe(self) -> str:
        return f"{self.kind}({', '.join(self.args)})"


@dataclass(frozen=True)
class RequirementGroup:
    """一组可接受的证据：any_of 命中任意一条即算该组完成（多组可接受证据）。"""
    gid: str
    alternatives: tuple[Triple, ...]
    weight: float = 1.0

    def satisfied_by(self, keys: set[tuple[str, str, str]]) -> bool:
        return any(alt.key() in keys for alt in self.alternatives)


@dataclass
class Question:
    qid: str
    question: str
    qtype: str
    budget: dict[str, int]
    requirement_groups: list[RequirementGroup]
    max_steps: int = 12
    sufficiency_threshold: float = 0.99
    note: str = ""


@dataclass
class Step:
    index: int
    decision: Decision
    stop_reason: str
    action: Action | None
    cost: ActionCost
    remaining: dict[str, int]
    sufficiency: float
    new_evidence: list[Evidence] = field(default_factory=list)
    note: str = ""


@dataclass
class RunTrace:
    qid: str
    question: str
    policy: str
    steps: list[Step] = field(default_factory=list)
    final_decision: Decision = Decision.INSUFFICIENT
    final_reason: str = StopReason.NONE.value
    spent: ActionCost = ZERO_COST
    evidence_count: int = 0
    final_evidence: list = field(default_factory=list)
    transitions: list[dict] = field(default_factory=list)