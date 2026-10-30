"""阶段契约层：把主流程的**每一步**都定义成可替换的接口。

七个步骤与对应接口：

    ①  解析       QuestionParser   question  ->  ParsedQuestion
    ②  状态编码   StateEncoder     PolicyState -> 向量
    ③  终止判断   StopRule         PolicyState -> (Decision, reason)
    ④  动作选择   ActionSelector   PolicyState + 候选动作 -> Selection
    ⑤  执行       ActionExecutor   动作 -> (证据, 真实成本)
    ⑥  记账       BudgetLedger     成本 -> 余额
    ⑦  更新证据   EvidenceUpdater  证据 -> 状态

约定：
  * 每个接口都有 **模型实现（默认优先）** 与 **规则实现（基线）**，见同目录各模块；
  * runner 只依赖本文件的协议，不依赖任何具体实现；
  * 新增一种实现 = 在对应模块里加一个类 + 注册，runner 与 CLI 都不用改。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from ..schema import Action, ActionCost, Decision, Evidence, HistoryItem
from .contracts import EvidenceSpec, QueryIntent
from ..state_construction import PolicyState


# ---------------------------------------------------------------- ① 解析

@dataclass(frozen=True)
class ParsedQuestion:
    "解析结果。字段参考 Condition-Gated 的 parse(q)={E,C,N} 与 BioWeave 的推理算子。"
    seeds: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    required_relations: tuple[str, ...] = ()
    excluded_entities: tuple[str, ...] = ()
    operator: str = "chain"          # chain | intefsection | counting | aggfegation
    target_types: tuple[str, ...] = ()        # 目标实体粗粒度类型组（用于类型目标问题）
    required_relation_chain: tuple[tuple[str, ...], ...] = ()  # 每跳允许的关系名集合
    unlinked_mentions: tuple[str, ...] = ()   # 未能链接到图谱节点的提及（-> insufficient 的来源）
    query_intent: QueryIntent | None = None
    evidence_spec: EvidenceSpec | None = None
    parse_status: str = "legacy"


@runtime_checkable
class QuestionParser(Protocol):
    name: str
    def parse(self, question: str, kg) -> ParsedQuestion: ...


# ---------------------------------------------------------------- ② 状态编码

@runtime_checkable
class StateEncoder(Protocol):
    name: str
    dim: int
    def encode(self, state: PolicyState) -> np.ndarray: ...


# ---------------------------------------------------------------- ③ 终止判断

@runtime_checkable
class StopRule(Protocol):
    name: str
    def decide(self, state: PolicyState) -> tuple[Decision, str]: ...


# ---------------------------------------------------------------- ④ 动作选择

@dataclass(frozen=True)
class Selection:
    action: Action | None = None
    blocked_reason: str = ""


@runtime_checkable
class ActionSelector(Protocol):
    name: str
    def select(self, state: PolicyState, candidates: Sequence[Action]) -> Selection: ...


# ---------------------------------------------------------------- ⑤ 执行

@runtime_checkable
class ActionExecutor(Protocol):
    name: str
    def legal_actions(self, state: PolicyState) -> list[Action]: ...
    def execute(self, action: Action, step: int) -> tuple[list[Evidence], ActionCost, str]: ...


# ---------------------------------------------------------------- ⑥ 记账

@runtime_checkable
class BudgetLedger(Protocol):
    name: str
    def remaining(self) -> dict[str, int]: ...
    def can_afford(self, cost: ActionCost) -> bool: ...
    def charge(self, cost: ActionCost, label: str) -> None: ...
    def spent(self) -> ActionCost: ...
    def entries(self) -> list[dict]: ...


# ---------------------------------------------------------------- ⑦ 更新证据

@dataclass
class EvidenceStore:
    """可变证据容器：⑦ 更新策略作用在其上。"""
    evidence: list[Evidence] = field(default_factory=list)
    looked_up: set[str] = field(default_factory=set)
    expanded: set[tuple[str, str]] = field(default_factory=set)
    tried_paths: set[tuple[str, str]] = field(default_factory=set)
    tried_passages: bool = False
    relations_seen: dict[str, tuple[str, ...]] = field(default_factory=dict)
    history: list[HistoryItem] = field(default_factory=list)


@runtime_checkable
class EvidenceUpdater(Protocol):
    name: str
    def update(self, store: EvidenceStore, action: Action, new_evidence: Sequence[Evidence],
               cost: ActionCost | None = None) -> None: ...

