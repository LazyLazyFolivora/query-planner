"""PolicyState：策略在每一步"能看到什么"的快照结构。

严格约定：PolicyState 内不含任何 gold 信息。
证据需求（gold）只出现在 evaluator 中，用于事后打分。

字段分四组：
  * 问题与解析结果  —— 来自 ① 解析，整轮不变；
  * 证据与覆盖痕迹  —— 来自 ⑦ 更新证据，每步变化；
  * 预算与进度      —— 来自 ⑥ 记账 + 动作历史，每步变化；
  * 动作空间摘要    —— 由 ⑤ 枚举候选动作后回填（见 action_space.py）。

本模块只依赖 constants / contracts / schema，不 import kg.py，
这样 kg.py 可以反向 import 本包而不产生循环依赖。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..common.contracts import EvidenceSpec, QueryIntent
from ..schema import Action, Evidence


class KGView(Protocol):
    """状态构建需要的知识图谱最小视图。"""

    aliases: dict[str, str]

    def entity_names(self) -> list[str]: ...
    def relation_names(self) -> list[str]: ...


@dataclass
class PolicyState:
    qid: str
    question: str
    qtype: str
    seeds: tuple[str, ...]
    targets: tuple[str, ...]
    relation_keywords: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    known_entities: tuple[str, ...]
    relations_seen: dict[str, tuple[str, ...]]
    looked_up: frozenset[str]
    expanded: frozenset[tuple[str, str]]
    tried_paths: frozenset[tuple[str, str]]
    tried_passages: bool
    remaining: dict[str, int]
    budget_limits: dict[str, int]
    step_index: int
    sufficiency_threshold: float
    max_steps: int

    # --- 供状态编码使用 ---
    operator: str = "chain"
    excluded_entities: tuple[str, ...] = ()
    recent_actions: tuple[str, ...] = ()        # 最近 K 个动作类型（时间顺序，最新在后）
    no_progress_streak: int = 0                 # 连续无新增证据步数
    target_types: tuple[str, ...] = ()          # 目标实体粗粒度类型组
    required_relation_groups: tuple[str, ...] = ()  # 必需关系的粗粒度组（去重）
    required_relation_chain: tuple[tuple[str, ...], ...] = ()  # 精确关系链：每跳允许关系集合
    query_intent: QueryIntent | None = None
    evidence_spec: EvidenceSpec | None = None
    parse_status: str = "legacy"
    unlinked_mentions: tuple[str, ...] = ()

    # 当前状态下的合法动作数与可负担动作数。None 表示尚未由执行层枚举；
    # runner 会在调用 stop 前通过 action_space.annotate_action_space 填充为 0 或正整数。
    legal_action_count: int | None = None
    affordable_action_count: int | None = None

    # ⑦ 枚举出来的候选动作本体。facade 一次枚举后缓存下来，
    # 供 ④ 动作选择复用——这样 legal_actions 每步只会被调用一次。
    # 由 state_construction.pipeline.build_state 写入，其他地方只读。
    candidate_actions: tuple[Action, ...] | None = None
