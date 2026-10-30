"""② 状态构建阶段：把"我现在知道什么"汇总成一份确定性快照。

对应主流程的第二步。每一步开头调用一次，产出一个 PolicyState，
交给 ③ 终止判断与 ④ 动作选择读取。**本阶段不含 LLM，也不含深度学习模型。**

    ① 解析结果 ─┐
    ⑦ 证据容器 ─┤
    ⑥ 预算账本 ─┼─► pipeline.build_state() ──► PolicyState ──┐
    ⑦ 动作历史 ─┤                                       │           │
                 └───────────────────────────────────────┘           │
                                                                          │
    ⑤ 候选动作 ─┘（以回调注入，第⑦步在函数内部一并完成）

模块分工：

    models.py         PolicyState 快照结构（字段与约定），无逻辑
    pipeline.py       单一入口：build_state() 的函数体就是② 的七步流程
    coverage.py       证据视图与图算法（邻接表 / 路径存在 / 解释性路径 / 关系链匹配）
    sufficiency.py    四项可观测检查 -> 充分性分数（三态：passed/failed/not_applicable）
    budget.py         预算快照（⑥ 账本的只读投影）
    progress.py       动作历史聚合（最近动作、连续无进展步数）
    action_space.py   合法动作数 / 可负担动作数
    lexicon_scan.py   裸文本扫图谱（共用底座，① 旧解析器仍在用）

注意：状态**编码**（PolicyState -> 定长向量）不在本包，
      是是 StateEncoder 的事，见 strategies/encoder.py。
"""
from __future__ import annotations

from .action_space import affordable_count, annotate_action_space, is_affordable
from .budget import BudgetSnapshot
from .pipeline import (
    build_state,
    candidates_of,
    entity_type_groups,
    known_entity_list,
    relation_group_list,
)
from .coverage import (
    any_path_exists,
    build_graph,
    evidence_passages,
    evidence_triples,
    find_explanatory_path,
    matches_relation_chain,
    node_matches_target,
)
from .lexicon_scan import find_entities, find_relation_keywords, parse_question
from .models import KGView, PolicyState
from .progress import (
    NO_PROGRESS_LIMIT,
    RECENT_WINDOW,
    is_stalled,
    no_progress_streak,
    recent_actions,
)
from .sufficiency import (
    CHECK_CAUSAL_CHAIN,
    CHECK_FAILED,
    CHECK_IDS,
    CHECK_LABELS,
    CHECK_NOT_APPLICABLE,
    CHECK_PASSED,
    CHECK_PATH,
    CHECK_REQUIRED_RELATIONS,
    CHECK_TEXT_SUPPORT,
    observable_sufficiency,
    observable_sufficiency_status,
)

__all__ = [
    # 主入口
    "build_state",
    "candidates_of",
    "annotate_action_space",
    "budget_snapshot",
    # 结构
    "PolicyState",
    "KGView",
    # 派生量
    "entity_type_groups",
    "relation_group_list",
    "known_entity_list",
    # 证据视图与图算法
    "evidence_triples",
    "evidence_passages",
    "build_graph",
    "any_path_exists",
    "find_explanatory_path",
    "matches_relation_chain",
    "node_matches_target",
    # 充分性
    "observable_sufficiency",
    "observable_sufficiency_status",
    "CHECK_IDS",
    "CHECK_LABELS",
    "CHECK_PATH",
    "CHECK_REQUIRED_RELATIONS",
    "CHECK_CAUSAL_CHAIN",
    "CHECK_TEXT_SUPPORT",
    "CHECK_PASSED",
    "CHECK_FAILED",
    "CHECK_NOT_APPLICABLE",
    # 预算
    "BudgetSnapshot",
    # 进度
    "recent_actions",
    "no_progress_streak",
    "is_stalled",
    "RECENT_WINDOW",
    "NO_PROGRESS_LIMIT",
    # 动作空间
    "is_affordable",
    "affordable_count",
    # 裸文本扫图谱
    "find_entities",
    "find_relation_keywords",
    "parse_question",
]
