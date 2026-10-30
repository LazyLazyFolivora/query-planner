"""状态构建（②）：实现已收拢到 state_construction 包。

本文件**保留仅为向后兼容**：历史上多处用 `from .state import PolicyState`，
现在统一改为 `from .state_construction import PolicyState`。
这里的每个名字都是 state_construction 里的同一个对象（不是副本），
所以两边混用也不会有类型不一致的问题。

新代码请直接用 state_construction。
"""
from __future__ import annotations

from .state_construction import (
    CHECK_CAUSAL_CHAIN,
    CHECK_FAILED,
    CHECK_IDS,
    CHECK_LABELS,
    CHECK_NOT_APPLICABLE,
    CHECK_PASSED,
    CHECK_PATH,
    CHECK_REQUIRED_RELATIONS,
    CHECK_TEXT_SUPPORT,
    KGView,
    PolicyState,
    any_path_exists,
    build_graph,
    evidence_passages,
    evidence_triples,
    find_entities,
    find_explanatory_path,
    find_relation_keywords,
    matches_relation_chain,
    node_matches_target,
    observable_sufficiency,
    observable_sufficiency_status,
    parse_question,
)

__all__ = [
    "PolicyState",
    "KGView",
    "CHECK_IDS",
    "CHECK_LABELS",
    "CHECK_PATH",
    "CHECK_REQUIRED_RELATIONS",
    "CHECK_CAUSAL_CHAIN",
    "CHECK_TEXT_SUPPORT",
    "CHECK_PASSED",
    "CHECK_FAILED",
    "CHECK_NOT_APPLICABLE",
    "evidence_triples",
    "evidence_passages",
    "build_graph",
    "any_path_exists",
    "find_explanatory_path",
    "matches_relation_chain",
    "node_matches_target",
    "observable_sufficiency",
    "observable_sufficiency_status",
    "find_entities",
    "find_relation_keywords",
    "parse_question",
]
