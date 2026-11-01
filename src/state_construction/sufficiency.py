"""可观测充分性信号：从已有证据派生"还差什么"。

这一层是纯确定性图算法 + 集合运算，不含 LLM，也不含深度学习模型。
产物是一组带三态（passed / failed / not_applicable）的检查项状态，
③ 终止判断与 ④ 动作选择都只读这一份信号。

四项检查（与第一阶段 EvidenceSpec.checks 同名）：
    path_exists              seed->target 是否已有证据路径
    required_relations_covered  必需关系组 / 关系链是否已覆盖
    causal_chain              是否存在 >=2 跳且含因果关系的路径（仅机制型问题适用）
    text_support              因果链上的三元组是否有文段支撑（仅机制型问题适用）
"""
from __future__ import annotations

from typing import Sequence

from ..constants import relation_group
from ..schema import Triple
from .coverage import (
    any_path_exists,
    evidence_passages,
    evidence_triples,
    find_explanatory_path,
    matches_relation_chain,
)

CHECK_PATH = "path_exists"
CHECK_REQUIRED_RELATIONS = "required_relations_covered"
CHECK_CAUSAL_CHAIN = "causal_chain"
CHECK_TEXT_SUPPORT = "text_support"

CHECK_IDS: tuple[str, ...] = (
    CHECK_PATH,
    CHECK_REQUIRED_RELATIONS,
    CHECK_CAUSAL_CHAIN,
    CHECK_TEXT_SUPPORT,
)

CHECK_LABELS: dict[str, str] = {
    CHECK_PATH: "seed->target 存在证据路径",
    CHECK_REQUIRED_RELATIONS: "必需关系组已全部覆盖",
    CHECK_CAUSAL_CHAIN: "存在 >=2 跳的因果链",
    CHECK_TEXT_SUPPORT: "因果链有文本证据支撑",
}

CHECK_PASSED = "passed"
CHECK_FAILED = "failed"
CHECK_NOT_APPLICABLE = "not_applicable"


def observable_sufficiency_status(state) -> tuple[tuple[str, str], ...]:
    """返回四项检查的显式状态，固定包含 not_applicable。

    状态取值：
      passed          已满足
      failed          适用但尚未满足
      not_applicable 该检查不适用于当前问题（例如非机制型问题的因果链检查）
    """
    external = getattr(state, "external_checks", None)
    if external is not None:
        return tuple(external)
    triples = evidence_triples(state.evidence)
    statuses: list[tuple[str, str]] = []

    # ① 可达性：起点到终点是否已经连通
    path_ok = any_path_exists(triples, state.seeds, state.targets, state.target_types)
    statuses.append((CHECK_PATH, CHECK_PASSED if path_ok else CHECK_FAILED))

    # ② 关系覆盖：优先用精确关系链，其次用粗粒度关系组
    if state.required_relation_chain:
        chain_ok = matches_relation_chain(triples, state.seeds, state.required_relation_chain)
        statuses.append((CHECK_REQUIRED_RELATIONS, CHECK_PASSED if chain_ok else CHECK_FAILED))
    elif state.required_relation_groups:
        covered = {relation_group(t.relation) for t in triples}
        missing = [g for g in state.required_relation_groups if g not in covered]
        statuses.append((
            CHECK_REQUIRED_RELATIONS,
            CHECK_PASSED if not missing else CHECK_FAILED,
        ))
    else:
        statuses.append((CHECK_REQUIRED_RELATIONS, CHECK_NOT_APPLICABLE))

    # ③④ 因果链与文本支撑：只对机制型问题适用
    if state.qtype == "mechanism":
        path = find_explanatory_path(triples, state.seeds, state.targets, state.target_types)
        statuses.append((CHECK_CAUSAL_CHAIN, CHECK_PASSED if path is not None else CHECK_FAILED))

        path_keys = {t.key() for t in (path or [])}
        supported = False
        for p in evidence_passages(state.evidence):
            for t in p.supports:
                if t.key() in path_keys:
                    supported = True
        statuses.append((CHECK_TEXT_SUPPORT, CHECK_PASSED if supported else CHECK_FAILED))
    else:
        statuses.append((CHECK_CAUSAL_CHAIN, CHECK_NOT_APPLICABLE))
        statuses.append((CHECK_TEXT_SUPPORT, CHECK_NOT_APPLICABLE))

    return tuple(statuses)


def observable_sufficiency(state) -> tuple[float, list[tuple[str, bool]]]:
    """策略侧充分性分数：只对 applicable 的检查求通过比例。

    返回格式保持旧接口：`(score, [(check_label, passed), ...])`。
    """
    statuses = observable_sufficiency_status(state)
    checks: list[tuple[str, bool]] = []
    for check_id, status in statuses:
        if status == CHECK_NOT_APPLICABLE:
            continue
        label = CHECK_LABELS.get(check_id, check_id)
        if check_id == CHECK_REQUIRED_RELATIONS and state.required_relation_groups:
            label = label + ": " + "/".join(state.required_relation_groups)
        checks.append((label, status == CHECK_PASSED))

    score = sum(1 for _, ok in checks if ok) / max(len(checks), 1)
    return score, checks

