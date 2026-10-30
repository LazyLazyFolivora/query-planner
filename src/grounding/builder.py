"""Build backend-independent evidence requirements from query intents."""
from __future__ import annotations

from ..common.contracts import (
    EVIDENCE_SHAPE_COUNT,
    EVIDENCE_SHAPE_PATH,
    EVIDENCE_SHAPE_RELATION_SET,
    EVIDENCE_SHAPE_SINGLE_FACT,
    EVIDENCE_SHAPE_SUBGRAPH,
    RELATION_PREFERRED,
    EvidenceCheck,
    EvidenceSpec,
    QueryIntent,
)


_TEMPLATE_SPECS = {
    # 因果解释，问题是：“为什么会这样”
    "mechanism_path": (
        EVIDENCE_SHAPE_PATH,
        (
            "path_exists",
            "required_relations",
            "causal_chain",
            "text_support",
        ),
    ),
    # 路径追踪，问题是：“X和Y有什么关系”
    "chain_path": (
        EVIDENCE_SHAPE_PATH,
        ("path_exists", "required_relations"),
    ),
    # 集合枚举，问题是：”X与哪些药物相互作用会Y“
    "interaction_set": (
        EVIDENCE_SHAPE_SUBGRAPH,
        ("path_exists", "required_relations", "answer_type", "set_closure"),
    ),
    # 属性查询，问题是：”X的禁忌症是什么“
    "attribute_lookup": (
        EVIDENCE_SHAPE_SINGLE_FACT,
        ("fact_exists", "answer_type"),
    ),
    # 计数，答案是一个数字
    "counting": (
        EVIDENCE_SHAPE_COUNT,
        ("fact_exists", "set_closure", "count_exact"),
    ),
    # 是非判断
    "yes_no": (
        EVIDENCE_SHAPE_SINGLE_FACT,
        ("witness_exists",),
    ),
    # 对比：两个/多个实体的同一属性横向对比
    "comparison": (
        EVIDENCE_SHAPE_RELATION_SET,
        ("fact_exists", "required_relations"),
    ),
    # 全景汇总，问题是：“布洛芬的完整作用机制”
    "aggregation": (
        EVIDENCE_SHAPE_SUBGRAPH,
        ("required_relations", "aggregation_complete"),
    ),
}


class EvidenceSpecBuilder:
    def build(self, intent: QueryIntent) -> EvidenceSpec:
        shape, check_ids = _TEMPLATE_SPECS.get(
            intent.template_id,
            (EVIDENCE_SHAPE_PATH, ("path_exists",)),
        )
        hard = tuple(c for c in intent.relation_constraints if c.strength != RELATION_PREFERRED)
        soft = tuple(c for c in intent.relation_constraints if c.strength == RELATION_PREFERRED)
        checks = tuple(
            EvidenceCheck(
                check_id=check_id,
                required=True,
                params=(
                    ("template_id", intent.template_id),
                    ("reasoning_mode", intent.reasoning_mode),
                    ("answer_mode", intent.answer_mode),
                ),
            )
            for check_id in check_ids
        )
        return EvidenceSpec(
            shape=shape,
            hard_constraints=hard,
            soft_constraints=soft,
            checks=checks,
            answer_slots=intent.answer_slots,
        )