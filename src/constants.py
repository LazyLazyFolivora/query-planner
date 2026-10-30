"""领域常量与关系先验。

单独成文件，避免 state.py 与 kg.py 互相 import 造成循环依赖。
"""
from __future__ import annotations

CAUSAL_RELATIONS = ("导致", "引起", "加重", "表现为", "机制", "相关")
MECHANISM_RELATIONS = CAUSAL_RELATIONS + ("药理作用",)

# 关系先验（占位：将来替换为可学习的关系打分模块）
RELATION_PRIORITY = (
    "导致", "引起", "加重", "机制", "表现为",
    "药理作用", "相互作用", "禁忌症", "不良反应", "常见不良反应", "适用症", "相关",
)

MAX_PATH_HOPS = 3


def relation_rank(relation: str) -> int:
    """关系先验排序；未收录的关系排在最后。"""
    try:
        return RELATION_PRIORITY.index(relation)
    except ValueError:
        return len(RELATION_PRIORITY)


# ============================================================ 粗粒度关系分组
#
# 目的：
#   1) 降维——细粒度关系可能有 30+ 种，分组后只有 6 维；
#   2) 缓解"语义相近的关系被当成正交"——"引起"与"导致"同属 causal 组，
#      在 multi-hot 里落在同一位，模型不会以为它们毫不相干。
#
# 匹配方式：按子串匹配，先命中先用；都没命中归入 "other"。
# 中英文模式都写了，便于从玩具 KG 切到 PrimeKG。
RELATION_GROUPS: tuple[str, ...] = (
    "causal", "property", "interaction", "adverse", "associative", "other",
)

_RELATION_GROUP_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("causal", ("导致", "引起", "加重", "表现", "机制", "cause", "induce", "lead_to", "result")),
    ("property", ("药理", "适用", "适应", "禁忌", "indication", "contraindic", "effect_of")),
    ("interaction", ("相互作用", "interact", "combination")),
    ("adverse", ("不良反应", "副作用", "adverse", "side_effect", "toxicity")),
    ("associative", ("相关", "关联", "associat", "related", "similar")),
)


def relation_group(relation: str) -> str:
    """把具体关系名映射到粗粒度组；未命中归入 other。"""
    name = (relation or "").lower()
    for group, keys in _RELATION_GROUP_PATTERNS:
        if any(k.lower() in name for k in keys):
            return group
    return "other"


# ============================================================ 粗粒度实体类型分组
#
# 4 类（对齐 PrimeKG 的节点类型），未命中时全部置 0（表示"类型未知"）。
ENTITY_TYPE_GROUPS: tuple[str, ...] = ("drug", "disease", "effect", "gene")

_ENTITY_TYPE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("drug", ("drug", "药", "compound", "chemical")),
    ("disease", ("disease", "病", "indication", "contraindication", "syndrome")),
    ("effect", ("effect", "phenotype", "mechanism", "表型", "机制", "adverse", "反应", "中毒")),
    ("gene", ("gene", "protein", "基因", "蛋白")),
)


def entity_type_group(entity_type: str) -> str:
    """把具体实体类型映射到粗粒度组；未命中返回空串。"""
    name = (entity_type or "").lower()
    for group, keys in _ENTITY_TYPE_PATTERNS:
        if any(k.lower() in name for k in keys):
            return group
    return ""