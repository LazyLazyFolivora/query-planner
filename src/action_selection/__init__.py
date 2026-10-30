"""第④阶段 动作选择：从执行层给出的合法候选里，挑一个下一步做。

    输入：PolicyState + Sequence[Action]（合法候选动作）
    输出：Selection(action, blocked_reason)
          action 为 None 表示"没有可选动作"，
          blocked_reason 说明被什么挡住了（无可行动作 / 预算耗尽）

三种实现（默认 model = 状态-动作联合 MLP 效用/成本头）：
    heuristic  HeuristicSelector     规则：四维手工打分
    linear     LinearModelSelector   线性基线：效用头 − λ(剩余预算) × 成本预测头
    model      ModelSelector         模型实现（默认）

包内分工：
    pipeline.py  公开门面（re-export），调用方只认这个入口
    selector.py  实现的唯一落点

注意：⑤ 执行层的 `legal_actions(state)` 负责**枚举**候选，
本阶段只负责从已枚举的集合里**挑选**，两者不要混。
"""
from __future__ import annotations

from .pipeline import (
    FEATURE_NAMES,
    KINDS,
    N_FEAT,
    SELECTORS,
    ActionSelector,
    HeuristicSelector,
    LinearModelSelector,
    ModelSelector,
    Selection,
    action_features,
    action_feature_vector,
)

__all__ = [
    "ActionSelector",
    "Selection",
    "SELECTORS",
    "HeuristicSelector",
    "LinearModelSelector",
    "ModelSelector",
    "action_features",
    "action_feature_vector",
    "FEATURE_NAMES",
    "KINDS",
    "N_FEAT",
]
