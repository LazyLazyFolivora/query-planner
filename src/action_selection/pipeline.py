"""第④阶段的门面：把候选动作打分与挑选的实现收敛成一个公开入口。

形态说明
--------
和第③阶段一样，④ 目前**没有需要拆出来的独立子流程**，门面是纯 re-export。

但三个实现的内部结构差异比③大得多，这是 ④ 值得单独记一笔的地方：

    HeuristicSelector     规则：四维手工打分（BioWeave 式）
    LinearModelSelector   线性基线：效用头 - λ(剩余预算) × 成本预测头
    ModelSelector         模型（默认）：状态-动作联合 MLP，共享 trunk 出两个 head

    LinearModelSelector / ModelSelector 的共同骨架是：
        _joint_features / features  ->  scores(效用 - λ·成本)  ->  argmax  ->  Selection

也就是说 ④ 隐含一条「特征 -> 打分 -> 取最大」的固定流水线，只是不值得为一个实现套一层类。
等 ModelSelector 的效用/成本双头需要独立校准或接 RL policy 时，这里就是拆分点。

主流程怎么用它::

    selection = self.pipe.selector.select(state, candidates_of(state))   # pipeline/runner.py

    selection.action is None  ->  没有可选动作，blocked_reason 说明被什么挡住了
                                 （无可行动作 / 预算耗尽），runner 记为 insufficient

轨迹采集复用同一套动作特征::

    from .pipeline import action_feature_vector
    action_feature_vector(action, state).tolist()

实现落点在 `selector.py`；新增一种实现 = 加一个类 + 挂进 SELECTORS，
runner 与 CLI 都不用改。
"""
from __future__ import annotations

from ..common.stage_contracts import ActionSelector, Selection
from .selector import (
    FEATURE_NAMES,
    KINDS,
    N_FEAT,
    SELECTORS,
    HeuristicSelector,
    LinearModelSelector,
    ModelSelector,
    action_features,
    action_feature_vector,
)

__all__ = [
    # 契约
    "ActionSelector",
    "Selection",
    # 注册表
    "SELECTORS",
    # 实现
    "HeuristicSelector",
    "LinearModelSelector",
    "ModelSelector",
    # 特征（轨迹采集复用）
    "action_features",
    "action_feature_vector",
    "FEATURE_NAMES",
    "KINDS",
    "N_FEAT",
]
