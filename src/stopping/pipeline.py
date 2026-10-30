"""第③阶段的门面：把四种终止策略的实现收敛成一个公开入口。

这一层存在的理由是**形态对齐**——① 解析有 `ParsingPipeline`、
② 状态整合有 `build_state()`，③ 也该有一个稳定的 import 门面，
而不是让调用方去知道实现恰好落在 `stop.py` 这个文件名上。

形态说明
--------
③ 是七个阶段里**唯一没有内部子步骤**的阶段：
    ① 解析     LLM 改写 -> KG 链接 -> 模板查表（三步）
    ② 状态整合 coverage / sufficiency / progress / budget（七路派生量聚合）
    ③ 终止判断 state -> 特征 -> 两个 head -> argmax（一步到位）

所以门面是**纯 re-export**，不引入 `StoppingPipeline` 这层间接。
真正的拆分（充分性 -> 三态分类 -> 阈值校准）出现之前，这里保持薄。

主流程怎么用它::

    decision, reason = self.pipe.stop.decide(state)     # pipeline/runner.py

`stop.py` 仍是实现的唯一落点，本文件只固定公开 API 与扩展位：

    新增一种实现 = 在 stop.py 里加一个类 + 挂进 STOP_RULES，
    runner 与 CLI 都不用改。
"""
from __future__ import annotations

from ..common.stage_contracts import StopRule
from .stop import (
    DEFAULT_COMPLETE_THRESHOLD,
    NO_PROGRESS_LIMIT,
    STOP_RULES,
    FlatModelStop,
    HierarchicalModelStop,
    NaiveTargetStop,
    RuleStop,
)

__all__ = [
    # 契约
    "StopRule",
    # 注册表与扩展位
    "STOP_RULES",
    "DEFAULT_COMPLETE_THRESHOLD",
    "NO_PROGRESS_LIMIT",
    # 实现
    "RuleStop",
    "FlatModelStop",
    "HierarchicalModelStop",
    "NaiveTargetStop",
]
