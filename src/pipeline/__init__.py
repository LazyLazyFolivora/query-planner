"""Pipeline 运行时：装配 + 主循环。

    assembly.py  阶段注册表、默认值、build_pipeline()——决定"跑哪套实现"
    runner.py    Session / run_question()——主循环本身，不含任何具体算法

主循环（runner.py）：

    ①解析一次 → 每轮：②状态整合 → ③终止判断 → ⑤枚举候选 → ④动作选择
                 → ⑤执行 → ⑥记账 → ⑦更新证据 → 回到 ②
"""
from __future__ import annotations

from .assembly import DEFAULTS, PARSERS, REGISTRY, STEPS, Pipeline, build_pipeline
from .runner import Session, run_question

__all__ = [
    "Pipeline", "build_pipeline", "REGISTRY", "DEFAULTS", "STEPS", "PARSERS",
    "Session", "run_question",
]