"""第⑤阶段 执行：把一个动作真正打到后端上。

    输入：Action + step
    输出：(Evidence 列表, 真实 ActionCost, note)
    另外 legal_actions(state) 给出当前状态下的合法动作集合（供第④阶段挑选）

实现：
    toy_kg    ToyKGExecutor   玩具知识图谱（默认，用于跑通与消融）
    primekg   PrimeKGExecutor 真实 PrimeKG
    （预留）RealKGExecutor / LiteratureExecutor / ModuleExecutor

这一层是唯一知道"证据从哪来"的地方，上层只看到 Evidence 与成本。
"""
from __future__ import annotations

from ..common.stage_contracts import ActionExecutor
from .executor import EXECUTORS, PrimeKGExecutor, ToyKGExecutor

__all__ = [
    "ActionExecutor",
    "EXECUTORS",
    "ToyKGExecutor",
    "PrimeKGExecutor",
]
