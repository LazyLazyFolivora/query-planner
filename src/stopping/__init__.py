"""第③阶段 终止判断：给定当前状态，决定"还要不要继续检索"。

    输入：PolicyState（② 状态整合的产物）
    输出：(Decision, reason)
          Decision ∈ {CONTINUE, COMPLETE, INSUFFICIENT}
          reason 是人类可读的停止理由，会写进轨迹

四种实现（默认 model = HierarchicalModelStop 三态头）：
    rule     RuleStop              纯规则：够不够 / 有没有路 / 预算耗尽
    model    HierarchicalModelStop 可训练三态分类头（默认）
    flat     FlatModelStop         消融：单头二分类
    naive    NaiveTargetStop       消融：只看目标是否已命中

包内分工：
    pipeline.py  公开门面（re-export），调用方只认这个入口
    stop.py      实现的唯一落点

主流程：
    decision, reason = self.pipe.stop.decide(state)     # pipeline/runner.py
"""
from __future__ import annotations

from .pipeline import (
    DEFAULT_COMPLETE_THRESHOLD,
    NO_PROGRESS_LIMIT,
    STOP_RULES,
    FlatModelStop,
    HierarchicalModelStop,
    NaiveTargetStop,
    RuleStop,
    StopRule,
)

__all__ = [
    "StopRule",
    "STOP_RULES",
    "RuleStop",
    "FlatModelStop",
    "HierarchicalModelStop",
    "NaiveTargetStop",
    "DEFAULT_COMPLETE_THRESHOLD",
    "NO_PROGRESS_LIMIT",
]
