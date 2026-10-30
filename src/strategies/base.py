"""兼容门面：见 :mod:`src.strategies` 的说明，实现已迁移到各阶段包。"""

from __future__ import annotations

from ..common.stage_contracts import (
    ActionExecutor,
    ActionSelector,
    BudgetLedger,
    EvidenceStore,
    EvidenceUpdater,
    ParsedQuestion,
    QuestionParser,
    Selection,
    StateEncoder,
    StopRule,
)

__all__ = [
    "ParsedQuestion", "QuestionParser", "StateEncoder", "StopRule", "ActionSelector", "Selection",
    "ActionExecutor", "BudgetLedger", "EvidenceStore", "EvidenceUpdater",
]