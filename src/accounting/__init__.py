"""第⑥阶段 记账：所有动作成本必须经此过账，余额是第③④阶段的硬约束。

    输入：ActionCost（真实成本，可能带权重）+ 动作标签
    输出：更新后的余额；并提供 remaining / can_afford / spent / entries 查询

实现：
    budget    BudgetLedger   多维预算账本（relation_access / tool_calls / tokens）
    （预留）WeightedLedger  若将来需要把多维成本折算成单一货币
"""
from __future__ import annotations

from ..common.stage_contracts import BudgetLedger as BudgetLedgerProtocol
from .ledger import DIMENSIONS, LEDGERS, BudgetLedger

__all__ = [
    "BudgetLedger",
    "BudgetLedgerProtocol",
    "LEDGERS",
    "DIMENSIONS",
]
