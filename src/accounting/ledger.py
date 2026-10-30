"""⑥ 记账策略：所有动作成本必须经此记账。

  BudgetLedger   预算账本（多维：关系访问数 / 工具调用 / token）
  （预留）WeightedLedger   若将来需要把多维成本折算成单一货币
"""
from __future__ import annotations

from ..schema import ActionCost

DIMENSIONS = ("relation_access", "tool_calls", "tokens")


class BudgetLedger:
    name = "budget"

    def __init__(self, limits: dict[str, int]) -> None:
        self.limits = {d: int(limits.get(d, 0)) for d in DIMENSIONS}
        self._spent = {d: 0 for d in DIMENSIONS}
        self._entries: list[dict] = []

    def remaining(self) -> dict[str, int]:
        return {d: self.limits[d] - self._spent[d] for d in DIMENSIONS}

    def can_afford(self, cost: ActionCost) -> bool:
        c = cost.as_dict()
        return all(c[d] <= self.limits[d] - self._spent[d] for d in DIMENSIONS)

    def charge(self, cost: ActionCost, label: str) -> None:
        c = cost.as_dict()
        for d in DIMENSIONS:
            self._spent[d] += c[d]
        self._entries.append({
            "label": label,
            "cost": c,
            "cumulative": dict(self._spent),
            "remaining": self.remaining(),
        })

    def spent(self) -> ActionCost:
        return ActionCost(**self._spent)

    def entries(self) -> list[dict]:
        return list(self._entries)


LEDGERS = {BudgetLedger.name: BudgetLedger}