"""预算快照：⑥ 记账的只读投影。

状态构建不修改预算，只读取账本的当前余额与上限。
预算是否可负担由 ④ 动作选择 / ⑤ 执行层关心，这里只提供数据。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BudgetSnapshot:
    """某一时刻的预算视图：上限、余额、已花比例。"""

    limits: dict[str, int] = field(default_factory=dict)
    remaining: dict[str, int] = field(default_factory=dict)

    def exhausted(self) -> bool:
        """所有维度都已归零（或上限为 0）即视为预算耗尽。"""
        if not self.limits:
            return True
        return all(self.remaining.get(d, 0) <= 0 for d in self.limits)

    def can_afford(self, cost: dict[str, int]) -> bool:
        """按当前余额判断某个成本是否还负担得起（④ 动作选择会用到）。"""
        return all(cost.get(d, 0) <= self.remaining.get(d, 0) for d in cost)
