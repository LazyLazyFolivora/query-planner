"""⑤ 执行策略：动作 -> 证据 + 真实成本。

  ToyKGExecutor   包裹玩具 KG（当前默认，用于跑通与消融）
  （预留）RealKGExecutor / LiteratureExecutor / ModuleExecutor
                  接入真实 KG、文献检索、A/E 模块——只要实现同样三个方法
"""
from __future__ import annotations

from ..kg import ToyKG
from ..schema import Action, ActionCost, Evidence
from ..state_construction import PolicyState


class ToyKGExecutor:
    name = "toy_kg"

    def __init__(self, kg: ToyKG) -> None:
        self.kg = kg

    def legal_actions(self, state: PolicyState) -> list[Action]:
        return self.kg.legal_actions(state)

    def estimated_cost(self, action: Action) -> ActionCost:
        return action.cost

    def execute(self, action: Action, step: int) -> tuple[list[Evidence], ActionCost, str]:
        return self.kg.execute(action, step)




class PrimeKGExecutor:
    """把 PrimeKG 适配层包成 ActionExecutor。"""

    name = "primekg"

    def __init__(self, kg) -> None:
        self.kg = kg

    def legal_actions(self, state: PolicyState) -> list[Action]:
        return self.kg.legal_actions(state)

    def estimated_cost(self, action: Action) -> ActionCost:
        return action.cost

    def execute(self, action: Action, step: int) -> tuple[list[Evidence], ActionCost, str]:
        return self.kg.execute(action, step)


EXECUTORS = {ToyKGExecutor.name: ToyKGExecutor, PrimeKGExecutor.name: PrimeKGExecutor}