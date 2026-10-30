"""动作空间摘要：本状态下的"还能做哪些动作"。

背景：③ 终止判断需要看到"现在还有没有路可走"。
      如果还没枚举候选动作就无法回答这个问题，所以主循环的顺序是：
          build_state()  ->  legal_actions()  ->  回填动作数  ->  stop.decide()

纯确定性统计：数个数 + 逐个比预算。不含 LLM，也不含深度学习模型。
"""
from __future__ import annotations

from typing import Sequence

from ..schema import Action


def is_affordable(action: Action, remaining: dict[str, int]) -> bool:
    """该动作的预计成本是否还在剩余预算内（逐维比较）。"""
    cost = action.cost.as_dict()
    return all(cost[dim] <= remaining.get(dim, 0) for dim in cost)


def affordable_count(actions: Sequence[Action], remaining: dict[str, int]) -> int:
    """可负担动作数：预算还够得着的动作有多少个。"""
    return sum(1 for action in actions if is_affordable(action, remaining))


def annotate_action_space(state, candidates: Sequence[Action]) -> None:
    """把候选动作数与可负担数回填进状态。

    状态是"两段式"构造的：先建主体（见 builder.build），枚举候选动作后再调本函数。
    """
    state.legal_action_count = len(candidates)
    state.affordable_action_count = affordable_count(candidates, state.remaining)
