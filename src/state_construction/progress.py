"""进度信号：⑦ 动作历史聚合出来的"走到哪了、卡没卡"。

这一层是纯确定性统计（列表切片 + 从尾部计数），不含 LLM，
也不含深度学习模型。信号直接给 ③ 终止判断与 ② 状态编码使用。
"""
from __future__ import annotations

from typing import Sequence

from ..schema import HistoryItem

RECENT_WINDOW = 3      # 状态里保留"最近动作"的个数
NO_PROGRESS_LIMIT = 4  # 连续无新增证据多少步后值得停（供规则终止器参考）


def recent_actions(history: Sequence[HistoryItem], window: int = RECENT_WINDOW) -> tuple[str, ...]:
    """最近 window 个动作类型，按时间顺序，最新在后。"""
    return tuple(h.kind for h in list(history)[-window:])


def no_progress_streak(history: Sequence[HistoryItem]) -> int:
    """从历史尾部数连续"无新增证据"的步数。

    规则：从最后一步往前数，只要某步 yielded>0 就停；全 0 则一直数到头。
    这是"卡住"的最直接的可观测代理。
    """
    streak = 0
    for item in reversed(list(history)):
        if item.yielded > 0:
            break
        streak += 1
    return streak


def is_stalled(history: Sequence[HistoryItem], limit: int = NO_PROGRESS_LIMIT) -> bool:
    """连续无进展是否已经超过阈值。"""
    return no_progress_streak(history) >= limit
