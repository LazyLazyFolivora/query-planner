"""第⑦阶段 证据与历史更新：把新证据并入可变状态，供下一轮 ② 重新整合。

    输入：EvidenceStore（可变证据容器）+ Action + 新证据 + 成本
    输出：无返回值，就地更新 store
          * evidence  证据列表（去重）
          * looked_up / expanded / tried_paths / relations_seen  掩码与覆盖记录
          * history   (动作类型, 新增证据数, 成本) 时间线，供 ② 状态编码的历史块使用

实现：
    standard   StandardUpdater   标准实现

EvidenceStore 定义在 common.stage_contracts（因为 ⑤/⑥/⑦ 和 runner 都要用），
这里 re-export 便于单阶段阅读。
"""
from __future__ import annotations

from ..common.stage_contracts import EvidenceStore, EvidenceUpdater
from .updater import UPDATERS, StandardUpdater

__all__ = [
    "EvidenceStore",
    "EvidenceUpdater",
    "UPDATERS",
    "StandardUpdater",
]
