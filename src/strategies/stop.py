"""兼容门面：`src.strategies.stop` -> :mod:src.stopping.pipeline。"""

from __future__ import annotations

from ..stopping.pipeline import *  # noqa: F401,F403
from ..stopping.pipeline import (  # noqa: F401
    DEFAULT_COMPLETE_THRESHOLD,
    NO_PROGRESS_LIMIT,
    STOP_RULES,
    FlatModelStop,
    HierarchicalModelStop,
    NaiveTargetStop,
    RuleStop,
    StopRule,
)
