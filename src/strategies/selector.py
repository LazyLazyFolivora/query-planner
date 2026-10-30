"""兼容门面：`src.strategies.selector` -> :mod:src.action_selection.pipeline。"""

from __future__ import annotations

from ..action_selection.pipeline import *  # noqa: F401,F403
from ..action_selection.pipeline import (  # noqa: F401
    FEATURE_NAMES,
    KINDS,
    N_FEAT,
    SELECTORS,
    ActionSelector,
    HeuristicSelector,
    LinearModelSelector,
    ModelSelector,
    Selection,
    action_features,
    action_feature_vector,
)
