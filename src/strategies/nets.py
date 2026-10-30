"""兼容门面：``src.strategies.nets`` -> :mod:`src.infrastructure.nets`。"""

from __future__ import annotations

from ..infrastructure.nets import *  # noqa: F401,F403
from ..infrastructure.nets import as_tensor, make_trunk, require_torch, zero_init_head  # noqa: F401