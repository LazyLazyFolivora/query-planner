"""兼容门面：``src.strategies.updater`` -> :mod:`src.evidence.updater`。"""

from __future__ import annotations

from ..evidence.updater import *  # noqa: F401,F403
from ..evidence.updater import UPDATERS, StandardUpdater  # noqa: F401