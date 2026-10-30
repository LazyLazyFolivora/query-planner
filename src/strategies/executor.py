"""兼容门面：``src.strategies.executor`` -> :mod:`src.execution.executor`。"""

from __future__ import annotations

from ..execution.executor import *  # noqa: F401,F403
from ..execution.executor import EXECUTORS, PrimeKGExecutor, ToyKGExecutor  # noqa: F401