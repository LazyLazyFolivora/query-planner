"""兼容门面：``src.strategies.encoder`` -> :mod:`src.state_construction.encoder`。"""

from __future__ import annotations

from ..state_construction.encoder import *  # noqa: F401,F403
from ..state_construction.encoder import ENCODERS, FeatureEncoder, ModelEncoder  # noqa: F401