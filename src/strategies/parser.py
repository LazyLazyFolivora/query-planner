"""兼容门面：``src.strategies.parser`` -> :mod:`src.parsing.legacy`。"""

from __future__ import annotations

from ..parsing.legacy import *  # noqa: F401,F403
from ..parsing.legacy import PARSERS, BioHopRParser, HybridParser, ModelParser, RuleParser  # noqa: F401