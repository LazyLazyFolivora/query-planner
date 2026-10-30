"""兼容门面：``src.strategies.ledger`` -> :mod:`src.accounting.ledger`。"""

from __future__ import annotations

from ..accounting.ledger import *  # noqa: F401,F403
from ..accounting.ledger import DIMENSIONS, LEDGERS, BudgetLedger  # noqa: F401