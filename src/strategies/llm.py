"""兼容门面：``src.strategies.llm`` -> :mod:`src.infrastructure.llm`。"""

from __future__ import annotations

from ..infrastructure.llm import *  # noqa: F401,F403
from ..infrastructure.llm import LLMClient, NullLLM, OpenAICompatClient, build_resolve_prompt  # noqa: F401