"""基础设施：与"第几阶段"无关的共用地基。

    llm.py       LLM 客户端协议 + NullLLM + OpenAI 兼容实现（默认关闭）
    embedder.py  文本嵌入器：BGE / PubMedBERT / 确定性 HashEmbedder
    nets.py      PyTorch MLP 组件（trunk / head / 张量工具），供③④的模型策略共用

依赖方向：阶段模块 -> infrastructure，反向不允许。
"""
from __future__ import annotations

from .embedder import DEFAULT_EMBED_MODEL, Embedder, HashEmbedder, SentenceEmbedder
from .llm import LLMClient, NullLLM, build_resolve_prompt
from .nets import as_tensor, make_trunk, require_torch, zero_init_head

__all__ = [
    "DEFAULT_EMBED_MODEL",
    "Embedder",
    "SentenceEmbedder",
    "HashEmbedder",
    "LLMClient",
    "NullLLM",
    "build_resolve_prompt",
    "require_torch",
    "make_trunk",
    "zero_init_head",
    "as_tensor",
]
