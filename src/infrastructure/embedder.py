"""文本嵌入器：所有“模型策略”的共用底座。

  SentenceEmbedder  默认真实编码器：BGE-small-zh / PubMedBERT / BGE-M3
  HashEmbedder      消融占位：字符 n-gram 哈希，确定性、零权重、零下载

默认模型可以通过环境变量覆盖：

    QRAG_EMBED_MODEL=BAAI/bge-small-zh-v1.5
    QRAG_EMBED_MODEL=pritamdeka/S-PubMedBert-MS-MARCO

注意：真实编码器需要安装 sentence-transformers / torch。
"""
from __future__ import annotations

import os
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

DEFAULT_EMBED_MODEL = os.getenv("QRAG_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    """字符 n-gram 哈希嵌入（仅用于消融，不是语义模型）。"""

    name = "hash"

    def __init__(self, dim: int = 256, ngram: int = 3) -> None:
        self.dim = dim
        self.ngram = ngram

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        s = f"^{text.strip()}$"
        n = self.ngram
        for i in range(max(len(s) - n + 1, 1)):
            gram = s[i : i + n]
            h = hash(gram) % self.dim
            vec[h] += 1.0
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0 else vec


class SentenceEmbedder:
    """真实语义编码器：sentence-transformers。

    推荐模型：
      - 4GB 显存 / 先跑通：BAAI/bge-small-zh-v1.5
      - 中英混合（大显存）：BAAI/bge-m3
      - 英文生物医学：pritamdeka/S-PubMedBert-MS-MARCO
      - 医学实体/术语：cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    """

    name = "sentence"

    def __init__(
        self,
        model_name: str = DEFAULT_EMBED_MODEL,
        dim: int | None = None,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "接入真实编码器需要安装 torch / sentence-transformers："
                "pip install torch transformers sentence-transformers"
            ) from exc

        self.model_name = model_name
        kwargs = {}
        if device:
            kwargs["device"] = device
        self._model = SentenceTransformer(model_name, **kwargs)
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = getattr(self._model, "get_sentence_embedding_dimension", None)
        inferred_dim = get_dim() if get_dim is not None else None
        self.dim = int(dim or inferred_dim or 768)

    def encode(self, text: str) -> np.ndarray:
        vec = self._model.encode([text], normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vecs = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)