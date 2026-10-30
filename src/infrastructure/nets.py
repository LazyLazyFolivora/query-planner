"""PyTorch MLP components."""
from __future__ import annotations

from typing import Any

import numpy as np


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "深度学习模型需要安装 PyTorch：pip install torch transformers sentence-transformers"
        ) from exc
    return torch


def make_trunk(torch, in_dim: int, hidden_dim: int, dropout: float = 0.0):
    nn = torch.nn
    layers: list[Any] = [
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
    ]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def zero_init_head(torch, in_dim: int, out_dim: int = 1):
    nn = torch.nn
    head = nn.Linear(in_dim, out_dim)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    return head


def as_tensor(torch, x: np.ndarray):
    return torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32)