"""行为克隆 checkpoint 加载。"""

from __future__ import annotations

from pathlib import Path

from ..action_selection.selector import ModelSelector
from ..infrastructure.nets import make_trunk, zero_init_head
from ..stopping.stop import HierarchicalModelStop


def load_stop_checkpoint(model: HierarchicalModelStop, path: str | Path) -> None:
    ckpt = model._torch.load(Path(path), map_location=model.device)
    hidden_dim = int(ckpt.get("hidden_dim", model.hidden_dim))
    if hidden_dim != model.hidden_dim:
        model.hidden_dim = hidden_dim
        model.trunk = make_trunk(model._torch, model.encoder.dim, hidden_dim).to(model.device)
        model.stop_head = zero_init_head(model._torch, hidden_dim, 1).to(model.device)
        model.complete_head = zero_init_head(model._torch, hidden_dim, 1).to(model.device)
    model.trunk.load_state_dict(ckpt["trunk"])
    model.stop_head.load_state_dict(ckpt["stop_head"])
    model.complete_head.load_state_dict(ckpt["complete_head"])
    model.complete_threshold = float(ckpt.get("complete_threshold", model.complete_threshold))
    model.use_heuristic_base = bool(ckpt.get("use_heuristic_base", False))


def load_selector_checkpoint(model: ModelSelector, path: str | Path) -> None:
    ckpt = model._torch.load(Path(path), map_location=model.device)
    hidden_dim = int(ckpt.get("hidden_dim", model.hidden_dim))
    if hidden_dim != model.hidden_dim:
        model.hidden_dim = hidden_dim
        model.trunk = make_trunk(model._torch, model.in_dim, hidden_dim).to(model.device)
        model.utility_head = zero_init_head(model._torch, hidden_dim, 1).to(model.device)
        model.cost_head = zero_init_head(model._torch, hidden_dim, 1).to(model.device)
    model.trunk.load_state_dict(ckpt["trunk"])
    model.utility_head.load_state_dict(ckpt["utility_head"])
    model.cost_head.load_state_dict(ckpt["cost_head"])
    model.use_heuristic_base = bool(ckpt.get("use_heuristic_base", False))
