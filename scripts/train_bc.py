#!/usr/bin/env python3
"""第一阶段行为克隆训练：三态停止头 + 动作效用/成本头。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.state_construction.encoder import FeatureEncoder
from src.action_selection.selector import ModelSelector
from src.stopping.stop import HierarchicalModelStop
from src.training.labels import action_dataset, stop_dataset
from src.training.rewards import gold_hit


def load_trajectories(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traces",
        default=str(ROOT / "outputs" / "trajectories" / "biohopr_rule.jsonl"),
    )
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "checkpoints"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    trajectories = load_trajectories(Path(args.traces))
    X_stop, y_stop = stop_dataset(trajectories)
    X_action, y_utility, y_cost = action_dataset(trajectories)
    print(f"trajectories={len(trajectories)} stop_samples={len(X_stop)} action_samples={len(X_action)}")

    encoder = FeatureEncoder()
    device = "cuda"
    stop = HierarchicalModelStop(encoder=encoder, hidden_dim=args.hidden_dim, device=device)
    selector = ModelSelector(encoder=encoder, hidden_dim=args.hidden_dim, device=device)
    print(f"device={selector.device}")

    stop.fit(X_stop, y_stop, epochs=args.epochs, learning_rate=args.lr, batch_size=32)
    selector.fit(
        X_action,
        y_utility,
        y_cost,
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=32,
    )

    torch = stop._torch
    with torch.no_grad():
        X = torch.as_tensor(X_stop, dtype=torch.float32).to(stop.device)
        hidden = stop.trunk(X)
        labels_t = torch.as_tensor(y_stop, dtype=torch.long).to(stop.device)
        p_stop = torch.sigmoid(stop.stop_head(hidden).squeeze(-1))
        p_complete = torch.sigmoid(stop.complete_head(hidden).squeeze(-1))
        pred = torch.where(
            p_stop < 0.5,
            torch.zeros_like(labels_t, dtype=torch.long),
            torch.where(
                p_complete >= stop.complete_threshold,
                torch.ones_like(labels_t, dtype=torch.long),
                torch.full_like(labels_t, 2, dtype=torch.long),
            ),
        )
        acc = float((pred.cpu().numpy() == y_stop).mean())
    hit_rate = sum(1 for t in trajectories if gold_hit(t)) / max(len(trajectories), 1)
    print(f"stop_bc_accuracy={acc:.3f} terminal_gold_hit_rate={hit_rate:.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dim": encoder.dim,
            "hidden_dim": args.hidden_dim,
            "trunk": stop.trunk.state_dict(),
            "stop_head": stop.stop_head.state_dict(),
            "complete_head": stop.complete_head.state_dict(),
            "complete_threshold": stop.complete_threshold,
            "use_heuristic_base": stop.use_heuristic_base,
        },
        out_dir / "bc_stop.pt",
    )
    torch.save(
        {
            "state_dim": encoder.dim,
            "action_dim": X_action.shape[1] - encoder.dim,
            "hidden_dim": args.hidden_dim,
            "trunk": selector.trunk.state_dict(),
            "utility_head": selector.utility_head.state_dict(),
            "cost_head": selector.cost_head.state_dict(),
            "use_heuristic_base": selector.use_heuristic_base,
        },
        out_dir / "bc_selector.pt",
    )
    print(f"saved -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
