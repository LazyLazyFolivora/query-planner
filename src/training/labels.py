"""从轨迹生成行为克隆标签。"""

from __future__ import annotations

from typing import Any

import numpy as np

DECISION_TO_INDEX = {
    "continue": 0,
    "complete": 1,
    "insufficient": 2,
}


def stop_dataset(trajectories: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[list[float]] = []
    ys: list[int] = []
    for trajectory in trajectories:
        for transition in trajectory.get("transitions", []):
            state = transition.get("state_vector") or []
            if not state:
                continue
            decision = transition.get("decision", "continue")
            if decision not in DECISION_TO_INDEX:
                continue
            xs.append([float(x) for x in state])
            ys.append(DECISION_TO_INDEX[decision])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def action_dataset(trajectories: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[list[float]] = []
    utility: list[float] = []
    cost: list[float] = []
    for trajectory in trajectories:
        for transition in trajectory.get("transitions", []):
            if not transition.get("action"):
                continue
            state = transition.get("state_vector") or []
            action_features = transition.get("action_features") or []
            if not state or not action_features:
                continue
            xs.append([float(x) for x in state] + [float(x) for x in action_features])
            utility.append(float(len(transition.get("new_evidence_ids") or [])))
            actual = transition.get("actual_cost") or {}
            cost.append(
                float(actual.get("relation_access", 0))
                + 0.5 * float(actual.get("tool_calls", 0))
                + float(actual.get("tokens", 0)) / 1000.0
            )
    return (
        np.asarray(xs, dtype=np.float32),
        np.asarray(utility, dtype=np.float32),
        np.asarray(cost, dtype=np.float32),
    )
