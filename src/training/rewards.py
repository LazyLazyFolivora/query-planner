"""轨迹奖励函数。"""

from __future__ import annotations


def gold_hit(trajectory: dict) -> bool:
    gold = [str(x).lower() for x in trajectory.get("metadata", {}).get("gold_answer", [])]
    evidence = " || ".join(trajectory.get("metadata", {}).get("final_evidence", [])).lower()
    return bool(gold) and any(item in evidence for item in gold)


def terminal_reward(trajectory: dict) -> float:
    decision = trajectory.get("final_decision", "")
    hit = gold_hit(trajectory)
    if decision == "complete":
        return 1.0 if hit else -1.0
    if decision == "insufficient":
        return -0.5 if hit else 0.0
    return 0.0


def backfill_terminal_reward(trajectory: dict) -> None:
    transitions = trajectory.get("transitions", [])
    if transitions:
        transitions[-1]["reward"] = terminal_reward(trajectory)
