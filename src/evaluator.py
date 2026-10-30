"""评测：使用 gold 证据需求给轨迹打分（对应 D 技术路线 8.1 节）。

这里刻意把"策略判断"和"评测打分"分开：
  * 策略只能看 PolicyState；
  * 只有本文件能看 Question.requirement_groups。
"""
from __future__ import annotations

from .schema import Decision, Question, RunTrace

VERDICT_CORRECT_COMPLETE = "正确完成"
VERDICT_WRONG_COMPLETE = "错误完成"
VERDICT_FALSE_INSUFFICIENT = "错误不足(证据已足)"
VERDICT_CORRECT_INSUFFICIENT = "正确不足"


def evaluate(question: Question, trace: RunTrace) -> dict:
    keys = {e.triple.key() for e in trace.final_evidence if e.triple is not None}
    hit: list[str] = []
    missed: list[str] = []
    total_w = 0.0
    hit_w = 0.0
    for g in question.requirement_groups:
        total_w += g.weight
        if g.satisfied_by(keys):
            hit.append(g.gid)
            hit_w += g.weight
        else:
            missed.append(g.gid)
    gold = hit_w / total_w if total_w else 0.0
    thr = question.sufficiency_threshold

    if trace.final_decision is Decision.COMPLETE and gold >= thr - 1e-9:
        verdict = VERDICT_CORRECT_COMPLETE
        wrong_complete = False
        false_insufficient = False
    elif trace.final_decision is Decision.COMPLETE:
        verdict = VERDICT_WRONG_COMPLETE
        wrong_complete = True
        false_insufficient = False
    elif gold >= thr - 1e-9:
        verdict = VERDICT_FALSE_INSUFFICIENT
        wrong_complete = False
        false_insufficient = True
    else:
        verdict = VERDICT_CORRECT_INSUFFICIENT
        wrong_complete = False
        false_insufficient = False

    return {
        "gold_sufficiency": gold,
        "groups_hit": hit,
        "groups_missed": missed,
        "verdict": verdict,
        "wrong_complete": wrong_complete,
        "false_insufficient": false_insufficient,
    }