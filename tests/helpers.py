"""Small builders so each test shows only what it is about."""
from planner.models import DecideRequest


def check(cid="Q1", status="open", evidence=None, required=True, importance=1.0):
    if status == "answered" and evidence is None:
        evidence = ["E1"]
    return {"id": cid, "text": cid, "required": required, "importance": importance,
            "status": status, "evidence": evidence or []}


def cand(kind="search", source="pubmed", targets=("Q1",), cost=1, **extra):
    return {"kind": kind, "source": source, "cost": cost, "target_checks": list(targets), **extra}


def hist(kind="search", source="pubmed", targets=("Q1",), gain=0.0, **extra):
    return {"kind": kind, "source": source, "target_checks": list(targets), "gain": gain, **extra}


def request(checks=None, candidates=(), history=(), steps=10, reads=20, **extra):
    body = {"question": "q", "checks": [check()] if checks is None else checks,
            "candidates": list(candidates), "history": list(history),
            "budget": {"steps_left": steps, "reads_left": reads}, **extra}
    return DecideRequest.model_validate(body)
