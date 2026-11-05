"""The planner policy: pick the next candidate or stop. Pure functions, no I/O.

Ported from the old kdd.py (kdd-information-gain-1.4), reduced to the SPEC 3.3 input.
"""
import json

from .models import Candidate, DecideRequest, DecideResponse

PRIOR = 0.5        # untrained prior chance that an action of this family helps a check
PRIOR_WEIGHT = 2   # the prior counts as this many past observations (Beta-style)
COST_WEIGHT = 0.25
EVIDENCE_WEIGHT = 0.5  # a family that keeps finding evidence without finishing a check is still worth trying
FLOOR = 0.25       # minimum score of an unread obligation (next page, named object, critical read)
PARTIAL = 0.25     # weight of an answered check while a required one is open: one record often answers
                   # only part of it (one of three companies, one period)
_SORTED_SETS = {"terms", "source_ids", "ids"}


# ---- checks --------------------------------------------------------------

def satisfied_ids(req: DecideRequest) -> set[str]:
    """'answered' only counts when it carries evidence; otherwise the check is still open."""
    return {c.id for c in req.checks if c.status == "answered" and c.evidence}


def required_met(req: DecideRequest) -> bool:
    required = {c.id for c in req.checks if c.required}
    return bool(required) and required <= satisfied_ids(req)


# ---- semantic dedup ------------------------------------------------------

def _norm(value, key=""):
    if isinstance(value, dict):
        return {k: _norm(v, k) for k, v in sorted(value.items())}
    if isinstance(value, list):
        items = [_norm(v, key) for v in value]
        if key in _SORTED_SETS and all(isinstance(v, str) for v in items):
            return sorted(set(items))
        return items
    if isinstance(value, str) and key in {"query", "terms"}:
        return " ".join(value.casefold().split())
    return value


def semantic_key(kind: str, source: str, args: dict) -> str:
    """Same kind + source + normalized args = same action (case, spacing and term order do not matter)."""
    return json.dumps([kind, source.strip().casefold(), _norm(args)], sort_keys=True,
                      ensure_ascii=False, default=str)


def live_candidates(req: DecideRequest) -> list[int]:
    """Indexes of candidates not tried before and not repeated earlier in the same list."""
    seen = {semantic_key(h.kind, h.source, h.args) for h in req.history}
    live = []
    for i, c in enumerate(req.candidates):
        key = semantic_key(c.kind, c.source, c.args)
        if key not in seen:
            live.append(i)
        seen.add(key)
    return live


def affordable(c: Candidate, req: DecideRequest) -> bool:
    return c.cost <= req.budget.steps_left and (c.kind != "read" or req.budget.reads_left >= 1)


# ---- scoring -------------------------------------------------------------

def _posterior(c: Candidate, check_id: str, req: DecideRequest) -> float:
    """Beta-style mean of the prior and past gains of the same action family (kind + source)."""
    family = (c.kind, c.source.strip().casefold())
    seen = [0.0 if h.status == "failed" else max(h.gain, EVIDENCE_WEIGHT * (h.evidence_gain or 0.0))
            for h in req.history
            if h.status != "pending" and (h.kind, h.source.strip().casefold()) == family
            and (h.status == "failed" or check_id in h.target_checks)]
    return (PRIOR_WEIGHT * PRIOR + sum(seen)) / (PRIOR_WEIGHT + len(seen))


def _pressure(c: Candidate, req: DecideRequest) -> float:
    pressure = c.cost / max(req.budget.steps_left, 1)
    if c.kind == "read":
        pressure = max(pressure, 1 / max(req.budget.reads_left, 1))
    return pressure


def _unverified_records(req: DecideRequest) -> set[tuple[str, str]]:
    """(source, record id) of series values the question uses that are not in the effective series now
    (unchecked, inconsistent, no longer readable) or that the radar may not compare now (withdrawn by a
    correction, no known domain): reading them again is how they could be verified."""
    return {
        (p.source.strip().casefold(), p.record_id)
        for p in req.series
        if not p.current or (p.comparability is not None and p.comparability.state == "not_comparable")
    }


def verifies(c: Candidate, req: DecideRequest) -> bool:
    """A read of the record behind a series value that is not verified now."""
    return c.kind == "read" and (c.source.strip().casefold(), str(c.args.get("record_id", ""))) in \
        _unverified_records(req)


def _critical_read(c: Candidate, open_required: set[str]) -> bool:
    """An unread record that bears on an open required check (or is marked critical)."""
    return c.kind == "read" and (c.critical or bool(open_required & set(c.target_checks)))


def score(c: Candidate, req: DecideRequest, done: set[str], open_required: set[str]) -> float:
    """importance x posterior gain x impact x scope_match over target checks (answered ones partly),
    divided by (1 + 0.25 x cost pressure); raised to a floor for unread obligations."""
    importance = {k.id: k.importance for k in req.checks}
    weight = {cid: PARTIAL if open_required else 0.0 for cid in done}
    gain = sum(importance[cid] * weight.get(cid, 1.0) * _posterior(c, cid, req) * c.impact * c.scope_match
               for cid in c.target_checks if cid in importance)
    pressure = _pressure(c, req)
    value = gain / (1 + COST_WEIGHT * pressure)
    if c.kind == "next_page" or c.names_object or _critical_read(c, open_required) or verifies(c, req):
        # A concrete unread thing must not be skipped just because an untrained
        # estimate says zero gain. This is an obligation, not measured gain.
        value = max(value, FLOOR * c.scope_match / (1 + pressure))
    return value


def priority(c: Candidate, open_required: set[str], req: DecideRequest | None = None) -> tuple[int, int, int]:
    """Ranked before the score: continuation page, then a named or newly found object, then a read the
    radar marked critical (correction, counter-evidence) or a read that would verify a series value the
    question uses but that is not verified now. An ordinary read for an open required check is only floored
    in score, not ranked first, so a run is not held to reading background record after background record
    while a search that follows up what was found waits."""
    verify = req is not None and verifies(c, req)
    return (int(c.kind == "next_page"), int(c.names_object or c.follows_up),
            int(c.kind == "read" and (c.critical or verify)))


def obligation(c: Candidate, open_required: set[str]) -> bool:
    """Work that must not be skipped while it can be afforded: it blocks stopping as complete."""
    return c.kind == "next_page" or c.names_object or _critical_read(c, open_required)


# ---- decision ------------------------------------------------------------

def decide(req: DecideRequest) -> DecideResponse:
    done = satisfied_ids(req)
    open_required = {c.id for c in req.checks if c.required} - done
    live = live_candidates(req)
    can_do = [i for i in live if affordable(req.candidates[i], req)]
    scores = [0.0] * len(req.candidates)
    for i in can_do:
        scores[i] = round(score(req.candidates[i], req, done, open_required), 6)

    def out(decision, reason, index=None):
        return DecideResponse(decision=decision, candidate=index, reason=reason, scores=scores)

    if any(h.status == "pending" for h in req.history):
        return out("insufficient", "pending_execution")
    # Work the radar can still do that must not be skipped: next page, named object, critical read.
    unfinished = [i for i in can_do if obligation(req.candidates[i], open_required)]
    if required_met(req) and not unfinished:
        return out("complete", "required_checks_answered")
    if req.budget.steps_left <= 0:
        return out("insufficient", "step_limit")
    if not live:
        return out("insufficient", "no_candidates")
    if not can_do:
        return out("insufficient", "nothing_affordable")
    best = min(can_do, key=lambda i: (tuple(-p for p in priority(req.candidates[i], open_required, req)),
                                      -scores[i], i))
    if scores[best] <= 0:
        return out("insufficient", "no_expected_gain")
    # the series state only steers which record is read next: a verification read is named as such
    return out("continue", "verify_series_value" if verifies(req.candidates[best], req) else "best_score", best)
