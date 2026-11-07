"""Fixed-order baseline: every distinct search once in the given order, then reads in the given order."""
from .models import Candidate, DecideRequest, DecideResponse
from .policy import eligibility, required_met
from .record import respond

ORDER = (("search", "baseline_search"), ("read", "baseline_read"))


def _reason(c: Candidate, state: str) -> str:
    """The baseline has no scores: a candidate it can take is named by its place in the fixed order."""
    return dict(ORDER).get(c.kind, "kind_not_used") if state == "live" else state


def decide(req: DecideRequest) -> DecideResponse:
    states = eligibility(req)
    reasons = [_reason(c, s) for c, s in zip(req.candidates, states)]

    def out(decision, reason, index=None):
        return respond(req, decision, reason, index, reasons)

    def stop(reason):
        return out("complete" if required_met(req) else "insufficient", reason)

    if any(h.status == "pending" for h in req.history):
        return out("insufficient", "pending_execution")
    if req.budget.steps_left <= 0:
        return stop("step_limit")
    if req.budget.reads_left == 0:
        return stop("reads_exhausted")
    for kind, reason in ORDER:
        for i, state in enumerate(states):
            if state == "live" and req.candidates[i].kind == kind:
                return out("continue", reason, i)
    return stop("nothing_left")
