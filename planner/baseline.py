"""Fixed-order baseline: every distinct search once in the given order, then reads in the given order."""
from .models import DecideRequest, DecideResponse
from .policy import affordable, live_candidates, required_met


def decide(req: DecideRequest) -> DecideResponse:
    def stop(reason):
        decision = "complete" if required_met(req) else "insufficient"
        return DecideResponse(decision=decision, reason=reason)

    if any(h.status == "pending" for h in req.history):
        return DecideResponse(decision="insufficient", reason="pending_execution")
    if req.budget.steps_left <= 0:
        return stop("step_limit")
    if req.budget.reads_left == 0:
        return stop("reads_exhausted")
    live = [i for i in live_candidates(req) if affordable(req.candidates[i], req)]
    for kind, reason in (("search", "baseline_search"), ("read", "baseline_read")):
        for i in live:
            if req.candidates[i].kind == kind:
                return DecideResponse(decision="continue", candidate=i, reason=reason)
    return stop("nothing_left")
