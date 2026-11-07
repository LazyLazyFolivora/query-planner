"""The decision record every response carries, for the planner and the baseline alike. Pure functions.

The policies only choose; this says how each candidate, each retrieval channel and each check stood when they did,
so the radar can store the response and later show why an action was taken or the investigation stopped."""
from .models import CHANNELS, ChannelInput, ChannelState, CheckState, DecideRequest, DecideResponse, Reason

# reasons of a candidate that cannot be chosen now
DEAD = frozenset({"source_out_of_scope", "channel_unavailable", "already_done", "duplicate_in_list",
                  "unaffordable", "kind_not_used"})


def respond(req: DecideRequest, decision: str, reason: str, candidate: int | None,
            reasons: list[Reason], scores: list[float] | None = None) -> DecideResponse:
    return DecideResponse(decision=decision, candidate=candidate, reason=reason, scores=scores or [],
                          reasons=reasons, channels=channel_states(req, reasons, candidate),
                          checks=check_states(req))


def check_states(req: DecideRequest) -> list[CheckState]:
    """'answered' without evidence is read as open: the planner keeps asking until something grounds it."""
    states = []
    for c in req.checks:
        ungrounded = c.status == "answered" and not c.evidence
        states.append(CheckState(id=c.id, state="open" if ungrounded else c.status, required=c.required,
                                 reason="answered_without_evidence" if ungrounded else ""))
    return states


def channel_states(req: DecideRequest, reasons: list[Reason], chosen: int | None) -> list[ChannelState]:
    """The channels the request mentions (declared, offered through, or used before), in a fixed order."""
    declared = {c.channel: c for c in req.channels}
    present = set(declared) | {c.channel for c in req.candidates} | {h.channel for h in req.history}
    return [_channel_state(name, req, declared.get(name), reasons, chosen) for name in CHANNELS if name in present]


def _channel_state(name: str, req: DecideRequest, declared: ChannelInput | None, reasons: list[Reason],
                   chosen: int | None) -> ChannelState:
    def state(status, reason=""):
        return ChannelState(channel=name, status=status, reason=reason)

    if declared is not None and not declared.available:
        return state("unavailable", declared.reason)
    own = [i for i, c in enumerate(req.candidates) if c.channel == name]
    if chosen in own:
        return state("chosen", reasons[chosen])
    if any(reasons[i] not in DEAD for i in own):
        return state("offered")
    dead = {reasons[i] for i in own}
    if dead & {"already_done", "duplicate_in_list"}:
        return state("not_needed", "already_done")
    if "kind_not_used" in dead:
        return state("not_needed", "kind_not_used")
    for reason in ("source_out_of_scope", "unaffordable"):
        if reason in dead:
            return state("unavailable", reason)
    return state("not_needed", "already_done" if any(h.channel == name for h in req.history) else "no_candidate")
