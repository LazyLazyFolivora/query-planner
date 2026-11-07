"""Request and response shapes of POST /v1/decide (BioRadar SPEC section 3.3)."""
import re
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from . import __version__

Kind = Literal["search", "read", "next_page", "fetch", "dataset", "analysis", "mechanism"]
KINDS = get_args(Kind)
# A source is material an investigation may use (who_don, gdelt_doc); a channel is only a way of finding
# candidates in it (Fabric literal search, WeKnora semantic search, the GDELT lead list, the graph).
# Allowing a channel never widens the sources.
Channel = Literal["fabric_literal", "weknora_semantic", "gdelt_leads", "graph"]
CHANNELS = get_args(Channel)
Source = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80)]


class _Strict(BaseModel):
    # A misspelled field is an error, not something silently ignored.
    model_config = ConfigDict(extra="forbid")


class Check(_Strict):
    id: str = Field(min_length=1)
    text: str = ""
    required: bool = True
    importance: float = Field(1.0, ge=0, le=1)
    status: Literal["open", "answered", "conflict"] = "open"
    evidence: list[str] = Field(default_factory=list)


class Candidate(_Strict):
    kind: Kind
    source: str = ""
    channel: Channel | None = None  # how the candidate was found; absent from older callers
    cost: float = Field(1.0, ge=0, allow_inf_nan=False)
    target_checks: list[str] = Field(default_factory=list)
    args: dict = Field(default_factory=dict)
    # Optional hints from the radar; the defaults give the plain untrained behaviour.
    impact: float = Field(1.0, ge=0, le=1)       # how much a hit would change the answer
    scope_match: float = Field(1.0, ge=0, le=1)  # how well it fits the question's scope
    critical: bool = False       # a read of a record the radar marked critical (correction, counter-evidence)
    names_object: bool = False   # looks up an object named in the question that is still unread
    follows_up: bool = False     # searches an object a record read so far named with a decisive fact


class HistoryItem(_Strict):
    kind: Kind
    source: str = ""
    channel: Channel | None = None  # send it with the candidates' channels, or an action is not recognised as done
    target_checks: list[str] = Field(default_factory=list)
    gain: float = Field(0.0, ge=0, le=1)  # how much the action moved checks towards answered
    # what it found apart from answering: new decisive evidence, a new object, a correction arriving
    # (None from callers that do not report it; then only gain counts, as before)
    evidence_gain: float | None = Field(None, ge=0, le=1)
    args: dict = Field(default_factory=dict)  # lets the planner recognise an action that was already tried
    status: Literal["completed", "failed", "pending"] = "completed"


class Budget(_Strict):
    steps_left: int = Field(ge=0)
    reads_left: int = Field(ge=0)


class ChannelInput(_Strict):
    """What the radar says about one retrieval channel this round."""
    channel: Channel
    available: bool = True
    reason: str = Field("", max_length=120)  # why not: not configured, not supported, quota used up, 03_not_run

    @model_validator(mode="after")
    def _unavailable_says_why(self):
        if not self.available and not self.reason:
            raise ValueError("an unavailable channel needs a reason")
        return self


class ExtractionCheck(_Strict):
    """The radar's persisted check of one measured value (BioCheck extraction consistency), as it is now."""
    status: Literal["consistent", "inconsistent", "unchecked"]
    gate: Literal["series", "candidate"]
    checker_version: str = Field(min_length=1, max_length=80)
    check_id: int | None = None
    reason: str | None = Field(None, max_length=120)


COMPARABILITY_REASONS = frozenset({
    "needs_subject", "domain_unknown", "withdrawn_by_correction", "cross_domain_correction",
    "family_not_determined", "efficacy_not_compared", "record_internal_conflict",
    "superseded_by_artifact",  # the value was read again from an upgraded artifact, or not given by it
})
DIM_KINDS = ("disease", "population", "date_basis", "case_definition", "coverage", "denominator_definition",
             "product_or_class", "milestone_definition", "indication", "entity", "stage")
DIM_REASON = re.compile(r"dim_(?:unknown|unchecked):(?:" + "|".join(DIM_KINDS) + r")")


class Comparability(_Strict):
    """Whether the radar may compare this value with the others of its series now, a layer apart from
    its extraction check: "provisional" - one known domain (the topic its record was placed in), the business
    subject not yet checked; "not_comparable" - withdrawn by a correction, or of no known domain."""
    version: str = Field(min_length=1, max_length=40)
    # v1: provisional / not_comparable; v2 (complete domain): comparable only when the metric family and every
    # required dimension are proven, not_comparable with each gap otherwise
    state: Literal["provisional", "comparable", "not_comparable"]
    domain_basis: Literal["topic", "unknown"]
    domain_id: str | None = Field(None, max_length=64)
    family: str = Field("", max_length=30)
    reasons: list[str] = Field(default_factory=list, max_length=16)

    @field_validator("reasons")
    @classmethod
    def _known_reasons(cls, reasons):
        for r in reasons:
            if r not in COMPARABILITY_REASONS and not DIM_REASON.fullmatch(r):
                raise ValueError(f"unknown comparability reason: {r}")
        return reasons


class SeriesPoint(_Strict):
    """A value the analysis is about, with its quality state. Read only: no decision changes it, and the
    planner has no way to admit a value to a series - it may only choose to re-read where it came from."""
    source: str = Field(min_length=1, max_length=80)
    record_id: str = Field(min_length=1, max_length=400)
    version_id: str = Field("", max_length=128)
    object: str = Field("", max_length=200)
    metric: str = Field("", max_length=200)
    value: str = Field("", max_length=40)
    unit: str = Field("", max_length=50)
    period: str = Field("", max_length=40)
    current: bool  # in the radar's effective series now (checked by the current checker, still readable)
    extraction_check: ExtractionCheck | None = None
    comparability: Comparability | None = None  # absent from older callers


class DecideRequest(_Strict):
    question: str
    checks: list[Check] = Field(default_factory=list, max_length=64)
    candidates: list[Candidate] = Field(default_factory=list, max_length=256)
    history: list[HistoryItem] = Field(default_factory=list, max_length=512)
    budget: Budget
    policy: Literal["planner", "baseline"] = "planner"
    # the series values the question rests on and their quality state (empty for older callers)
    series: list[SeriesPoint] = Field(default_factory=list, max_length=64)
    # the material sources this investigation may use; empty = not declared, nothing is held back by source
    sources: list[Source] = Field(default_factory=list, max_length=16)
    # the radar's word on each retrieval channel (empty for older callers)
    channels: list[ChannelInput] = Field(default_factory=list, max_length=len(CHANNELS))

    @model_validator(mode="after")
    def _unique_check_ids(self):
        ids = [c.id for c in self.checks]
        if len(set(ids)) != len(ids):
            raise ValueError("check ids must be unique")
        return self

    @model_validator(mode="after")
    def _unique_channels(self):
        names = [c.channel for c in self.channels]
        if len(set(names)) != len(names):
            raise ValueError("a channel may be declared once")
        return self


# Why a candidate scored what it did. Ranked in this order when it can be chosen: an obligation or a priority
# (continuation page, named object, follow-up, critical read, a read that verifies a series value, a read the
# open required checks need) or its score alone; the baseline names its fixed order. Otherwise why it cannot be
# chosen now.
Reason = Literal[
    "next_page", "named_object", "follows_up", "critical_read", "verify_series_value", "open_required_read", "scored",
    "baseline_search", "baseline_read",
    "source_out_of_scope", "channel_unavailable", "already_done", "duplicate_in_list", "unaffordable", "kind_not_used",
]


class ChannelState(_Strict):
    """One channel this round: offered (a candidate through it could be chosen), chosen, not_needed (what it
    offers was done already, or the policy has no use for it), unavailable (the radar says so, or nothing it
    offers can be done: out of the source set, over the budget)."""
    channel: Channel
    status: Literal["offered", "chosen", "not_needed", "unavailable"]
    reason: str = ""


class CheckState(_Strict):
    """A check as the planner read it. "answered" only with evidence: a candidate answer the read stage
    grounded, not a verified one."""
    id: str
    state: Literal["open", "answered", "conflict"]
    required: bool
    reason: str = ""  # answered_without_evidence: sent as answered, read as open


class DecideResponse(_Strict):
    decision: Literal["continue", "complete", "insufficient"]
    candidate: int | None = None  # index into the request's candidates; null unless decision == continue
    reason: str  # why it continues; for complete / insufficient, why it stops
    scores: list[float] = Field(default_factory=list)  # one per candidate, same order; 0.0 if not eligible
    # The rest is the decision record: what the radar stores to rebuild why this was decided.
    reasons: list[Reason] = Field(default_factory=list)  # one per candidate, same order
    channels: list[ChannelState] = Field(default_factory=list)
    checks: list[CheckState] = Field(default_factory=list)
    planner_version: str = __version__
