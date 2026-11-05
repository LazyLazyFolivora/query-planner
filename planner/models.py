"""Request and response shapes of POST /v1/decide (BioRadar SPEC section 3.3)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Kind = Literal["search", "read", "next_page", "fetch", "dataset", "analysis", "mechanism"]


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


class ExtractionCheck(_Strict):
    """The radar's persisted check of one measured value (BioCheck extraction consistency), as it is now."""
    status: Literal["consistent", "inconsistent", "unchecked"]
    gate: Literal["series", "candidate"]
    checker_version: str = Field(min_length=1, max_length=80)
    check_id: int | None = None
    reason: str | None = Field(None, max_length=120)


class Comparability(_Strict):
    """Whether the radar may compare this value with the others of its series now (T-24), a layer apart from
    its extraction check: "provisional" - one known domain (the topic its record was placed in), the business
    subject not yet checked; "not_comparable" - withdrawn by a correction, or of no known domain."""
    version: str = Field(min_length=1, max_length=40)
    state: Literal["provisional", "not_comparable"]
    domain_basis: Literal["topic", "unknown"]
    domain_id: str | None = Field(None, max_length=64)
    reasons: list[Literal["needs_subject", "domain_unknown", "withdrawn_by_correction",
                          "cross_domain_correction"]] = Field(default_factory=list, max_length=8)


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
    comparability: Comparability | None = None  # T-24; absent from older callers


class DecideRequest(_Strict):
    question: str
    checks: list[Check] = Field(default_factory=list, max_length=64)
    candidates: list[Candidate] = Field(default_factory=list, max_length=256)
    history: list[HistoryItem] = Field(default_factory=list, max_length=512)
    budget: Budget
    policy: Literal["planner", "baseline"] = "planner"
    # T-29 P3: the series values the question rests on and their quality state (empty for older callers)
    series: list[SeriesPoint] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def _unique_check_ids(self):
        ids = [c.id for c in self.checks]
        if len(set(ids)) != len(ids):
            raise ValueError("check ids must be unique")
        return self


class DecideResponse(_Strict):
    decision: Literal["continue", "complete", "insufficient"]
    candidate: int | None = None  # index into the request's candidates; null unless decision == continue
    reason: str
    scores: list[float] = Field(default_factory=list)  # one per candidate, same order; 0.0 if not eligible
