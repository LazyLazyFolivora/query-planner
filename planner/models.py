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


class HistoryItem(_Strict):
    kind: Kind
    source: str = ""
    target_checks: list[str] = Field(default_factory=list)
    gain: float = Field(0.0, ge=0, le=1)
    args: dict = Field(default_factory=dict)  # lets the planner recognise an action that was already tried
    status: Literal["completed", "failed", "pending"] = "completed"


class Budget(_Strict):
    steps_left: int = Field(ge=0)
    reads_left: int = Field(ge=0)


class DecideRequest(_Strict):
    question: str
    checks: list[Check] = Field(default_factory=list, max_length=64)
    candidates: list[Candidate] = Field(default_factory=list, max_length=256)
    history: list[HistoryItem] = Field(default_factory=list, max_length=512)
    budget: Budget
    policy: Literal["planner", "baseline"] = "planner"

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
