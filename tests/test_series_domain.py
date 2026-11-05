"""T-24: a series value's comparability, apart from its extraction check. Not comparable now (withdrawn by a
correction, no known domain) makes reading its record a verification read; provisional changes nothing; the
planner never admits or compares a value."""
import copy

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from helpers import cand, request
from planner import baseline, policy
from planner.service import app
from test_series_quality import CANDIDATES, POINT

PROVISIONAL = {"version": "series_domain.v1", "state": "provisional", "domain_basis": "topic", "domain_id": "t-1",
               "reasons": ["needs_subject"]}


def point(**comparability):
    p = copy.deepcopy(POINT)
    p["comparability"] = dict(PROVISIONAL, **comparability)
    return p


def test_a_provisional_value_changes_nothing():
    plain = policy.decide(request(candidates=CANDIDATES))
    assert policy.decide(request(candidates=CANDIDATES, series=[point()])) == plain


@pytest.mark.parametrize("comparability", [
    {"state": "not_comparable", "reasons": ["withdrawn_by_correction", "cross_domain_correction", "needs_subject"]},
    {"state": "not_comparable", "domain_basis": "unknown", "domain_id": None, "reasons": ["domain_unknown"]},
])
def test_a_value_not_comparable_now_makes_reading_its_record_a_verification_read(comparability):
    got = policy.decide(request(candidates=CANDIDATES, series=[point(**comparability)]))
    assert (got.decision, got.candidate, got.reason) == ("continue", 2, "verify_series_value")


def test_an_unknown_reason_or_state_is_refused():
    with pytest.raises(ValidationError):
        request(candidates=CANDIDATES, series=[point(reasons=["looks_similar"])])
    with pytest.raises(ValidationError):
        request(candidates=CANDIDATES, series=[point(state="maybe")])  # comparable is a v2 state


def test_the_baseline_ignores_comparability():
    for p in (point(), point(state="not_comparable", reasons=["withdrawn_by_correction"])):
        assert baseline.decide(request(candidates=CANDIDATES, series=[p])) == baseline.decide(
            request(candidates=CANDIDATES))


def test_over_http_the_field_is_accepted_and_older_bodies_still_are(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN", "t")
    body = request(candidates=CANDIDATES, series=[point(state="not_comparable", reasons=["withdrawn_by_correction"])])
    old = request(candidates=CANDIDATES, series=[POINT])
    with TestClient(app) as c:
        h = {"Authorization": "Bearer t"}
        got = c.post("/v1/decide", json=body.model_dump(mode="json"), headers=h)
        assert got.status_code == 200 and got.json()["reason"] == "verify_series_value"
        assert c.post("/v1/decide", json=old.model_dump(mode="json", exclude_none=True), headers=h).status_code == 200



V2 = {"version": "series_domain.v2", "state": "comparable", "domain_basis": "topic", "domain_id": "t-1",
      "family": "case_count", "reasons": []}


def point_v2(**over):
    p = copy.deepcopy(POINT)
    p["comparability"] = dict(V2, **over)
    return p


def test_v2_a_comparable_value_changes_nothing():
    plain = policy.decide(request(candidates=CANDIDATES))
    assert policy.decide(request(candidates=CANDIDATES, series=[point_v2()])) == plain


@pytest.mark.parametrize("reasons", [["family_not_determined"], ["dim_unknown:population"],
                                     ["dim_unchecked:disease", "dim_unknown:date_basis"], ["efficacy_not_compared"]])
def test_v2_a_value_with_domain_gaps_is_a_verification_read(reasons):
    got = policy.decide(request(candidates=CANDIDATES, series=[point_v2(state="not_comparable", reasons=reasons)]))
    assert (got.decision, got.candidate, got.reason) == ("continue", 2, "verify_series_value")


@pytest.mark.parametrize("reasons", [["dim_unknown:mood"], ["dim_guessed:disease"], ["looks_similar"]])
def test_v2_an_unknown_reason_or_dimension_is_refused(reasons):
    with pytest.raises(ValidationError):
        request(candidates=CANDIDATES, series=[point_v2(state="not_comparable", reasons=reasons)])
