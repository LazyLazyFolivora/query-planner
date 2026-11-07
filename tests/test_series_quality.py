"""The radar's series quality state as a read-only decide input. It only steers which record is read
next (a value not verified now makes reading its record a verification read); it changes nothing for callers
that do not send it, and the baseline ignores it."""
import copy

import pytest
from fastapi.testclient import TestClient
from helpers import cand, check, request

from planner import baseline, policy
from planner.models import DecideRequest, DecideResponse
from planner.service import app

POINT = {"source": "nmpa", "record_id": "R4", "version_id": "v4", "object": "国家药监局",
         "metric": "批准注册医疗器械产品", "value": "318", "unit": "个", "period": "2026-04", "current": True,
         "extraction_check": {"status": "consistent", "gate": "series", "checker_version": "biocheck.extraction_rules.v7",
                              "check_id": 7, "reason": None}}
CANDIDATES = [cand("search", "nmpa", impact=0.9), cand("read", "nmpa", args={"record_id": "R1"}),
              cand("read", "nmpa", args={"record_id": "R4"}, impact=0.2)]


def point(**over):
    p = copy.deepcopy(POINT)
    p.update(over)
    return p


def test_a_request_without_series_decides_as_before():
    old = request(candidates=CANDIDATES)
    new = request(candidates=CANDIDATES, series=[])
    assert old.series == [] and policy.decide(old) == policy.decide(new)
    assert policy.decide(old).reason == "best_score"


def test_a_current_value_changes_nothing():
    plain = policy.decide(request(candidates=CANDIDATES))
    assert policy.decide(request(candidates=CANDIDATES, series=[point()])) == plain


@pytest.mark.parametrize("state", [
    {"current": False, "extraction_check": {"status": "unchecked", "gate": "candidate",
                                            "checker_version": "biocheck.extraction_rules.v7", "reason": "source_unreadable"}},
    {"current": False, "extraction_check": {"status": "inconsistent", "gate": "candidate",
                                            "checker_version": "biocheck.extraction_rules.v7", "reason": "not_consistent"}},
    {"current": False, "extraction_check": None},
])
def test_a_value_not_verified_now_makes_reading_its_record_a_verification_read(state):
    got = policy.decide(request(candidates=CANDIDATES, series=[point(**state)]))
    assert (got.decision, got.candidate, got.reason) == ("continue", 2, "verify_series_value")
    assert got.scores[2] >= policy.FLOOR / 2


def test_a_verification_read_does_not_hold_back_completion():
    body = dict(checks=[check(status="answered")], candidates=CANDIDATES,
                series=[point(current=False, extraction_check=None)])
    assert policy.decide(request(**body)).decision == "complete"


def test_the_baseline_ignores_the_series_state():
    for series in ([], [point()], [point(current=False)]):
        req = request(candidates=CANDIDATES, series=series)
        assert baseline.decide(req) == baseline.decide(request(candidates=CANDIDATES))


def test_the_answer_can_name_a_next_step_only():
    assert set(DecideResponse.model_fields) == {"decision", "candidate", "reason", "scores"}


@pytest.mark.parametrize("bad", [dict(point(), admit=True), point(current="maybe"),
                                 point(extraction_check={**POINT["extraction_check"], "gate": "promote"}),
                                 point(extraction_check={**POINT["extraction_check"], "extra": 1})])
def test_the_series_state_is_strict(bad):
    with pytest.raises(Exception):
        request(candidates=CANDIDATES, series=[bad])


def test_over_http_the_state_is_received_and_reaches_the_decision(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN", "t")
    client = TestClient(app)
    body = request(candidates=CANDIDATES, series=[point(current=False)]).model_dump()
    r = client.post("/v1/decide", json=body, headers={"Authorization": "Bearer t"})
    assert r.status_code == 200 and r.json()["reason"] == "verify_series_value" and r.json()["candidate"] == 2
    body["series"][0]["admit_to_series"] = True
    r = client.post("/v1/decide", json=body, headers={"Authorization": "Bearer t"})
    assert r.status_code == 422
    legacy = {k: v for k, v in body.items() if k != "series"}
    assert client.post("/v1/decide", json=legacy, headers={"Authorization": "Bearer t"}).status_code == 200
    assert DecideRequest.model_validate(legacy).series == []
