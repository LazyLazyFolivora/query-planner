import pytest
from fastapi.testclient import TestClient

from planner.service import app

from helpers import cand, check

client = TestClient(app)
AUTH = {"Authorization": "Bearer secret"}


@pytest.fixture(autouse=True)
def token(monkeypatch):
    monkeypatch.setenv("SERVICE_TOKEN", "secret")


def body(**extra):
    return {"question": "q", "checks": [check()], "candidates": [cand(), cand("read", args={"ids": ["E1"]})],
            "history": [], "budget": {"steps_left": 5, "reads_left": 5}, **extra}


def test_health_needs_no_token():
    assert client.get("/health").json() == {"status": "ok"}


def test_decide_planner_and_baseline():
    r = client.post("/v1/decide", json=body(), headers=AUTH)
    assert r.status_code == 200
    out = r.json()
    assert out["decision"] == "continue" and len(out["scores"]) == 2
    best = max(range(2), key=lambda i: (out["scores"][i], -i))
    assert out["candidate"] == best  # an ordinary read is ranked by its score like the search
    r = client.post("/v1/decide", json=body(policy="baseline"), headers=AUTH)
    assert r.json()["candidate"] == 0 and r.json()["reason"] == "baseline_search"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer nope"}, {"Authorization": "secret"}])
def test_missing_or_wrong_token_is_401(headers):
    r = client.post("/v1/decide", json=body(), headers=headers)
    assert r.status_code == 401
    assert set(r.json()) == {"error_code", "message"} and r.json()["error_code"] == "unauthorized"


def test_unconfigured_token_is_refused(monkeypatch):
    monkeypatch.delenv("SERVICE_TOKEN")
    r = client.post("/v1/decide", json=body(), headers=AUTH)
    assert r.status_code == 503 and r.json()["error_code"] == "token_not_configured"


def test_bad_request_uses_the_uniform_error_shape():
    for bad in ({"question": "q"}, body(policy="other"), body(candidates=[{"kind": "search", "cost": -1}]),
                body(checks=[check("Q1"), check("Q1")]), body(budget={"steps_left": 1, "reads_left": 1, "x": 1})):
        r = client.post("/v1/decide", json=bad, headers=AUTH)
        assert r.status_code == 422
        assert set(r.json()) == {"error_code", "message"} and r.json()["error_code"] == "invalid_request"
    r = client.post("/v1/decide", content="{not json", headers={**AUTH, "Content-Type": "application/json"})
    assert r.status_code == 422 and r.json()["error_code"] == "invalid_request"


def test_unknown_route_and_method_use_the_uniform_error_shape():
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error_code"] == "not_found"
    r = client.get("/health/x")
    assert r.status_code == 404
