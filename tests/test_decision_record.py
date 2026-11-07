"""What every response records (reasons, channels, checks), sources kept apart from channels, upstream check
states, and that a request without any of the new fields decides as before."""
import pytest
from helpers import cand, check, hist, request
from pydantic import ValidationError

from planner import baseline
from planner.models import DecideRequest
from planner.policy import decide

README_REQUEST = {
    "question": "BCMA×CD3 双特异性抗体管线有哪些证据和风险？",
    "checks": [check("Q1"), check("Q2", "answered", importance=0.8)],
    "candidates": [cand(args={"query": "BCMA CD3 bispecific pipeline"}),
                   cand("read", targets=["Q1"], args={"ids": ["E3"]}),
                   cand(source="weknora", args={"query": "BCMA CD3 管线"})],
    "history": [hist(args={"query": "bcma cd3"})],
    "budget": {"steps_left": 8, "reads_left": 15},
}


def channels_of(out):
    return {c.channel: (c.status, c.reason) for c in out.channels}


def test_a_request_without_the_new_fields_decides_as_before():
    """The decision and scores are those of the planner before the record existed (README example)."""
    req = DecideRequest.model_validate(README_REQUEST)
    out = decide(req)
    assert (out.decision, out.candidate, out.reason, out.scores) == ("continue", 1, "best_score",
                                                                      [0.323232, 0.484848, 0.484848])
    assert out.reasons == ["scored", "open_required_read", "scored"]
    assert out.channels == []  # nothing said about channels, so nothing is reported
    assert [(c.id, c.state) for c in out.checks] == [("Q1", "open"), ("Q2", "answered")]
    old = baseline.decide(req)
    assert (old.decision, old.candidate, old.reason, old.scores) == ("continue", 0, "baseline_search", [])


def test_every_candidate_has_a_score_and_a_reason_and_a_stop_is_named():
    candidates = [cand(args={"query": "tried"}), cand(args={"query": "new"}), cand(args={"query": "NEW"}),
                  cand(cost=99, args={"query": "dear"}), cand("next_page", targets=[], args={"cursor": "c"}),
                  cand(targets=[], names_object=True, args={"query": "Orion"})]
    out = decide(request(candidates=candidates, history=[hist(args={"query": "tried"})], steps=5))
    assert out.reasons == ["already_done", "scored", "duplicate_in_list", "unaffordable", "next_page", "named_object"]
    assert (out.decision, out.candidate, out.reason) == ("continue", 4, "best_score")
    assert [s > 0 for s in out.scores] == [False, True, False, False, True, True]
    stopped = decide(request(candidates=[cand()], steps=0))
    assert (stopped.decision, stopped.candidate, stopped.reason, stopped.reasons) == (
        "insufficient", None, "step_limit", ["unaffordable"])


def test_a_channel_widens_no_source():
    sources = ["who_don", "gdelt_doc"]
    candidates = [cand(source="who_don", channel="fabric_literal", args={"terms": ["outbreak"]}),
                  cand(source="who_don", channel="weknora_semantic", args={"terms": ["outbreak"]}),
                  cand(source="nmpa", channel="weknora_semantic", args={"terms": ["outbreak"]}),  # a hit outside the set
                  cand(source="weknora", args={"terms": ["outbreak"]})]  # the channel posing as a source
    out = decide(request(candidates=candidates, sources=sources))
    assert out.reasons == ["scored", "scored", "source_out_of_scope", "source_out_of_scope"]
    assert out.scores[2] == out.scores[3] == 0.0
    assert channels_of(out) == {"fabric_literal": ("chosen", "scored"), "weknora_semantic": ("offered", "")}
    assert decide(request(candidates=candidates)).reasons == ["scored"] * 4  # no source set declared: no limit


def test_the_same_words_through_another_channel_are_another_action():
    tried = hist(source="who_don", channel="fabric_literal", args={"terms": ["outbreak"]})
    candidates = [cand(source="who_don", channel="fabric_literal", args={"terms": ["outbreak"]}),
                  cand(source="who_don", channel="weknora_semantic", args={"terms": ["Outbreak"]})]
    out = decide(request(candidates=candidates, history=[tried]))
    assert (out.reasons, out.candidate) == (["already_done", "scored"], 1)
    assert channels_of(out) == {"fabric_literal": ("not_needed", "already_done"),
                                "weknora_semantic": ("chosen", "scored")}
    failed = [hist(source="who_don", channel="fabric_literal", status="failed", args={"terms": ["a"]})]
    new = {"terms": ["b"]}
    literal, semantic = decide(request(candidates=[cand(source="who_don", channel="fabric_literal", args=new),
                                                   cand(source="who_don", channel="weknora_semantic", args=new)],
                                       history=failed)).scores
    assert literal < semantic  # what a channel did before is its own record


def test_a_channel_the_radar_calls_unavailable_is_never_chosen():
    declared = [{"channel": "graph", "available": False, "reason": "03_not_run"}, {"channel": "gdelt_leads"}]
    candidates = [cand("read", source="gdelt_doc", channel="graph", args={"record_id": "g1"}), cand()]
    out = decide(request(candidates=candidates, channels=declared))
    assert (out.reasons, out.candidate) == (["channel_unavailable", "scored"], 1)
    assert channels_of(out) == {"gdelt_leads": ("not_needed", "no_candidate"), "graph": ("unavailable", "03_not_run")}
    dear = decide(request(candidates=[cand(cost=99, channel="gdelt_leads")], steps=3))
    assert channels_of(dear) == {"gdelt_leads": ("unavailable", "unaffordable")}


@pytest.mark.parametrize("bad", [
    {"channels": [{"channel": "graph", "available": False}]},  # unavailable without a reason
    {"channels": [{"channel": "graph"}, {"channel": "graph"}]},
    {"channels": [{"channel": "wekonra"}]},
    {"sources": [" "]},
    {"candidates": [cand(channel="weknora")]},
    {"history": [hist(channel="weknora")]},
])
def test_the_new_fields_are_strict(bad):
    with pytest.raises(ValidationError):
        request(**bad)


def test_upstream_states_reopen_a_check_and_only_a_grounded_answer_closes_it():
    plain = [cand()]
    closed = decide(request([check(status="answered")], plain))  # the read stage's candidate answer, with E#
    assert (closed.decision, closed.checks[0].state) == ("complete", "answered")
    for upstream in ("open", "conflict"):  # 05, 06 or 07 sent the check back although it had evidence
        out = decide(request([check(status=upstream, evidence=["E1"])], plain))
        assert (out.decision, out.candidate, out.checks[0].state) == ("continue", 0, upstream)
    bare = decide(request([check(status="answered", evidence=[])], plain))
    assert (bare.decision, bare.checks[0].state, bare.checks[0].reason) == ("continue", "open", "answered_without_evidence")
    spent = decide(request([check(status="conflict", evidence=["E1"]), check("Q2", "answered")], plain, steps=0))
    assert (spent.decision, spent.reason) == ("insufficient", "step_limit")  # never complete, and the open one is listed
    assert [(c.id, c.state, c.required) for c in spent.checks] == [("Q1", "conflict", True), ("Q2", "answered", True)]


def test_the_baseline_carries_the_same_record():
    candidates = [cand("read", args={"ids": ["E1"]}), cand(args={"query": "a"}),
                  cand("analysis", channel="gdelt_leads", args={"x": 1})]
    out = baseline.decide(request(candidates=candidates))
    assert (out.candidate, out.scores, out.reasons) == (1, [], ["baseline_read", "baseline_search", "kind_not_used"])
    assert channels_of(out) == {"gdelt_leads": ("not_needed", "kind_not_used")}
