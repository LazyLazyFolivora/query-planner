from helpers import cand, check, hist, request

from planner.baseline import decide


def walk(candidates, checks=None, **budget):
    """Run the baseline as the radar would: execute the choice, record it in history."""
    history, order = [], []
    for _ in range(50):
        out = decide(request(checks, candidates, history, **budget))
        if out.decision != "continue":
            return order, out
        c = candidates[out.candidate]
        order.append(out.candidate)
        history.append(hist(c["kind"], c["source"], c["target_checks"], args=c.get("args", {})))
    raise AssertionError("baseline did not stop")


def test_searches_first_in_given_order_then_reads_in_given_order():
    candidates = [cand("read", args={"ids": ["E1"]}), cand("search", "pubmed", args={"query": "a"}),
                  cand("read", args={"ids": ["E2"]}), cand("search", "weknora", args={"query": "a"})]
    order, out = walk(candidates)
    assert order == [1, 3, 0, 2]
    assert (out.decision, out.reason) == ("insufficient", "nothing_left")


def test_identical_searches_are_run_once():
    candidates = [cand(args={"query": "BCMA CD3"}), cand(args={"query": "bcma  cd3"}), cand(source="b")]
    order, _ = walk(candidates)
    assert order == [0, 2]


def test_stops_when_reads_run_out_then_complete_only_if_required_answered():
    candidates = [cand(args={"query": "a"}), cand("read", args={"ids": ["E1"]}), cand("read", args={"ids": ["E2"]})]
    out = decide(request(None, candidates, [], reads=0))
    assert (out.decision, out.reason) == ("insufficient", "reads_exhausted")
    out = decide(request([check(status="answered")], candidates, [], reads=0))
    assert (out.decision, out.reason) == ("complete", "reads_exhausted")


def test_nothing_left_completes_when_answered():
    out = decide(request([check(status="answered")], []))
    assert (out.decision, out.reason) == ("complete", "nothing_left")


def test_answered_without_evidence_does_not_complete():
    assert decide(request([check(status="answered", evidence=[])], [])).decision == "insufficient"


def test_other_kinds_are_not_used():
    candidates = [cand("next_page", args={"cursor": "c"}), cand("analysis"), cand("fetch")]
    order, out = walk(candidates)
    assert order == [] and out.reason == "nothing_left"


def test_unaffordable_and_step_limit():
    assert decide(request(None, [cand(cost=9)], steps=3)).reason == "nothing_left"
    assert decide(request(None, [cand()], steps=0)).reason == "step_limit"
    assert decide(request(None, [cand()], [hist(status="pending")])).reason == "pending_execution"
