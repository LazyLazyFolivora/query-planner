import pytest

from planner.policy import decide

from helpers import cand, check, hist, request


# ---- stop rules, in order ------------------------------------------------

def test_pending_comes_before_everything_even_complete():
    req = request([check(status="answered")], [cand()], [hist(status="pending")])
    out = decide(req)
    assert (out.decision, out.reason, out.candidate) == ("insufficient", "pending_execution", None)


def test_complete_when_all_required_answered_with_evidence():
    out = decide(request([check("Q1", "answered"), check("Q2", "answered"),
                          check("Q3", required=False)], [cand(targets=["Q3"])]))
    assert (out.decision, out.candidate) == ("complete", None)


def test_answered_without_evidence_is_downgraded_to_open():
    req = request([check("Q1", "answered", evidence=[])], [cand()])
    out = decide(req)
    assert out.decision == "continue" and out.candidate == 0  # not complete: Q1 still needs evidence


def test_conflict_is_not_answered():
    assert decide(request([check(status="conflict", evidence=["E1"])], [cand()])).decision == "continue"


def test_no_required_checks_never_completes():
    out = decide(request([check(required=False)], []))
    assert (out.decision, out.reason) == ("insufficient", "no_candidates")


def test_complete_beats_step_limit_but_step_limit_beats_candidates():
    assert decide(request([check(status="answered")], [], steps=0)).decision == "complete"
    out = decide(request([check()], [cand()], steps=0))
    assert (out.decision, out.reason) == ("insufficient", "step_limit")


def test_no_candidates():
    assert decide(request([check()], [])).reason == "no_candidates"


def test_all_candidates_already_tried_counts_as_no_candidates():
    req = request([check()], [cand(args={"query": "BCMA"})], [hist(args={"query": "BCMA"})])
    assert decide(req).reason == "no_candidates"


def test_nothing_affordable_by_cost_or_by_reads():
    assert decide(request([check()], [cand(cost=5)], steps=3)).reason == "nothing_affordable"
    out = decide(request([check()], [cand("read")], reads=0))
    assert (out.decision, out.reason) == ("insufficient", "nothing_affordable")


def test_zero_expected_gain_stops():
    # the only candidate targets a check that is already answered
    req = request([check("Q1"), check("Q2", "answered")], [cand(targets=["Q2"])])
    assert decide(req).reason == "no_expected_gain"
    assert decide(request([check(importance=0)], [cand()])).reason == "no_expected_gain"


# ---- scoring -------------------------------------------------------------

def test_candidate_is_an_index_into_the_request_order():
    req = request([check("Q1", importance=0.2), check("Q2", importance=0.9)],
                  [cand(targets=["Q1"]), cand("search", "other", targets=["Q2"])])
    out = decide(req)
    assert out.candidate == 1
    assert out.scores[1] > out.scores[0] > 0


def test_importance_impact_and_scope_match_multiply():
    base = decide(request(candidates=[cand()])).scores[0]
    assert decide(request(candidates=[cand(impact=0.5)])).scores[0] == pytest.approx(base / 2, abs=1e-6)
    assert decide(request(candidates=[cand(scope_match=0.5)])).scores[0] == pytest.approx(base / 2, abs=1e-6)
    assert decide(request([check(importance=0.5)], [cand()])).scores[0] == pytest.approx(base / 2, abs=1e-6)


def test_cost_pressure_lowers_score():
    cheap, dear = decide(request(candidates=[cand(cost=1), cand(source="b", cost=8)])).scores
    assert cheap > dear
    assert round(cheap, 6) == round(0.5 / (1 + 0.25 * 0.1), 6)


def test_gain_adds_up_over_target_checks_but_not_answered_ones():
    checks = [check("Q1"), check("Q2"), check("Q3", "answered")]
    one, two, three = decide(request(checks, [cand(targets=["Q1"]), cand(source="b", targets=["Q1", "Q2"]),
                                              cand(source="c", targets=["Q1", "Q2", "Q3"])])).scores
    assert two > one and two == three


def test_past_gain_of_same_family_raises_score_and_failures_lower_it():
    new = {"query": "new"}
    history = [hist("search", "pubmed", gain=1.0, args={"query": "a"}), hist("search", "pubmed", gain=1.0, args={"query": "b"})]
    good, other = decide(request(candidates=[cand(args=new), cand(source="weknora", args=new)], history=history)).scores
    assert good > other
    failed = [hist("search", "pubmed", status="failed", args={"query": "a"})]
    bad, other = decide(request(candidates=[cand(args=new), cand(source="weknora", args=new)], history=failed)).scores
    assert bad < other


def test_history_of_another_check_or_family_is_ignored():
    new = {"query": "new"}
    history = [hist("search", "pubmed", targets=["Q2"], gain=1.0, args={"query": "a"}),
               hist("read", "pubmed", gain=1.0, args={"query": "b"})]
    plain = decide(request(candidates=[cand(args=new)])).scores
    assert decide(request([check(), check("Q2")], [cand(args=new)], history)).scores == plain


# ---- semantic dedup ------------------------------------------------------

def test_dedup_ignores_case_spacing_and_term_order():
    tried = hist(args={"terms": ["BCMA", "CD3"], "query": "BCMA  CD3"})
    same = cand(source="PubMed ", args={"terms": ["cd3", "bcma"], "query": " bcma cd3"})
    other = cand(args={"query": "different"})
    out = decide(request(candidates=[same, other], history=[tried]))
    assert out.candidate == 1 and out.scores[0] == 0.0


def test_dedup_different_kind_or_source_or_args_is_new():
    tried = hist(args={"query": "a"})
    for new in (cand("read", args={"query": "a"}), cand(source="weknora", args={"query": "a"}),
                cand(args={"query": "b"})):
        assert decide(request(candidates=[new], history=[tried])).candidate == 0


def test_duplicate_inside_candidates_keeps_first_only():
    out = decide(request(candidates=[cand(args={"query": "a"}), cand(args={"query": "A"})]))
    assert out.candidate == 0 and out.scores[1] == 0.0


# ---- priority floors -----------------------------------------------------

def test_next_page_goes_before_a_higher_scoring_new_search():
    req = request([check(importance=1.0)],
                  [cand(args={"query": "new"}), cand("next_page", targets=[], args={"cursor": "c2"})])
    out = decide(req)
    assert out.candidate == 1
    assert out.scores[1] == round(0.25 / 1.1, 6)  # floor / (1 + pressure)


def test_next_page_blocks_complete_while_affordable():
    checks = [check(status="answered")]
    page = cand("next_page", targets=[], args={"cursor": "c2"})
    out = decide(request(checks, [page]))
    assert (out.decision, out.candidate) == ("continue", 0)
    assert decide(request(checks, [dict(page, cost=50)])).decision == "complete"  # cannot be afforded


def test_read_for_open_required_check_beats_higher_scoring_search():
    req = request([check()], [cand(), cand("read", "pubmed", impact=0, args={"ids": ["E9"]})])
    out = decide(req)
    assert out.candidate == 1 and out.scores[1] > 0  # zero estimated impact, still floored


def test_read_of_answered_or_optional_check_has_no_floor():
    req = request([check("Q1", "answered"), check("Q2", required=False), check("Q3")],
                  [cand("read", targets=["Q2"], impact=0), cand(source="b", targets=["Q3"])])
    assert decide(req).candidate == 1


def test_critical_unread_record_blocks_complete():
    checks = [check(status="answered")]
    correction = cand("read", targets=[], critical=True, args={"ids": ["E7"]})
    out = decide(request(checks, [correction]))
    assert (out.decision, out.candidate) == ("continue", 0)
    assert decide(request(checks, [correction], reads=0)).decision == "complete"


def test_named_object_lookup_is_floored_and_blocks_complete():
    lookup = cand(targets=[], names_object=True, args={"query": "Orion"})
    out = decide(request([check(status="answered")], [lookup]))
    assert (out.decision, out.candidate) == ("continue", 0)
    out = decide(request([check()], [cand(), lookup]))
    assert out.candidate == 1
