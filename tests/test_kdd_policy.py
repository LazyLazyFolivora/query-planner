"""KDD contract and routing regressions; fixtures are synthetic, not outcomes."""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import radar_policy


def request():
    return json.loads((ROOT / "contracts/radar-policy-v2.json").read_text())["example"]


def receipt(action, **values):
    return {"action_id": action["id"], "kind": action["kind"], "arguments": copy.deepcopy(action["arguments"]),
            "status": "completed", "new_evidence_ids": [], "new_artifact_refs": [], **values}


class KDDRoutingTests(unittest.TestCase):
    def test_information_contribution_and_real_receipt_change_route(self):
        req = request()
        first = radar_policy.decide(req)
        self.assertEqual(first["action"]["id"], "descriptive")
        req["artifacts"].append({"id": "counts", "version": "1", "kind": "analysis_result"})
        req["checks"][1].update(status="passed", artifact_refs=[{"id": "counts", "version": "1"}])
        req["history"] = [receipt(first["action"], new_artifact_refs=[{"id": "counts", "version": "1"}], observed_information={"pattern": 1})]
        req["budget"]["remaining"]["tool_calls"] -= 1
        second = radar_policy.decide(req)
        self.assertEqual(second["action"]["id"], "search-denominator")
        self.assertEqual(second["state_summary"]["unmet_gaps"][0]["missing_inputs"], ["population denominator"])

    def test_empirical_receipt_yield_changes_same_family_next_choice(self):
        req = request(); req["candidates"] = [req["candidates"][0]]
        a = req["candidates"][0]; a["expected_information"][0]["probability"] = 0.9
        b = copy.deepcopy(a); b["id"] = "semantic"; b["arguments"]["provider"] = "weknora"
        b["expected_information"][0]["probability"] = 0.65
        req["candidates"].append(b)
        self.assertEqual(radar_policy.decide(req)["action"]["id"], a["id"])
        old = copy.deepcopy(a); old["id"] = "old"; old["arguments"]["query"] = "previous query"
        req["history"] = [receipt(old, observed_information={"comparable_denominator": 0})]
        changed = radar_policy.decide(req)
        self.assertEqual(changed["action"]["id"], "semantic")
        self.assertEqual(changed["state_summary"]["candidate_scores"][1]["checks"][0]["receipt_count"], 1)
        # An unsupported positive gain cannot rescue the lower yielding family.
        req["history"][0]["observed_information"]["comparable_denominator"] = 1
        self.assertEqual(radar_policy.decide(req)["action"]["id"], "semantic")
        req["history"][0]["new_evidence_ids"] = ["bulletin-fixed-1"]
        self.assertEqual(radar_policy.decide(req)["action"]["id"], a["id"])

    def test_ineligible_analysis_does_not_block_supplement_or_qualified_alternative(self):
        req = request()
        req["capabilities"][0]["input_requirements"] = ["analysis_dataset"]
        out = radar_policy.decide(req)
        self.assertEqual(out["action"]["kind"], "document_search")
        self.assertIn({"id": "descriptive", "reason": "missing_required_input_kind"}, out["state_summary"]["excluded_candidates"])
        req["candidates"] = [req["candidates"][1]]
        out = radar_policy.decide(req)
        self.assertEqual(out["decision"], "insufficient")
        self.assertEqual(out["state_summary"]["unmet_gaps"][0]["acceptable_alternatives"], ["notification_descriptive"])

    def test_direction_scope_and_artifact_version_are_real_gates(self):
        for change, reason in [
            (lambda r: r["capabilities"][0].update(directions=["research"]), "direction_not_supported"),
            (lambda r: r["candidates"][1]["arguments"]["scope"].update(as_of="2026-10-02"), "scope_mismatch"),
            (lambda r: r["candidates"][1]["arguments"]["input_refs"][0].update(version="future"), "input_version_unavailable"),
            (lambda r: r["candidates"][1]["arguments"].update(capability_version="2"), "capability_version_or_kind_mismatch"),
            (lambda r: r.update(capabilities=[]), "unregistered_capability")]:
            req = request(); change(req); out = radar_policy.decide(req)
            self.assertEqual(out["action"]["id"], "search-denominator")
            self.assertIn(reason, [item["reason"] for item in out["state_summary"]["excluded_candidates"]])

    def test_all_four_directions_and_three_registered_action_kinds(self):
        for direction in ("pipeline", "research", "business", "public_health"):
            for kind in ("dataset_prepare", "analysis_run", "professional_assess"):
                req = request(); req["direction"] = direction; req["scope"]["direction"] = direction
                req["capabilities"][0].update(directions=[direction], kind=kind)
                action = req["candidates"][1]; action["kind"] = kind; action["arguments"]["scope"]["direction"] = direction
                out = radar_policy.decide(req)
                self.assertEqual(out["action"]["kind"], kind)
                self.assertEqual(out["policy"]["training_status"], "heuristic_untrained")

    def test_unknown_receipts_precede_completion_budget_and_method_change(self):
        for status in ("prepared", "sent", "unknown", "unconfirmed"):
            req = request(); action = req["candidates"][1]
            req["history"] = [receipt(action, status=status)]
            req["budget"]["remaining"]["tool_calls"] = 0
            for check in req["checks"]: check.update(status="passed", evidence_ids=["bulletin-fixed-1"])
            out = radar_policy.decide(req)
            self.assertEqual(out["reason"], "pending_execution_or_reconciliation")
            self.assertIsNone(out["action"])

    def test_no_retries_by_renaming_or_reordering_inputs(self):
        for status in ("completed", "failed", "not_sent"):
            req = request(); req["candidates"] = [req["candidates"][1]]
            original = copy.deepcopy(req["candidates"][0])
            req["history"] = [receipt(original, status=status)]
            req["candidates"][0]["id"] = "renamed"
            out = radar_policy.decide(req)
            self.assertEqual(out["decision"], "insufficient")
            self.assertEqual(out["state_summary"]["excluded_candidates"][0]["reason"], "already_attempted")

    def test_observed_support_required_and_complete_is_only_check_coverage(self):
        req = request()
        for c in req["checks"]: c["status"] = "passed"
        self.assertEqual(radar_policy.decide(req)["decision"], "continue")
        for c in req["checks"]: c["artifact_refs"] = [{"id": "bulletins", "version": "sha256:fixed"}]
        req["budget"]["remaining"]["tool_calls"] = 0
        out = radar_policy.decide(req)
        self.assertEqual(out["decision"], "complete")
        self.assertIn("not_scientific_or_business_approval", out["policy"]["scope"])

    def test_same_space_comparator_is_separately_named_and_deterministic(self):
        req = request(); before = copy.deepcopy(req)
        primary = radar_policy.decide_kdd(req)
        baseline = radar_policy.decide_kdd(req, selector="coverage_cost_baseline_v1")
        self.assertNotEqual(primary["policy"]["selector"], baseline["policy"]["selector"])
        self.assertEqual(req, before)
        req["candidates"].reverse()
        self.assertEqual(primary, radar_policy.decide_kdd(req))
        primary["action"]["arguments"]["parameters"]["mode"] = "mutation"
        self.assertEqual(req["candidates"][0]["arguments"]["parameters"]["mode"], "notification_counts_only")

    def test_bounded_numerical_contract(self):
        for change in [lambda r: r["checks"][0].update(importance=float("nan")),
                       lambda r: r["candidates"][0]["expected_information"][0].update(probability=1.1),
                       lambda r: r["candidates"][1]["arguments"].update(capability_id="different"),
                       lambda r: r.update(max_steps=0)]:
            req = request(); change(req)
            with self.assertRaises(ValueError): radar_policy.decide(req)


if __name__ == "__main__": unittest.main()
