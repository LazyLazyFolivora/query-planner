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

    def test_reference_metadata_cannot_bypass_completed_action_deduplication(self):
        req = request(); req["candidates"] = [req["candidates"][1]]
        original = copy.deepcopy(req["candidates"][0])
        req["history"] = [receipt(original)]
        req["candidates"][0]["id"] = "new-id-with-display-label"
        req["candidates"][0]["arguments"]["input_refs"][0]["label"] = "renamed display only"
        out = radar_policy.decide(req)
        self.assertEqual(out["decision"], "insufficient")
        self.assertIsNone(out["action"])
        self.assertEqual(out["state_summary"]["excluded_candidates"],
                         [{"id": "new-id-with-display-label", "reason": "already_attempted"}])

    def test_observed_support_required_and_complete_is_only_check_coverage(self):
        req = request()
        for c in req["checks"]: c["status"] = "passed"
        self.assertEqual(radar_policy.decide(req)["decision"], "continue")
        for c in req["checks"]: c["artifact_refs"] = [{"id": "bulletins", "version": "sha256:fixed"}]
        req["budget"]["remaining"]["tool_calls"] = 0
        out = radar_policy.decide(req)
        self.assertEqual(out["decision"], "complete")
        self.assertEqual(out["policy"]["stop"], "observed_checks_reading_pages_objects_v4")
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

    def catalog_request(self):
        return json.loads((ROOT / "contracts/radar-policy-v2-catalog.example.json").read_text())

    def test_catalog_discovery_is_an_explicit_bounded_fabric_action(self):
        req = self.catalog_request(); before = copy.deepcopy(req)
        out = radar_policy.decide(req)
        self.assertEqual(out["decision"], "continue")
        self.assertEqual(out["action"], req["candidates"][0])
        self.assertEqual(out["policy"]["method_version"], "kdd-information-gain-1.4")
        self.assertEqual(out["policy"]["training_status"], "heuristic_untrained")
        self.assertEqual(req, before)
        self.assertEqual(out["action"]["arguments"]["terms"], [])
        self.assertNotIn("query", out["action"]["arguments"])

    def test_catalog_discovery_does_not_relax_ordinary_empty_or_invalid_search(self):
        changes = [
            lambda a: a.update(provider="weknora"),
            lambda a: a.update(query=None), lambda a: a.update(query=""), lambda a: a.update(query="*"),
            lambda a: a.update(terms=["placeholder"]), lambda a: a.update(catalog_discovery=False),
            lambda a: a.update(catalog_discovery="true"), lambda a: a.update(catalog_discovery=1),
            lambda a: a.pop("catalog_discovery"), lambda a: a.pop("limit"),
            lambda a: a.update(limit=0), lambda a: a.update(limit=21),
            lambda a: a.update(limit=True), lambda a: a.update(limit=1.5),
            lambda a: a.update(cursor="next-page"), lambda a: a.update(phase="followup"),
            lambda a: a["plan"].update(source_id="another-source"),
            lambda a: a["plan"].update(terms=["placeholder"]),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                req = self.catalog_request(); change(req["candidates"][0]["arguments"])
                with self.assertRaises(ValueError): radar_policy.decide(req)
        req = self.catalog_request(); req["schema_version"] = "query-planner.external.v1"
        with self.assertRaises(ValueError): radar_policy.decide(req)

    def test_catalog_discovery_requires_explicit_matching_scope_and_one_call(self):
        for key in ("source_id", "mode", "as_of", "is_demo", "scope"):
            req = self.catalog_request(); req["candidates"][0]["arguments"].pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError): radar_policy.decide(req)
        for key, value in (("mode", "other-mode"), ("as_of", "2027-01-01"), ("is_demo", False)):
            req = self.catalog_request(); a = req["candidates"][0]["arguments"]
            a[key] = value; a["scope"][key] = value
            out = radar_policy.decide(req)
            self.assertIn("scope_mismatch", [x["reason"] for x in out["state_summary"]["excluded_candidates"]])
        for change in [lambda r: r["scope"].pop("is_demo"),
                       lambda r: r["candidates"][0]["arguments"]["scope"].update(is_demo=1),
                       lambda r: r["candidates"][0]["estimated_cost"].update(tool_calls=0),
                       lambda r: r["candidates"][0]["estimated_cost"].update(tool_calls=2)]:
            req = self.catalog_request(); change(req)
            with self.assertRaises(ValueError): radar_policy.decide(req)
        req = self.catalog_request(); req["evidence"] = [{"id": "already-read"}]
        out = radar_policy.decide(req)
        self.assertIn("catalog_discovery_requires_empty_evidence", [x["reason"] for x in out["state_summary"]["excluded_candidates"]])

    def test_catalog_first_page_cannot_repeat_by_id_metadata_or_limit(self):
        for status in ("completed", "failed", "not_sent"):
            req = self.catalog_request(); original = copy.deepcopy(req["candidates"][0])
            req["history"].append(receipt(original, status=status))
            candidate = req["candidates"][0]; candidate["id"] = "renamed"
            candidate["arguments"].update(index=99, limit=20)
            candidate["arguments"]["plan"]["reason"] = "Different display wording"
            out = radar_policy.decide(req)
            self.assertEqual(out["decision"], "insufficient")
            self.assertIsNone(out["action"])
            self.assertEqual(out["state_summary"]["excluded_candidates"], [{"id": "renamed", "reason": "already_attempted"}])
        for status in ("prepared", "sent", "unknown", "unconfirmed"):
            req = self.catalog_request(); req["history"].append(receipt(req["candidates"][0], status=status))
            req["budget"]["remaining"]["tool_calls"] = 0
            self.assertEqual(radar_policy.decide(req)["reason"], "pending_execution_or_reconciliation")

    def test_catalog_discovery_respects_normal_action_quota_and_budget(self):
        req = self.catalog_request(); req["action_limits"] = {"document_search": 2}
        out = radar_policy.decide(req)
        self.assertEqual(out["decision"], "insufficient")
        self.assertIn("action_limit_reached", [x["reason"] for x in out["state_summary"]["excluded_candidates"]])
        req = self.catalog_request(); req["budget"]["remaining"]["tool_calls"] = 0
        self.assertEqual(radar_policy.decide(req)["decision"], "insufficient")

    def test_catalog_example_and_invalid_shapes_match_additive_schema(self):
        import jsonschema
        schema = json.loads((ROOT / "contracts/radar-policy-v2.schema.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        req = self.catalog_request(); jsonschema.validate(req, schema)
        for change in [lambda r: r["candidates"][0]["arguments"].update(query=None),
                       lambda r: r["candidates"][0]["arguments"].update(provider="weknora"),
                       lambda r: r["candidates"][0]["arguments"].update(limit=21),
                       lambda r: r["candidates"][0]["arguments"].update(cursor="next"),
                       lambda r: r["candidates"][0]["arguments"].update(catalog_discovery=1),
                       lambda r: r["scope"].pop("is_demo")]:
            bad = copy.deepcopy(req); change(bad)
            with self.assertRaises(jsonschema.ValidationError): jsonschema.validate(bad, schema)

    def reading_request(self):
        return json.loads((ROOT / "contracts/radar-policy-v2-reading-fetch.example.json").read_text())

    def test_unknown_coverage_is_preserved_and_not_rewarded(self):
        req = request()
        a = req["candidates"][0]
        a["covers"].append("model:unknown")
        a["expected_information"].append({"check_id": "model:unknown", "probability": 1, "impact": 1})
        old = copy.deepcopy(a); old["id"] = "earlier"; old["arguments"]["query"] = "earlier query"
        req["history"] = [receipt(old, new_evidence_ids=["bulletin-fixed-1"],
            observed_information={"model:unknown": 1},
            check_updates=[{"id": "model:unknown", "status": "passed", "evidence_ids": ["bulletin-fixed-1"]}])]
        before = copy.deepcopy(req)
        out = radar_policy.decide(req)
        self.assertEqual(out["action"]["id"], "descriptive")
        self.assertEqual(req, before)
        self.assertEqual(len(out["state_summary"]["coverage_issues"]), 4)
        self.assertTrue(all(r["status"] == "unknown" for r in out["state_summary"]["coverage_issues"]))
        req["candidates"] = [a]
        out = radar_policy.decide(req)
        self.assertEqual(out["action"], a)
        self.assertNotIn("model:unknown", [r["check_id"] for r in out["state_summary"]["candidate_scores"][0]["checks"]])

    def test_coverage_tolerance_never_swallows_invalid_contracts(self):
        mutations = [lambda r: r["checks"].append(copy.deepcopy(r["checks"][0])),
                     lambda r: r["candidates"][0].update(kind="illegal_graph_tool"),
                     lambda r: r["candidates"][0]["covers"].append(r["candidates"][0]["covers"][0]),
                     lambda r: r["candidates"][0]["expected_information"].append(copy.deepcopy(r["candidates"][0]["expected_information"][0]))]
        for mutate in mutations:
            req = request(); mutate(req)
            with self.assertRaises(ValueError): radar_policy.decide(req)

    def test_visible_correction_read_precedes_prior_and_zero_estimate(self):
        req = self.reading_request(); before = copy.deepcopy(req)
        out = radar_policy.decide(req)
        self.assertEqual(out["action"]["id"], "read-visible-update")
        self.assertEqual(out["state_summary"]["candidate_scores"][0]["expected_contribution"], 0)
        self.assertEqual(out["state_summary"]["candidate_scores"][0]["reading_priority"], 1)
        self.assertEqual(req, before)
        req["reading_manifest"][0]["priority"] = "normal"
        req["candidates"] = req["candidates"][:1]
        self.assertEqual(radar_policy.decide(req)["decision"], "continue")
        req["reading_manifest"][0]["basis"]["reference_ids"] = ["hidden-answer"]
        with self.assertRaisesRegex(ValueError, "visible sources"): radar_policy.decide(req)

    def test_unread_is_not_complete_even_when_checks_passed(self):
        req = self.reading_request()
        req["checks"][0].update(status="passed", evidence_ids=["visible-source-link"])
        self.assertEqual(radar_policy.decide(req)["decision"], "continue")
        req["candidates"] = []
        out = radar_policy.decide(req)
        self.assertEqual(out["decision"], "insufficient")
        self.assertEqual(len(out["state_summary"]["unmet_reading"]), 1)
        req["reading_manifest"][0]["status"] = "unavailable"
        self.assertEqual(radar_policy.decide(req)["decision"], "insufficient")
        req["reading_manifest"][0]["status"] = "read"
        self.assertEqual(radar_policy.decide(req)["decision"], "complete")

    def test_exact_read_identity_scope_and_version_cannot_be_relaxed(self):
        req = self.reading_request(); req["candidates"] = req["candidates"][:1]
        original = copy.deepcopy(req["candidates"][0]); req["history"] = [receipt(original)]
        req["candidates"][0]["id"] = "renamed"
        self.assertEqual(radar_policy.decide(req)["state_summary"]["excluded_candidates"][0]["reason"], "already_attempted")
        req["history"] = []
        req["candidates"][0]["arguments"]["records"][0]["version_id"] = "future-version"
        self.assertEqual(radar_policy.decide(req)["state_summary"]["excluded_candidates"][0]["reason"], "record_version_not_in_visible_manifest")
        req["candidates"][0] = original
        req["candidates"][0]["arguments"]["scope"]["is_demo"] = False
        self.assertEqual(radar_policy.decide(req)["state_summary"]["excluded_candidates"][0]["reason"], "scope_mismatch")
        req["candidates"][0]["arguments"]["records"][0]["label"] = "display"
        with self.assertRaises(ValueError): radar_policy.decide(req)

    def test_action_quotas_leave_other_kinds_and_pending_first(self):
        req = self.reading_request()
        req["action_limits"]["document_read"] = 0
        out = radar_policy.decide(req)
        self.assertEqual(out["action"]["kind"], "source_fetch")
        self.assertIn({"id": "read-visible-update", "reason": "action_limit_reached"}, out["state_summary"]["excluded_candidates"])
        req["history"] = [receipt(out["action"], status="unknown")]
        req["action_limits"]["source_fetch"] = 1
        self.assertEqual(radar_policy.decide(req)["reason"], "pending_execution_or_reconciliation")
        req["history"][0]["status"] = "not_sent"
        self.assertEqual(radar_policy.decide(req)["state_summary"]["action_attempt_counts"], {"source_fetch": 1})
        req["max_steps"] = 1
        self.assertEqual(radar_policy.decide(req)["reason"], "max_steps_reached")

    def test_registered_fetch_does_not_bypass_input_or_permission_gates(self):
        req = self.reading_request(); req["candidates"] = req["candidates"][1:]
        req["capabilities"][0]["input_requirements"] = ["declared_identifiers"]
        self.assertEqual(radar_policy.decide(req)["state_summary"]["excluded_candidates"][0]["reason"], "missing_required_input_kind")
        req["capabilities"][0]["input_requirements"] = []
        req["candidates"][0]["applicability"]["status"] = "ineligible"
        self.assertEqual(radar_policy.decide(req)["decision"], "insufficient")

    def test_published_additive_example_matches_formal_schema(self):
        import jsonschema
        schema = json.loads((ROOT / "contracts/radar-policy-v2.schema.json").read_text())
        jsonschema.Draft202012Validator(schema).validate(self.reading_request())
        jsonschema.Draft202012Validator(schema).validate(request())

    def test_exact_read_batch_reordering_cannot_create_a_second_execution(self):
        req = self.reading_request(); req["candidates"] = req["candidates"][:1]
        another = copy.deepcopy(req["reading_manifest"][0]); another["source_record_id"] = "second-visible-update"
        req["reading_manifest"].append(another)
        req["candidates"][0]["arguments"]["records"].append({"source_record_id": another["source_record_id"], "version_id": another["version_id"]})
        req["history"] = [receipt(copy.deepcopy(req["candidates"][0]), status="failed")]
        req["candidates"][0]["id"] = "renamed-batch"
        req["candidates"][0]["arguments"]["records"].reverse()
        out = radar_policy.decide(req)
        self.assertIsNone(out["action"])
        self.assertEqual(out["state_summary"]["excluded_candidates"][0]["reason"], "already_attempted")

    def test_host_locator_window_can_continue_same_version_but_not_repeat_first(self):
        req = self.reading_request(); req["candidates"] = req["candidates"][:1]
        first = copy.deepcopy(req["candidates"][0]); req["history"] = [receipt(first)]
        req["candidates"][0]["id"] = "explicit-zero"
        req["candidates"][0]["arguments"]["records"][0]["window_start"] = 0
        self.assertEqual(radar_policy.decide(req)["decision"], "insufficient")
        req["reading_manifest"][0]["status"] = "read"
        next_window = copy.deepcopy(req["reading_manifest"][0]); next_window.update(status="unread", window_start=6000)
        req["reading_manifest"].append(next_window)
        req["candidates"][0]["id"] = "actual-host-locator"
        req["candidates"][0]["arguments"]["records"][0]["window_start"] = 6000
        out = radar_policy.decide(req)
        self.assertEqual(out["action"]["id"], "actual-host-locator")
        self.assertEqual(out["state_summary"]["unread_version_count"], 1)
        self.assertEqual(out["state_summary"]["unread_window_count"], 1)
        req["candidates"][0]["arguments"]["records"][0]["window_start"] = 1
        self.assertEqual(radar_policy.decide(req)["decision"], "insufficient")
        for invalid in [-1, 0.5, True]:
            req["candidates"][0]["arguments"]["records"][0]["window_start"] = invalid
            with self.assertRaises(ValueError): radar_policy.decide(req)


if __name__ == "__main__": unittest.main()
