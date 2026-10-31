"""Synthetic controls for the explicitly enabled partial-evidence catalog."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import radar_policy
VERSION = "bounded-required-gap-discovery-v1"

def request():
    req = json.loads((ROOT / "contracts/radar-policy-v2-catalog.example.json").read_text())
    req["discovery_version"] = VERSION
    req["evidence"] = [{"id": "SYNTHETIC-known"}]
    req["checks"] = [{"id": "gap:study_population", "status": "unknown", "required": True, "importance": 1}]
    action = req["candidates"][0]
    action["covers"] = ["gap:study_population"]
    action["discovery_basis"] = {"version": VERSION, "kind": "required_gap_repeat_search",
                                 "query_index": 0, "check_ids": ["gap:study_population"]}
    for row in req["history"]:
        row["arguments"]["scope"] = copy.deepcopy(req["scope"])
    return req

class RequiredGapCatalogTests(unittest.TestCase):
    def test_opted_in_completed_repeat_then_empty_is_selectable(self):
        req = request()
        original = copy.deepcopy(req)
        out = radar_policy.decide(req)
        self.assertEqual(out["decision"], "continue")
        self.assertEqual(out["action"], req["candidates"][0])
        self.assertEqual(req, original)
        self.assertEqual(out["policy"]["method_version"], "kdd-information-gain-1.4")

    def test_old_policy_still_rejects_nonempty_evidence(self):
        req = request(); req.pop("discovery_version")
        out = radar_policy.decide(req)
        self.assertIsNone(out["action"])
        self.assertIn("catalog_discovery_requires_empty_evidence",
                      [x["reason"] for x in out["state_summary"]["excluded_candidates"]])

    def test_bad_basis_or_changed_observations_never_unlock(self):
        changes = [
            lambda r: r["candidates"][0].pop("discovery_basis"),
            lambda r: r["candidates"][0]["discovery_basis"].update(query_index=True),
            lambda r: r["candidates"][0]["discovery_basis"].update(query_index=99),
            lambda r: r["candidates"][0]["discovery_basis"].update(check_ids=["missing"]),
            lambda r: r["checks"][0].update(required=False),
            lambda r: r["checks"][0].update(importance=0),
            lambda r: r["checks"][0].update(status="passed", evidence_ids=["SYNTHETIC-known"]),
            lambda r: r["history"][0].update(status="failed"),
            lambda r: r["history"][1].update(new_evidence_ids=["SYNTHETIC-known"]),
            lambda r: r["history"][1].update(actual_cost={"tool_calls": None}),
            lambda r: r["history"][1]["arguments"]["scope"].update(as_of="2099-01-01"),
            lambda r: r["history"][0]["arguments"].update(as_of="2099-01-01"),
            lambda r: r["history"][0]["arguments"].update(is_demo=False),
            lambda r: r["history"][0]["arguments"].update(mode="as_known"),
            lambda r: r["history"][0]["arguments"].update(source_id="other"),
            lambda r: r["history"][0].pop("new_evidence_ids"),
        ]
        for index, change in enumerate(changes):
            with self.subTest(index=index):
                req = request(); change(req)
                self.assertIsNone(radar_policy.decide(req)["action"])

    def test_pending_and_budget_still_stop_dispatch(self):
        for status in ("unknown", "sent"):
            req = request(); req["history"][1]["status"] = status
            self.assertEqual(radar_policy.decide(req)["reason"], "pending_execution_or_reconciliation")
        req = request(); req["budget"]["remaining"]["tool_calls"] = 0
        self.assertIsNone(radar_policy.decide(req)["action"])

    def test_new_marker_does_not_change_any_empty_evidence_catalog_result(self):
        req = request(); req["evidence"] = []
        req["candidates"][0].pop("discovery_basis")
        old = copy.deepcopy(req); old.pop("discovery_version")
        self.assertEqual(radar_policy.decide(req), radar_policy.decide(old))
