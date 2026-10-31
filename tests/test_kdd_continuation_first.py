"""Synthetic regressions for section 16; no provider/model or evaluator targets."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import radar_policy


def request():
    value = json.loads((ROOT / "contracts/radar-policy-v2-pagination.example.json").read_text())
    value["named_objects"] = []
    value["max_steps"] = 128
    return value


def new_query(aid="new-query"):
    return {"id": aid, "kind": "document_search", "arguments": {"provider": "fabric", "query": "public generic context " + aid}, "covers": ["source_scope"], "expected_information": [{"check_id": "source_scope", "probability": 1, "impact": 1}], "estimated_cost": {"tool_calls": 1}}


def receipt(action, more=True):
    page = action["arguments"]["page_number"]
    return {"action_id": action["id"], "kind": "document_search", "arguments": copy.deepcopy(action["arguments"]), "status": "completed", "actual_cost": {"tool_calls": 1}, "pagination": {"has_more": more, "next_cursor": "cursor-" + str(page + 1) if more else None, "max_seq": 42, "page_number": page}}


class ContinuationFirstTests(unittest.TestCase):
    def test_real_continuation_precedes_higher_scoring_new_query(self):
        req = request(); req["candidates"][0]["covers"] = []; req["candidates"].append(new_query())
        original = copy.deepcopy(req)
        result = radar_policy.decide(req)
        self.assertEqual(result["action"]["id"], "catalog-page-2")
        self.assertEqual(result["policy"]["method_version"], "kdd-information-gain-1.4")
        self.assertEqual(req, original)

    def test_continuation_does_not_spend_new_query_allowance_but_costs_tool_call(self):
        req = request(); req["action_limits"]["document_search"] = 16
        for i in range(15):
            action = new_query(str(i)); req["history"].append({"action_id": action["id"], "kind": action["kind"], "arguments": action["arguments"], "status": "completed", "actual_cost": {"tool_calls": 1}})
        result = radar_policy.decide(req)
        self.assertEqual(result["action"]["id"], "catalog-page-2")
        self.assertEqual(result["state_summary"]["new_query_attempt_count"], 16)
        self.assertEqual(result["state_summary"]["continuation_attempt_count"], 0)
        req["history"].append(receipt(req["candidates"][0], False)); req["candidates"] = [new_query("another")]
        result = radar_policy.decide(req)
        self.assertIsNone(result["action"])
        self.assertEqual(result["state_summary"]["action_attempt_counts"]["document_search"], 17)
        self.assertEqual(result["state_summary"]["new_query_attempt_count"], 16)
        self.assertEqual(result["state_summary"]["continuation_attempt_count"], 1)
        req = request(); req["budget"]["remaining"]["tool_calls"] = 0
        self.assertEqual(radar_policy.decide(req)["reason"], "budget_exhausted")

    def test_total_chain_pages_are_capped_at_five(self):
        req = request(); page = req["candidates"][0]
        for number in range(2, 6):
            page["arguments"]["page_number"] = number; page["id"] = "page-" + str(number)
            req["history"].append(receipt(page))
            page = copy.deepcopy(page); page["arguments"].update(previous_action_id=req["history"][-1]["action_id"], cursor="cursor-" + str(number + 1), page_number=number + 1)
        page["id"] = "page-6"
        req["candidates"] = []
        result = radar_policy.decide(req)
        self.assertIsNone(result["action"])
        self.assertEqual(result["state_summary"]["incomplete_search_chains"][0]["unavailable_reason"], "search_chain_page_limit")
        self.assertTrue(result["state_summary"]["terminal"])

    def test_missing_continuation_candidate_requests_host_rebuild_before_new_query(self):
        for candidates in [[], [new_query()]]:
            req = request(); req["candidates"] = candidates
            result = radar_policy.decide(req)
            self.assertEqual(result["reason"], "continuation_candidate_required")
            self.assertIsNone(result["action"])
            self.assertFalse(result["state_summary"]["terminal"])
            self.assertEqual(result["state_summary"]["execution_state"], "awaiting_host_candidates")
            self.assertEqual(result["state_summary"]["requested_continuations"][0]["previous_action_id"], "catalog-page-1")

    def test_unknown_receipt_still_precedes_any_action_or_host_rebuild(self):
        req = request(); entry = receipt(req["candidates"][0]); entry.pop("pagination"); entry["status"] = "unknown"; req["history"].append(entry)
        result = radar_policy.decide(req)
        self.assertEqual(result["reason"], "pending_execution_or_reconciliation")
        self.assertIsNone(result["action"])

    def test_named_object_unread_cannot_complete_and_gets_legal_zero_gain_lookup(self):
        req = request(); req["history"] = []; req["question"] = "Compare Orion and Lyra."; req["checks"][0].update(status="passed", evidence_ids=["seen-document-1"])
        req["named_objects"] = [{"id": "subject-1", "name": "Orion", "question_span": {"start": 8, "end": 13}, "status": "unread", "evidence_ids": [], "candidate_ids": ["lookup-subject"]}]
        query = new_query("lookup-subject"); query["covers"] = []; query["expected_information"] = []
        req["candidates"] = [query]
        result = radar_policy.decide(req)
        self.assertEqual(result["decision"], "continue")
        self.assertEqual(result["action"]["id"], "lookup-subject")
        self.assertEqual(result["state_summary"]["unmet_named_objects"][0]["id"], "subject-1")

    def test_all_unread_named_objects_and_reasons_survive_stop(self):
        req = request(); req["history"] = []; req["candidates"] = []; req["question"] = "Orion Lyra"
        req["named_objects"] = [{"id": "a", "name": "Orion", "question_span": {"start": 0, "end": 5}, "status": "unread", "candidate_ids": []}, {"id": "b", "name": "Lyra", "question_span": {"start": 6, "end": 10}, "status": "unavailable", "reason": "source_unavailable", "candidate_ids": []}]
        result = radar_policy.decide(req)
        self.assertEqual(result["reason"], "named_objects_unread")
        rows = result["state_summary"]["unmet_named_objects"]
        self.assertEqual({r["id"] for r in rows}, {"a", "b"})
        self.assertTrue(all(r["unavailable_reason"] for r in rows))

    def test_named_objects_must_come_from_question_and_current_read_support(self):
        req = request(); req["question"] = "Orion"; req["history"] = []; req["candidates"] = []
        req["checks"][0].update(status="passed", evidence_ids=["seen-document-1"])
        req["named_objects"] = [{"id": "a", "name": "Orion", "question_span": {"start": 0, "end": 5}, "status": "read", "evidence_ids": ["not-adopted"], "candidate_ids": []}]
        result = radar_policy.decide(req)
        self.assertNotEqual(result["decision"], "complete")
        req["named_objects"][0]["name"] = "Hidden target"
        with self.assertRaises(ValueError): radar_policy.decide(req)

    def test_formal_schema_example_and_terminal_named_reason(self):
        import jsonschema
        schema = json.loads((ROOT / "contracts/radar-policy-v2.schema.json").read_text())
        example = json.loads((ROOT / "contracts/radar-policy-v2-continuation-first.example.json").read_text())
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.validate(example, schema)
        result = radar_policy.decide(example)
        self.assertEqual(result["action"]["id"], "catalog-page-2")
        example["max_steps"] = 1
        result = radar_policy.decide(example)
        self.assertEqual(result["state_summary"]["unmet_named_objects"][0]["unavailable_reason"], "max_steps_reached")
        example["candidates"][0]["arguments"]["page_number"] = 6
        with self.assertRaises(jsonschema.ValidationError): jsonschema.validate(example, schema)
        with self.assertRaises(ValueError): radar_policy.decide(example)

    def test_named_inventory_and_pagination_keep_strict_input_boundaries(self):
        for mutate in [lambda r:r.update(named_objects=None), lambda r:r["scope"].pop("is_demo"), lambda r:r["candidates"][0]["arguments"].update(page_number=True)]:
            req = request(); mutate(req)
            with self.assertRaises(ValueError): radar_policy.decide(req)


if __name__ == "__main__": unittest.main()
