"""Offline contract tests use the actual selectors and stop rules, no model calls."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import radar_policy


def candidate(aid="doc", kind="document_search", covers=None, **kwargs):
    args = {"provider": "fabric", "source_id": "europe_pmc", "query": "drug A trials"}
    if kind != "document_search":
        args = {"provider": "neo4j", "entity_id": "drug-a", "relation": "targets"}
    result = {"id": aid, "kind": kind, "arguments": args,
              "covers": covers if covers is not None else ["clinical"], "estimated_cost": {"tool_calls": 1}}
    result.update(kwargs)
    return result


def request():
    return {"schema_version": radar_policy.SCHEMA_VERSION, "question": "Which clinical and mechanism evidence supports drug A?",
        "checks": [{"id": "clinical", "status": "unknown"}, {"id": "mechanism", "status": "unknown"}],
        "evidence": [], "history": [], "candidates": [candidate()],
        "budget": {"limits": {"tool_calls": 8}, "remaining": {"tool_calls": 8}}, "max_steps": 12}


def receipt(action, status="completed", new=None):
    return {"action_id": action["id"], "kind": action["kind"], "arguments": action["arguments"],
            "status": status, "new_evidence_ids": new or []}


class PolicyContractTests(unittest.TestCase):
    def test_actual_feedback_changes_selection_without_fake_graph(self):
        req = request()
        req["candidates"] += [candidate("graph", "graph_neighbors", ["mechanism"])]
        policy = radar_policy.create_policy()
        with patch.object(policy.selector, "select", wraps=policy.selector.select) as choose, patch.object(policy.stop, "decide", wraps=policy.stop.decide) as stop:
            first = policy.decide(req)
            self.assertEqual(first["action"]["id"], "doc")
            state = choose.call_args.args[0]
            self.assertEqual(state.evidence, ())
            self.assertEqual(state.seeds, ())
            self.assertEqual(stop.call_count, 1)
        # Same candidates, different observed check/evidence feedback changes choice.
        req["evidence"] = [{"id": "paper-1", "source": "real host receipt"}]
        req["checks"][0].update(status="passed", evidence_ids=["paper-1"])
        second = policy.decide(req)
        self.assertEqual(second["action"]["id"], "graph")
        req["history"] = [receipt(first["action"], new=["paper-1"])]
        req["budget"]["remaining"]["tool_calls"] = 7
        self.assertEqual(policy.decide(req)["action"]["id"], "graph")

    def test_expected_coverage_is_not_observed_sufficiency(self):
        req = request()
        req["candidates"][0]["covers"] = ["clinical", "mechanism"]
        self.assertEqual(radar_policy.decide(req)["decision"], "continue")
        for c in req["checks"]:
            c["status"] = "passed"
        self.assertEqual(radar_policy.decide(req)["decision"], "continue")
        req["evidence"] = [{"id": "e"}]
        for c in req["checks"]:
            c["evidence_ids"] = ["e"]
        req["budget"]["remaining"]["tool_calls"] = 0
        out = radar_policy.decide(req)
        self.assertEqual(out["decision"], "complete")
        self.assertIsNone(out["action"])

    def test_no_required_checks_never_complete(self):
        req = request()
        req["checks"] = []
        req["candidates"] = []
        out = radar_policy.decide(req)
        self.assertEqual(out["decision"], "insufficient")
        self.assertIn("missing_required_checks", out["state_summary"]["warnings"])

    def test_uncertain_receipts_block_new_id_and_different_action(self):
        for status in ("prepared", "sent", "unknown", "unconfirmed"):
            req = request()
            req["history"] = [receipt(req["candidates"][0], status)]
            req["candidates"] = [candidate("new-id"), candidate("graph", "graph_path")]
            out = radar_policy.decide(req)
            self.assertEqual(out["decision"], "insufficient")
            self.assertEqual(out["reason"], "pending_execution_or_reconciliation")
            self.assertIsNone(out["action"])

    def test_terminal_receipts_semantic_dedup_and_no_automatic_retry(self):
        for status in ("completed", "failed", "not_sent"):
            req = request()
            req["history"] = [receipt(req["candidates"][0], status)]
            req["candidates"] = [candidate("new-id")]
            req["candidates"][0]["arguments"]["query"] = " DRUG   a TRIALS "
            out = radar_policy.decide(req)
            self.assertEqual(out["decision"], "insufficient")
            self.assertEqual(out["state_summary"]["excluded_candidates"][0]["reason"], "already_attempted")

    def test_terms_set_semantics_and_distinct_provider(self):
        req = request()
        req["candidates"] = [candidate("a", arguments={"provider": "fabric", "terms": ["Drug A", "Trials"]}),
                             candidate("b", arguments={"provider": "fabric", "terms": [" trials ", "DRUG A"]}),
                             candidate("c", arguments={"provider": "weknora", "query": "Drug A trials"})]
        out = radar_policy.decide(req)
        self.assertEqual(out["state_summary"]["legal_action_count"], 2)
        self.assertEqual(out["state_summary"]["excluded_candidates"], [{"id": "b", "reason": "duplicate_semantic_action"}])

    def test_budget_unknown_zero_and_omitted_dimension(self):
        req = request()
        req["budget"]["limits"]["tokens"] = req["budget"]["remaining"]["tokens"] = 10
        self.assertEqual(radar_policy.decide(req)["decision"], "insufficient")
        req["candidates"][0]["estimated_cost"]["tokens"] = None
        self.assertEqual(radar_policy.decide(req)["decision"], "insufficient")
        req["candidates"][0]["estimated_cost"]["tokens"] = 0
        self.assertEqual(radar_policy.decide(req)["decision"], "continue")
        del req["budget"]["limits"]["tokens"], req["budget"]["remaining"]["tokens"]
        del req["candidates"][0]["estimated_cost"]["tokens"]
        out = radar_policy.decide(req)
        self.assertNotIn("tokens", out["action"]["estimated_cost"])
        req["budget"]["remaining"]["tool_calls"] = 0
        self.assertEqual(radar_policy.decide(req)["reason"], "budget_exhausted")

    def test_restart_is_deterministic_and_input_not_mutated(self):
        req = request()
        req["candidates"].append(candidate("other", arguments={"provider": "weknora", "query": "drug A trials"}))
        saved = copy.deepcopy(req)
        one = radar_policy.decide(req)
        two = radar_policy.decide(json.loads(json.dumps(req)))
        self.assertEqual(one, two)
        req["candidates"].reverse()
        self.assertEqual(one, radar_policy.decide(req))
        req["candidates"].reverse()
        self.assertEqual(req, saved)
        one["action"]["arguments"]["query"] = "modified result"
        self.assertEqual(req, saved)

    def test_max_steps_and_no_progress(self):
        req = request()
        req["max_steps"] = 1
        req["history"] = [receipt(candidate("previous", arguments={"provider": "fabric", "query": "different"}))]
        self.assertEqual(radar_policy.decide(req)["reason"], "max_steps_reached")
        req["max_steps"] = 12
        req["history"] = [receipt(candidate(str(i), arguments={"provider": "fabric", "query": str(i)})) for i in range(4)]
        self.assertEqual(radar_policy.decide(req)["reason"], "no_progress")

    def test_all_graph_kinds_reuse_selector_and_untrained_metadata(self):
        for kind in ("entity_relations", "graph_neighbors", "graph_path"):
            req = request()
            req["candidates"] = [candidate(kind, kind)]
            for selector in ("heuristic", "linear"):
                result = radar_policy.create_policy(selector=selector).decide(req)
                self.assertEqual(result["action"]["kind"], kind)
                self.assertEqual(result["policy"]["training_status"], "heuristic_untrained")

    def test_model_stop_cannot_certify_unmet_checks(self):
        policy = radar_policy.create_policy()
        with patch.object(policy.stop, "decide", return_value=(radar_policy._api.Decision.COMPLETE, "learned")):
            out = policy.decide(request())
        self.assertEqual(out["decision"], "insufficient")
        self.assertEqual(out["reason"], "retrieval_checks_not_satisfied")

    def test_bounded_input_and_one_provider_contract(self):
        for mutate in (lambda r: r.update(max_steps=0), lambda r: r.update(max_steps=257),
                       lambda r: r.update(candidates=[candidate(str(i)) for i in range(129)]),
                       lambda r: r["candidates"][0]["estimated_cost"].update(tool_calls=2),
                       lambda r: r["candidates"][0]["arguments"].update(provider="fabric+weknora"),
                       lambda r: r["candidates"][0].update(covers=[{}])):
            req = request(); mutate(req)
            with self.assertRaises(ValueError):
                radar_policy.decide(req)

    def test_default_import_has_no_host_src_collision_or_torch_download(self):
        script = """import importlib.util, sys, types
src = types.ModuleType('src'); src.marker = 'host'; sys.modules['src'] = src
spec=importlib.util.spec_from_file_location('radar_policy', %r)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
r=m.decide(%r)
assert r['decision']=='continue'
assert sys.modules['src'] is src
assert not any(k in sys.modules for k in ('torch','sentence_transformers','transformers'))
""" % (str(ROOT / "radar_policy.py"), request())
        process = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, cwd="/tmp")
        self.assertEqual(process.returncode, 0, process.stderr)


class PublicLoaderTests(unittest.TestCase):
    def row(self):
        return {"schema_version": "task-construction-v2", "qid": "q1", "question": "Drug A targets what?",
            "qtype": "constrained_entity", "environment_ref": "public-world", "budget": {"relation_access": 8, "tool_calls": 12},
            "max_steps": 8, "sufficiency_threshold": 1, "note": ""}

    def test_loader_accepts_real_public_contract_without_gold(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "public" / "planner_questions.jsonl"; path.parent.mkdir()
            path.write_text(json.dumps(self.row()) + "\n")
            q = radar_policy.load_public_questions(path)[0]
            self.assertEqual(q.requirement_groups, [])
            self.assertEqual(q.budget, self.row()["budget"])
            self.assertEqual(q.note, "")
            self.assertEqual(q.environment_ref, "public-world")

    def test_unmapped_factory_budget_is_explicitly_unsupported(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "public" / "planner_questions.jsonl"; path.parent.mkdir()
            row = self.row(); row["budget"] = {"ctd_access": 4}
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "unsupported_budget.*ctd_access"):
                radar_policy.load_public_questions(path)

    def test_loader_refuses_old_exploration_hidden_and_private_fields(self):
        with tempfile.TemporaryDirectory() as root:
            public = Path(root) / "public"; public.mkdir()
            for name in ("planner_training_samples.jsonl", "generation_traces.jsonl"):
                path = public / name; path.write_text(json.dumps(self.row()))
                with self.assertRaises(ValueError): radar_policy.load_public_questions(path)
            path = public / "planner_questions.jsonl"
            for private in ("answer", "gold", "seed_entity_id", "hop1", "requirement_groups"):
                row = self.row(); row[private] = "hint"; path.write_text(json.dumps(row))
                with self.assertRaises(ValueError): radar_policy.load_public_questions(path)
            hidden = Path(root) / "hidden" / "public" / "planner_questions.jsonl"
            hidden.parent.mkdir(parents=True); hidden.write_text(json.dumps(self.row()))
            with self.assertRaises(ValueError): radar_policy.load_public_questions(hidden)


if __name__ == "__main__":
    unittest.main()
