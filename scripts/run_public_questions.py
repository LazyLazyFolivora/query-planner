#!/usr/bin/env python3
"""Run fresh offline solver trajectories from the factory's public-only release.

No training/generation traces or hidden evaluation targets are read. This is an
integration baseline, not biomedical validation or proof of a learned policy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.integration.public_questions import load_public_questions
from src.kg import ToyKG
from src.primekg import PrimeKG
from src.pipeline import build_pipeline, run_question
from src.trajectory import Trajectory


class ReadOnlyPrimeKG(PrimeKG):
    def _conn(self):
        return sqlite3.connect(self.db_path.resolve().as_uri() + "?mode=ro", uri=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--backend", choices=("toy", "primekg"), required=True)
    parser.add_argument("--world", required=True, help="public ToyKG JSON or existing PrimeKG SQLite")
    parser.add_argument("--environment-ref", required=True, help="explicit public environment identity bound to --world")
    parser.add_argument("--out", required=True, help="new JSONL output; refuses overwrite")
    args = parser.parse_args()
    questions = load_public_questions(args.questions)
    if any(q.environment_ref != args.environment_ref for q in questions):
        parser.error("unsupported_environment: public environment_ref does not match configured --world binding")
    if any(not {"relation_access", "tool_calls"}.issubset(q.budget) for q in questions):
        parser.error("offline graph runner requires explicit relation_access and tool_calls budgets")
    if args.backend == "toy" and any("tokens" not in q.budget for q in questions):
        parser.error("toy passage retrieval requires an explicit tokens budget; missing is not zero")
    # PrimeKG executes only local graph operations and measures zero model tokens.
    world = Path(args.world).resolve(strict=True)
    if "hidden" in world.parts:
        parser.error("world must be public retrieval data")
    kg = ToyKG.from_json(world) if args.backend == "toy" else ReadOnlyPrimeKG(world, world)
    pipeline = build_pipeline(kg, parser="rule", encoder="feature", stop="rule",
        selector="heuristic", executor="toy_kg" if args.backend == "toy" else "primekg")
    world_digest = hashlib.sha256()
    with world.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            world_digest.update(chunk)
    source = Path(args.questions).resolve()
    meta = {"input_contract": "task-construction-v2/public", "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "trajectory_origin": "independent_solver_rollout", "policy_training_status": "heuristic_untrained",
        "backend": args.backend, "world": str(world), "world_sha256": world_digest.hexdigest(),
        "environment_ref": args.environment_ref, "oracle_used_for_planning": False,
        "scope": "offline_integration_baseline_not_biomedical_validation"}
    output = Path(args.out).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for question in questions:
            trace = run_question(question, kg, pipeline, record_transitions=True)
            row = Trajectory.from_run(trace, metadata=meta).to_dict()
            row["metadata"]["final_triples"] = [list(e.triple.key()) for e in trace.final_evidence if e.triple is not None]
            row["metadata"]["declared_budget"] = question.budget
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"qid": question.qid, "decision": trace.final_decision.value,
                "actions": sum(t.get("action") is not None for t in trace.transitions),
                "evidence_count": trace.evidence_count, "spent": trace.spent.as_dict()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
