#!/usr/bin/env python3
"""采集 BioHopR + PrimeKG 轨迹。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.biohopr import load_biohopr
from src.primekg import PrimeKG
from src.pipeline import run_question
from src.pipeline import build_pipeline
from src.trajectory import Trajectory, append_jsonl
from src.training.checkpoint import load_selector_checkpoint, load_stop_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kg-csv",
        default=r"E:\Cursor\assumption\datasets\PrimeKG\kg.csv",
        help="PrimeKG kg.csv 路径",
    )
    ap.add_argument(
        "--db",
        default=r"E:\Cursor\assumption\datasets\PrimeKG\primekg.sqlite",
        help="SQLite 索引路径",
    )
    ap.add_argument(
        "--biohopr",
        default=r"E:\Cursor\assumption\datasets\BioHopR\BioHopR.json",
        help="BioHopR.json 路径",
    )
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "trajectories" / "biohopr_rule.jsonl"))
    ap.add_argument("--parser", default="biohopr")
    ap.add_argument("--encoder", default="feature")
    ap.add_argument("--stop", dest="stop_name", default="rule")
    ap.add_argument("--selector", default="heuristic")
    ap.add_argument("--executor", default="primekg")
    ap.add_argument("--stop-checkpoint", default=None)
    ap.add_argument("--selector-checkpoint", default=None)
    ap.add_argument("--build-index", action="store_true", help="若 SQLite 不存在则构建索引")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_path = Path(args.out)
    if args.overwrite and out_path.exists():
        out_path.unlink()

    primekg = PrimeKG(args.kg_csv, args.db, build_if_missing=args.build_index)
    records = load_biohopr(args.biohopr, limit=args.limit)
    pipeline = build_pipeline(
        primekg,
        parser=args.parser,
        encoder=args.encoder,
        stop=args.stop_name,
        selector=args.selector,
        executor=args.executor,
    )
    if args.stop_checkpoint:
        load_stop_checkpoint(pipeline.stop, args.stop_checkpoint)
    if args.selector_checkpoint:
        load_selector_checkpoint(pipeline.selector, args.selector_checkpoint)

    for i, record in enumerate(records, start=1):
        question = record.to_question()
        trace = run_question(question, primekg, pipeline, record_transitions=True)
        trajectory = Trajectory.from_run(
            trace,
            metadata={
                "gold_answer": list(record.answer),
                "hop1": record.hop1,
                "hop2": record.hop2,
                "relation_hop1": record.relation_hop1,
                "relation_hop2": record.relation_hop2,
                "target_type": record.target_type,
            },
        )
        append_jsonl(out_path, trajectory.to_dict())
        print(
            f"[{i}/{len(records)}] {record.qid} | "
            f"{trace.final_decision.value} | {trace.final_reason} | "
            f"steps={len(trace.steps)} ra={trace.spent.relation_access}"
        )

    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
