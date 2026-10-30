"""Public-only factory question ingestion. Gold is never needed by the parser.

Only task-construction-v2 planner_questions.jsonl is accepted. This loader never
opens a sibling manifest, hidden directory, exploration trace or evaluation file.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from ..schema import Question

PUBLIC_FIELDS = {"schema_version", "qid", "question", "qtype", "environment_ref",
                 "budget", "max_steps", "sufficiency_threshold", "note"}


@dataclass
class PublicQuestion(Question):
    environment_ref: str = ""


def load_public_questions(path) -> list[Question]:
    source = Path(path).resolve(strict=True)
    if source.name != "planner_questions.jsonl" or source.parent.name != "public":
        raise ValueError("expected release/public/planner_questions.jsonl; legacy exploration is not solver input")
    if "hidden" in source.parts or source.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("hidden or oversized question file")
    questions, seen = [], set()
    with source.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if (not isinstance(row, dict) or set(row) - PUBLIC_FIELDS
                    or row.get("schema_version") != "task-construction-v2"):
                raise ValueError(f"line {line_no}: unsupported public schema or private fields")
            for key in ("qid", "question", "qtype", "environment_ref"):
                if not isinstance(row.get(key), str) or not row[key].strip() or len(row[key]) > 4000:
                    raise ValueError(f"line {line_no}: invalid {key}")
            if row["qid"] in seen:
                raise ValueError("duplicate qid")
            seen.add(row["qid"])
            budget = row.get("budget")
            if isinstance(budget, dict) and set(budget) - {"relation_access", "tool_calls", "tokens"}:
                raise ValueError("unsupported_budget: no unit mapping for " + ",".join(sorted(set(budget) - {"relation_access", "tool_calls", "tokens"})))
            if (not isinstance(budget, dict) or not budget
                    or set(budget) - {"relation_access", "tool_calls", "tokens"}
                    or any(type(v) is not int or not 0 <= v <= 10_000_000 for v in budget.values())):
                raise ValueError("invalid public budget")
            steps = row.get("max_steps", 8)
            threshold = row.get("sufficiency_threshold", 1.0)
            if type(steps) is not int or not 1 <= steps <= 256:
                raise ValueError("max_steps must be 1..256")
            if (type(threshold) not in (int, float) or not math.isfinite(threshold)
                    or not 0 < threshold <= 1):
                raise ValueError("invalid sufficiency_threshold")
            note = row.get("note", "")
            if not isinstance(note, str) or len(note) > 4000:
                raise ValueError("invalid public note")
            questions.append(PublicQuestion(qid=row["qid"], question=row["question"], qtype=row["qtype"],
                budget=dict(budget), requirement_groups=[], max_steps=steps,
                sufficiency_threshold=float(threshold), note=note, environment_ref=row["environment_ref"]))
            if len(questions) > 10_000:
                raise ValueError("public batch exceeds 10000 questions")
    if not questions:
        raise ValueError("empty public questions")
    return questions
