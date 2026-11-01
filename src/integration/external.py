"""Incremental host policy. Selectors and stop rules remain the D implementations.

The host owns observations, candidate construction, permissions, actual execution,
remaining budgets and durable receipts. ``complete`` means supplied retrieval
checks are met; it never asserts a clinical or scientific conclusion.
"""
from __future__ import annotations
import copy
import hashlib
import json
from dataclasses import dataclass
import numpy as np
from ..schema import Action, ActionCost, Decision
from ..state_construction.models import PolicyState
from ..action_selection.selector import HeuristicSelector, LinearModelSelector, ModelSelector
from ..stopping.stop import RuleStop, HierarchicalModelStop

SCHEMA_VERSION = "query-planner.external.v1"
DIMENSIONS = ("tool_calls", "tokens", "relation_access")
KINDS = {"document_search": "passage_retrieve", "entity_relations": "entity_lookup",
         "graph_neighbors": "relation_expand", "graph_path": "bounded_path"}
HISTORY_STATUSES = {"prepared", "sent", "unconfirmed", "unknown", "completed", "failed", "not_sent"}
MAX_CANDIDATES, MAX_HISTORY, MAX_CHECKS = 128, 256, 64


def _text(value, name, maximum=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(name + " must be nonempty bounded text")
    return value


def _number(value, name, maximum=10_000_000):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(name + " must be a bounded nonnegative integer")
    return value


def _list(value, name, maximum):
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(name + " must be a bounded list")
    return value


def _canonical_arguments(value, key="", depth=0):
    if depth > 12:
        raise ValueError("arguments nesting exceeds 12 levels")
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("argument object keys must be strings")
        return {k: _canonical_arguments(v, k, depth + 1) for k, v in sorted(value.items())}
    if isinstance(value, list):
        values = [_canonical_arguments(v, key, depth + 1) for v in value]
        if key in {"terms", "source_ids", "knowledge_base_ids"} and all(isinstance(v, str) for v in values):
            return sorted(set(values))
        return values
    if isinstance(value, str) and key in {"query", "terms"}:
        return " ".join(value.casefold().split())
    if value is not None and not isinstance(value, (str, bool, int, float)):
        raise ValueError("arguments must be JSON values")
    return value


def semantic_key(kind, arguments):
    if kind not in KINDS or not isinstance(arguments, dict):
        raise ValueError("action kind or arguments invalid")
    raw = json.dumps([kind, _canonical_arguments(arguments)], sort_keys=True,
                     ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(raw.encode()) > 16000:
        raise ValueError("action arguments exceed 16000 bytes")
    return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class HostAction(Action):
    coverage_ids: tuple[str, ...] = ()
    cost_pressure: float = 0.0


@dataclass
class HostState(PolicyState):
    external_checks: tuple[tuple[str, str], ...] = ()
    budget_axis: str = "tool_calls"
    evidence_count: int = 0


class HostEncoder:
    """A versioned small observed-state vector; no pretrained encoder is loaded."""
    name, dim = "external_observations_v1", 12

    def encode(self, state):
        checks = state.external_checks
        passed = sum(status == "passed" for _, status in checks)
        ratios = [state.remaining.get(d, 0) / max(state.budget_limits.get(d, 0), 1)
                  for d in DIMENSIONS]
        return np.asarray([min(len(checks) / 64, 1), passed / max(len(checks), 1),
            min(state.evidence_count / 100, 1), min(state.no_progress_streak / 4, 1),
            min(state.step_index / max(state.max_steps, 1), 1),
            min((state.legal_action_count or 0) / 128, 1),
            min((state.affordable_action_count or 0) / 128, 1),
            *ratios, float("document_search" in state.recent_actions),
            float(any(k.startswith("graph_") for k in state.recent_actions))], dtype=np.float32)


def _inputs(request):
    if not isinstance(request, dict) or request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported policy schema_version")
    question = _text(request.get("question"), "question")
    evidence = _list(request.get("evidence", []), "evidence", 256)
    evidence_ids = {_text(e.get("id"), "evidence.id", 400) for e in evidence if isinstance(e, dict)}
    if len(evidence_ids) != len(evidence):
        raise ValueError("evidence must contain unique object IDs")
    checks, check_ids, warnings = [], set(), []
    for check in _list(request.get("checks", []), "checks", MAX_CHECKS):
        if not isinstance(check, dict):
            raise ValueError("check must be an object")
        key = _text(check.get("id"), "check.id", 200)
        if key in check_ids or check.get("status") not in {"passed", "failed", "unknown"}:
            raise ValueError("duplicate check or invalid check status")
        check_ids.add(key)
        if type(check.get("required", True)) is not bool:
            raise ValueError("check.required must be boolean")
        status = check["status"]
        if status == "passed":
            refs = check.get("evidence_ids", [])
            if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) or ref not in evidence_ids for ref in refs):
                status = "unknown"
                warnings.append("passed_check_without_current_evidence:" + key)
        if check.get("required", True):
            checks.append((key, status))
    if not checks:
        checks = [("retrieval_scope_not_specified", "unknown")]
        warnings.append("missing_required_checks")
    budget = request.get("budget")
    if not isinstance(budget, dict) or not isinstance(budget.get("remaining"), dict) or not isinstance(budget.get("limits"), dict):
        raise ValueError("budget.remaining and budget.limits are required")
    remaining, limits = budget["remaining"], budget["limits"]
    if set(remaining) != set(limits) or set(remaining) - set(DIMENSIONS):
        raise ValueError("budget dimensions must match and be supported")
    for dim in remaining:
        _number(remaining[dim], "remaining." + dim)
        _number(limits[dim], "limits." + dim)
        if remaining[dim] > limits[dim]:
            raise ValueError("remaining exceeds original budget")
    history = _list(request.get("history", []), "history", MAX_HISTORY)
    attempted, ids, outstanding = set(), set(), []
    for entry in history:
        if not isinstance(entry, dict) or entry.get("status") not in HISTORY_STATUSES:
            raise ValueError("invalid history status")
        aid = _text(entry.get("action_id"), "history.action_id", 200)
        if aid in ids:
            raise ValueError("history requires one latest receipt per unique action_id")
        ids.add(aid)
        attempted.add(semantic_key(entry.get("kind"), entry.get("arguments")))
        yielded = entry.get("new_evidence_ids", [])
        if not isinstance(yielded, list) or any(not isinstance(x, str) for x in yielded):
            raise ValueError("history.new_evidence_ids must be IDs")
        if entry["status"] in {"sent", "unconfirmed", "unknown", "prepared"}:
            outstanding.append(aid)
    max_steps = _number(request.get("max_steps", 12), "max_steps", MAX_HISTORY)
    if max_steps == 0:
        raise ValueError("max_steps must be positive")
    candidates, originals, excluded, seen = [], {}, [], set()
    for item in sorted(_list(request.get("candidates", []), "candidates", MAX_CANDIDATES),
                       key=lambda x: str(x.get("id", "")) if isinstance(x, dict) else ""):
        if not isinstance(item, dict):
            raise ValueError("candidate must be an object")
        aid = _text(item.get("id"), "candidate.id", 200)
        if aid in originals:
            raise ValueError("candidate IDs must be unique")
        originals[aid] = item
        kind, arguments = item.get("kind"), item.get("arguments")
        key = semantic_key(kind, arguments)
        reason = None
        if aid in ids or key in attempted:
            reason = "already_attempted"
        elif key in seen:
            reason = "duplicate_semantic_action"
        seen.add(key)
        covers = item.get("covers", [])
        if (not isinstance(covers, list) or any(not isinstance(c, str) for c in covers)
                or len(set(covers)) != len(covers) or any(c not in check_ids for c in covers)):
            raise ValueError("candidate.covers must reference unique declared check IDs")
        costs = item.get("estimated_cost")
        if not isinstance(costs, dict) or set(costs) - set(DIMENSIONS):
            raise ValueError("candidate estimated_cost invalid")
        for dim, cost in costs.items():
            if cost is not None:
                _number(cost, "estimated_cost." + dim)
        if costs.get("tool_calls") is None or costs["tool_calls"] < 1:
            reason = reason or "unknown_call_cost"
        if any(costs.get(dim) is None for dim in remaining):
            reason = reason or "unknown_limited_cost"
        if kind == "document_search":
            if arguments.get("provider") not in {"fabric", "weknora"}:
                raise ValueError("document_search requires one explicit fabric or weknora provider")
            if costs.get("tool_calls") != 1:
                raise ValueError("document_search denotes one provider query, tool_calls must be 1")
            query, terms = arguments.get("query"), arguments.get("terms")
            if bool(query) == bool(terms):
                raise ValueError("document_search requires one query or one terms list")
            if query:
                _text(query, "query", 1000)
            if terms and (not isinstance(terms, list) or not 1 <= len(terms) <= 8
                          or any(not isinstance(t, str) or not t.strip() for t in terms)):
                raise ValueError("document terms invalid")
        if reason:
            excluded.append({"id": aid, "reason": reason})
            continue
        pressure = max((costs[d] / max(remaining[d], 1) for d in remaining), default=0.0)
        # Missing unlimited dimensions stay missing in public output. These internal
        # zero placeholders are never billed, reported as measured, or affordability-tested.
        cost = ActionCost(**{d: costs.get(d) or 0 for d in DIMENSIONS})
        candidates.append(HostAction(aid, KINDS[kind], (), cost, tuple(covers), min(pressure, 1.0)))
    affordable = [a for a in candidates if all(a.cost.as_dict()[d] <= remaining[d] for d in remaining)]
    streak = 0
    for entry in reversed(history):
        if entry["status"] in {"prepared", "sent", "unconfirmed", "unknown", "not_sent"}:
            continue
        if entry.get("new_evidence_ids"):
            break
        streak += 1
    axis = "tool_calls" if "tool_calls" in remaining else next(iter(sorted(remaining)), "tool_calls")
    state = HostState(qid=str(request.get("question_id", "external")), question=question,
        qtype="external", seeds=(), targets=(), relation_keywords=(), evidence=(), known_entities=(),
        relations_seen={}, looked_up=frozenset(), expanded=frozenset(), tried_paths=frozenset(),
        tried_passages=False, remaining=dict(remaining), budget_limits=dict(limits),
        step_index=len(history), sufficiency_threshold=1.0, max_steps=max_steps,
        no_progress_streak=streak, recent_actions=tuple(e["kind"] for e in history[-3:]),
        external_checks=tuple(checks), budget_axis=axis, evidence_count=len(evidence_ids),
        legal_action_count=len(candidates), affordable_action_count=len(affordable),
        candidate_actions=tuple(candidates))
    return state, originals, excluded, warnings, outstanding


class HostPolicy:
    def __init__(self, *, selector="heuristic", stop="rule", version=SCHEMA_VERSION):
        encoder = HostEncoder()
        selectors = {"heuristic": HeuristicSelector, "linear": LinearModelSelector, "model": ModelSelector}
        stops = {"rule": RuleStop, "model": HierarchicalModelStop}
        if selector not in selectors or stop not in stops:
            raise ValueError("unsupported selector or stop")
        self.selector = selectors[selector]() if selector == "heuristic" else selectors[selector](encoder=encoder)
        self.stop = stops[stop]() if stop == "rule" else stops[stop](encoder=encoder)
        self.version = version
        self.selector_name, self.stop_name = selector, stop

    def decide(self, request):
        state, originals, excluded, warnings, outstanding = _inputs(request)
        selected = None
        if outstanding:
            decision, reason = Decision.INSUFFICIENT, "pending_execution_or_reconciliation"
        else:
            decision, reason = self.stop.decide(state)
            # Learned early stopping may decline more work, but cannot certify
            # host retrieval checks that have not actually passed.
            if decision is Decision.COMPLETE and any(s != "passed" for _, s in state.external_checks):
                decision, reason = Decision.INSUFFICIENT, "retrieval_checks_not_satisfied"
            if decision is Decision.CONTINUE:
                selection = self.selector.select(state, state.candidate_actions)
                selected = selection.action
                if selected is None:
                    decision, reason = Decision.INSUFFICIENT, selection.blocked_reason
                else:
                    reason = "selected_by_" + self.selector_name
        learned = any(getattr(model, "use_heuristic_base", True) is False for model in (self.stop, self.selector))
        return {"schema_version": SCHEMA_VERSION, "decision": decision.value,
            "action": copy.deepcopy(originals[selected.aid]) if selected else None,
            "reason": reason,
            "policy": {"version": self.version, "selector": self.selector_name, "stop": self.stop_name,
                "encoder": HostEncoder.name, "training_status": "trained_in_memory" if learned else "heuristic_untrained",
                "scope": "retrieval_coverage_only_not_scientific_approval"},
            "state_summary": {"checks": [{"id": k, "status": s} for k, s in state.external_checks],
                "evidence_count": state.evidence_count, "step": state.step_index,
                "legal_action_count": state.legal_action_count,
                "affordable_action_count": state.affordable_action_count,
                "remaining": copy.deepcopy(request["budget"]["remaining"]),
                "no_progress_streak": state.no_progress_streak,
                "excluded_candidates": excluded, "pending_action_ids": outstanding, "warnings": warnings}}


def decide(request):
    return HostPolicy().decide(request)
