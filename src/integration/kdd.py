"""Pure, observation-dependent routing over host-registered KDD capabilities.

This contract is separate from external.v1 and the original graph baseline.
The host constructs legal candidates, checks inputs/permissions, executes one
operation and adopts its durable receipt before asking for another decision.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from .external import DIMENSIONS, HISTORY_STATUSES, _canonical_arguments, _list, _number, _text

SCHEMA_VERSION = "query-planner.external.v2"
METHOD_VERSION = "kdd-information-gain-1.0"
DIRECTIONS = {"pipeline", "research", "business", "public_health"}
RETRIEVAL_KINDS = {"document_search", "entity_relations", "graph_neighbors", "graph_path"}
CAPABILITY_KINDS = {"dataset_prepare", "analysis_run", "professional_assess"}
KINDS = RETRIEVAL_KINDS | CAPABILITY_KINDS
SELECTORS = {"information_gain_v1", "coverage_cost_baseline_v1"}
PENDING = {"prepared", "sent", "unknown", "unconfirmed"}


def _unit(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + " must be finite in [0, 1]")
    return float(value)


def _objects(value, name, maximum):
    items = _list(value, name, maximum)
    if any(not isinstance(item, dict) for item in items):
        raise ValueError(name + " must contain objects")
    return items


def _ids(value, name, maximum=64):
    items = _list(value, name, maximum)
    for item in items:
        _text(item, name, 400)
    if len(set(items)) != len(items):
        raise ValueError(name + " must be unique")
    return items


def _refs(value, name):
    refs = _objects(value, name, 256)
    keys = [(_text(r.get("id"), name + ".id", 400), _text(r.get("version"), name + ".version", 400)) for r in refs]
    if len(set(keys)) != len(keys):
        raise ValueError(name + " must be unique versioned references")
    return set(keys)


def semantic_key(kind, arguments):
    if kind not in KINDS or not isinstance(arguments, dict):
        raise ValueError("invalid KDD action kind or arguments")
    # References are a set; reordered equivalent input versions cannot bypass receipts.
    normalized = _canonical_arguments(arguments)
    if "input_refs" in normalized:
        _refs(normalized["input_refs"], "arguments.input_refs")
        normalized["input_refs"] = sorted(normalized["input_refs"], key=lambda r: (r["id"], r["version"]))
    raw = json.dumps([kind, normalized], sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    if len(raw.encode()) > 16000:
        raise ValueError("action arguments exceed 16000 bytes")
    return hashlib.sha256(raw.encode()).hexdigest()


def _family(action):
    args = action["arguments"]
    return (action["kind"], args.get("capability_id", ""), args.get("capability_version", ""), args.get("provider", ""))


def _current_support(item, evidence, artifacts):
    ids = _ids(item.get("evidence_ids", []), "evidence_ids", 256)
    refs = _refs(item.get("artifact_refs", []), "artifact_refs")
    return bool(ids or refs) and set(ids).issubset(evidence) and refs.issubset(artifacts)


def _validate(request):
    if not isinstance(request, dict) or request.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported KDD schema_version")
    _text(request.get("question"), "question")
    direction = request.get("direction")
    if direction not in DIRECTIONS:
        raise ValueError("direction must be pipeline/research/business/public_health")
    scope = request.get("scope", {})
    if not isinstance(scope, dict):
        raise ValueError("scope must be an object")
    if scope.get("direction", direction) != direction:
        raise ValueError("scope direction differs from task direction")
    if scope.get("as_of") is not None:
        _text(scope["as_of"], "scope.as_of", 100)
    evidence_rows = _objects(request.get("evidence", []), "evidence", 256)
    evidence = {_text(e.get("id"), "evidence.id", 400) for e in evidence_rows}
    if len(evidence) != len(evidence_rows):
        raise ValueError("evidence IDs must be unique")
    artifact_rows = _objects(request.get("artifacts", []), "artifacts", 256)
    artifact_keys = _refs(artifact_rows, "artifacts")
    artifacts = {(a["id"], a["version"]): a for a in artifact_rows}
    for a in artifact_rows:
        _text(a.get("kind"), "artifact.kind", 200)
    capabilities = {}
    for cap in _objects(request.get("capabilities", []), "capabilities", 128):
        cid = _text(cap.get("id"), "capability.id", 200)
        if cid in capabilities or cap.get("kind") not in KINDS:
            raise ValueError("duplicate capability ID or unsupported capability kind")
        _text(cap.get("version"), "capability.version", 200)
        directions = _ids(cap.get("directions", []), "capability.directions", 4)
        if not directions or set(directions) - DIRECTIONS:
            raise ValueError("capability requires supported directions")
        _ids(cap.get("input_requirements", []), "capability.input_requirements")
        capabilities[cid] = cap
    checks, warnings = {}, []
    for raw in _objects(request.get("checks", []), "checks", 64):
        check = copy.deepcopy(raw)
        cid = _text(check.get("id"), "check.id", 200)
        if cid in checks or check.get("status") not in {"passed", "failed", "unknown"}:
            raise ValueError("duplicate check or invalid check status")
        if type(check.get("required", True)) is not bool:
            raise ValueError("check.required must be boolean")
        check["importance"] = _unit(check.get("importance", 1), "check.importance")
        _ids(check.get("missing_inputs", []), "check.missing_inputs")
        _ids(check.get("acceptable_alternatives", []), "check.acceptable_alternatives")
        if check["status"] == "passed" and not _current_support(check, evidence, artifact_keys):
            check["status"] = "unknown"
            warnings.append("passed_check_without_current_support:" + cid)
        checks[cid] = check
    required = {cid for cid, c in checks.items() if c.get("required", True)}
    if not required:
        warnings.append("missing_required_checks")
    budget = request.get("budget", {})
    remaining, limits = budget.get("remaining"), budget.get("limits")
    if not isinstance(remaining, dict) or not isinstance(limits, dict) or set(remaining) != set(limits) or set(remaining) - set(DIMENSIONS):
        raise ValueError("matching supported budget.remaining and budget.limits required")
    for dim in remaining:
        _number(remaining[dim], "remaining." + dim)
        _number(limits[dim], "limits." + dim)
        if remaining[dim] > limits[dim]:
            raise ValueError("remaining exceeds original budget")
    max_steps = _number(request.get("max_steps", 24), "max_steps", 256)
    if max_steps == 0:
        raise ValueError("max_steps must be positive")
    history = _objects(request.get("history", []), "history", 256)
    attempted, history_ids, pending, outcomes = set(), set(), [], []
    for entry in history:
        aid = _text(entry.get("action_id"), "history.action_id", 200)
        if aid in history_ids or entry.get("status") not in HISTORY_STATUSES:
            raise ValueError("one latest valid receipt per action_id required")
        history_ids.add(aid)
        attempted.add(semantic_key(entry.get("kind"), entry.get("arguments")))
        new_ids = _ids(entry.get("new_evidence_ids", []), "new_evidence_ids", 256)
        new_refs = _refs(entry.get("new_artifact_refs", []), "new_artifact_refs")
        updates = _objects(entry.get("check_updates", []), "check_updates", 64)
        for update in updates:
            if update.get("id") not in checks or update.get("status") not in {"passed", "failed", "unknown"}:
                raise ValueError("history check_updates must reference declared checks")
        information = entry.get("observed_information", {})
        if not isinstance(information, dict) or set(information) - set(checks):
            raise ValueError("observed_information must reference declared checks")
        for cid, gain in information.items():
            _unit(gain, "observed_information." + cid)
        if entry["status"] in PENDING:
            pending.append(aid)
            continue
        # Only an actual terminal receipt with current supporting objects informs
        # successful yield. A completed tool call alone is not information gain.
        support = (bool(new_ids or new_refs) and set(new_ids).issubset(evidence) and new_refs.issubset(artifact_keys))
        support = support or any(u["status"] == "passed" and _current_support(u, evidence, artifact_keys) for u in updates)
        if entry["status"] == "failed":
            outcomes.append((_family(entry), {}, True))
        elif entry["status"] == "completed":
            if information and not support and any(information.values()):
                warnings.append("observed_information_without_current_support:" + aid)
            gains = information if support else {cid: 0.0 for cid in information}
            # A receipt explicitly containing zero gain is informative; absence
            # of a gain estimate is not silently converted into a zero measurement.
            if gains:
                outcomes.append((_family(entry), gains, False))
    candidates, excluded, seen, candidate_ids = [], [], set(), set()
    for raw in sorted(_objects(request.get("candidates", []), "candidates", 128), key=lambda x: str(x.get("id", ""))):
        aid = _text(raw.get("id"), "candidate.id", 200)
        if aid in candidate_ids:
            raise ValueError("candidate IDs must be unique")
        candidate_ids.add(aid)
        kind, args = raw.get("kind"), raw.get("arguments")
        key = semantic_key(kind, args)
        reason = "already_attempted" if aid in history_ids or key in attempted else ("duplicate_semantic_action" if key in seen else None)
        seen.add(key)
        covers = _ids(raw.get("covers", []), "candidate.covers")
        if set(covers) - set(checks):
            raise ValueError("covers must reference declared checks")
        costs = raw.get("estimated_cost")
        if not isinstance(costs, dict) or set(costs) - set(DIMENSIONS):
            raise ValueError("estimated_cost must use supported dimensions")
        for dim, value in costs.items():
            if value is not None:
                _number(value, "estimated_cost." + dim)
        if costs.get("tool_calls") is None or costs["tool_calls"] < 1:
            reason = reason or "unknown_call_cost"
        if any(costs.get(dim) is None for dim in remaining):
            reason = reason or "unknown_limited_cost"
        applicability = raw.get("applicability", {"status": "eligible", "scope_match": 1})
        if not isinstance(applicability, dict) or applicability.get("status") not in {"eligible", "ineligible", "unknown"}:
            raise ValueError("invalid applicability")
        match = _unit(applicability.get("scope_match", 1), "applicability.scope_match")
        if applicability["status"] != "eligible":
            reason = reason or "capability_not_eligible"
        if kind == "document_search":
            if args.get("provider") not in {"fabric", "weknora"} or costs.get("tool_calls") != 1:
                raise ValueError("document_search requires one fabric/weknora query with tool_calls=1")
            query, terms = args.get("query"), args.get("terms")
            if bool(query) == bool(terms):
                raise ValueError("document_search requires query or terms")
            if query:
                _text(query, "query", 1000)
            else:
                if not 1 <= len(_ids(terms, "terms", 8)) <= 8:
                    raise ValueError("terms must not be empty")
        if kind in CAPABILITY_KINDS:
            cid = _text(args.get("capability_id"), "arguments.capability_id", 200)
            if raw.get("capability_id") != cid:
                raise ValueError("candidate and arguments capability_id must match")
            version = _text(args.get("capability_version"), "arguments.capability_version", 200)
            refs = _refs(args.get("input_refs"), "arguments.input_refs")
            action_scope = args.get("scope")
            if not isinstance(action_scope, dict) or not isinstance(args.get("parameters"), dict):
                raise ValueError("capability actions require scope and parameters objects")
            if "applicability" not in raw:
                reason = reason or "missing_applicability"
            cap = capabilities.get(cid)
            if cap is None:
                reason = reason or "unregistered_capability"
            elif cap["kind"] != kind or cap["version"] != version:
                reason = reason or "capability_version_or_kind_mismatch"
            elif direction not in cap["directions"]:
                reason = reason or "direction_not_supported"
            if action_scope.get("direction") != direction or ("as_of" in scope and action_scope.get("as_of") != scope["as_of"]):
                reason = reason or "scope_mismatch"
            if not refs.issubset(artifact_keys):
                reason = reason or "input_version_unavailable"
            elif cap is not None and set(cap.get("input_requirements", [])) - {artifacts[r]["kind"] for r in refs}:
                reason = reason or "missing_required_input_kind"
        estimates = raw.get("expected_information")
        if estimates is None:
            estimates = [{"check_id": cid, "probability": 0.5, "impact": 1.0} for cid in covers]
            prior_source = "untrained_default_prior"
        else:
            estimates = _objects(estimates, "expected_information", 64)
            prior_source = "host_estimate_not_observation"
        expected = {}
        for estimate in estimates:
            cid = estimate.get("check_id")
            if cid not in covers or cid in expected:
                raise ValueError("expected_information must uniquely reference covers")
            expected[cid] = (_unit(estimate.get("probability"), "probability"), _unit(estimate.get("impact"), "impact"))
        if reason:
            excluded.append({"id": aid, "reason": reason})
        else:
            candidates.append({"action": raw, "expected": expected, "scope_match": match, "prior_source": prior_source})
    return dict(checks=checks, required=required, candidates=candidates, excluded=excluded, warnings=warnings,
                remaining=remaining, pending=pending, outcomes=outcomes, history=history, max_steps=max_steps,
                evidence_count=len(evidence), artifact_count=len(artifacts))


def _score(candidate, state, selector):
    action, parts = candidate["action"], []
    family = _family(action)
    for cid, (prior, impact) in candidate["expected"].items():
        check = state["checks"][cid]
        if check["status"] == "passed":
            continue
        observations = [0.0 if failed else gains[cid] for f, gains, failed in state["outcomes"] if f == family and (failed or cid in gains)]
        posterior = (2 * prior + sum(observations)) / (2 + len(observations))
        value = check["importance"] * posterior * impact * candidate["scope_match"]
        parts.append({"check_id": cid, "importance": check["importance"], "prior": prior,
                      "posterior": round(posterior, 6), "receipt_count": len(observations),
                      "impact": impact, "expected_contribution": round(value, 6)})
    pressure = max((action["estimated_cost"][d] / max(state["remaining"][d], 1) for d in state["remaining"]), default=0)
    if selector == "coverage_cost_baseline_v1":
        gain = sum(state["checks"][cid]["status"] != "passed" for cid in action.get("covers", []))
        score = gain / (1 + pressure)
    else:
        gain = sum(p["expected_contribution"] for p in parts)
        score = gain / (1 + 0.25 * pressure)
    return {"id": action["id"], "score": round(score, 8), "expected_contribution": round(gain, 6),
            "cost_pressure": round(pressure, 6), "prior_source": candidate["prior_source"], "checks": parts}


def decide(request, *, selector="information_gain_v1", version=METHOD_VERSION):
    if selector not in SELECTORS:
        raise ValueError("unsupported KDD selector")
    state = _validate(request)
    affordable = [c for c in state["candidates"] if all(c["action"]["estimated_cost"][d] <= state["remaining"][d] for d in state["remaining"])]
    ranked = sorted([_score(c, state, selector) for c in affordable], key=lambda r: (-r["score"], r["id"]))
    unmet = [cid for cid in state["required"] if state["checks"][cid]["status"] != "passed"]
    selected = None
    if state["pending"]:
        decision, reason = "insufficient", "pending_execution_or_reconciliation"
    elif state["required"] and not unmet:
        decision, reason = "complete", "required_observations_satisfied"
    elif len(state["history"]) >= state["max_steps"]:
        decision, reason = "insufficient", "max_steps_reached"
    elif not state["candidates"]:
        decision, reason = "insufficient", "no_eligible_action"
    elif not affordable:
        decision, reason = "insufficient", "budget_exhausted"
    elif not ranked or ranked[0]["score"] <= 0:
        decision, reason = "insufficient", "no_expected_information_gain"
    else:
        decision, reason = "continue", "selected_by_" + selector
        selected = next(c["action"] for c in affordable if c["action"]["id"] == ranked[0]["id"])
    gaps = [{"check_id": cid, "missing_inputs": state["checks"][cid].get("missing_inputs", []),
             "acceptable_alternatives": state["checks"][cid].get("acceptable_alternatives", []),
             "candidate_ids": [c["action"]["id"] for c in affordable if cid in c["action"].get("covers", [])]}
            for cid in sorted(unmet)]
    return {"schema_version": SCHEMA_VERSION, "decision": decision, "reason": reason,
            "action": copy.deepcopy(selected),
            "policy": {"version": version, "method_version": METHOD_VERSION, "selector": selector,
                       "training_status": "heuristic_untrained", "encoder": "kdd_observed_information_v1",
                       "scope": "host_observation_coverage_only_not_scientific_or_business_approval"},
            "state_summary": {"direction": request["direction"], "checks": list(state["checks"].values()),
                "evidence_count": state["evidence_count"], "artifact_count": state["artifact_count"],
                "step": len(state["history"]), "legal_action_count": len(state["candidates"]),
                "affordable_action_count": len(affordable), "remaining": copy.deepcopy(state["remaining"]),
                "pending_action_ids": state["pending"], "excluded_candidates": state["excluded"],
                "warnings": state["warnings"], "unmet_gaps": gaps, "candidate_scores": ranked}}
