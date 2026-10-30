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
METHOD_VERSION = "kdd-information-gain-1.1"
# Pending receipts precede every stop; visible unread obligations prevent a
# coverage-only completion. A stop never certifies scientific/business validity.
STOP_VERSION = "observed_checks_reading_v2"
DIRECTIONS = {"pipeline", "research", "business", "public_health"}
RETRIEVAL_KINDS = {"document_search", "document_read", "entity_relations", "graph_neighbors", "graph_path"}
CAPABILITY_KINDS = {"dataset_prepare", "analysis_run", "professional_assess", "source_fetch"}
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


def _window_start(row, name):
    value = row.get("window_start", 0)
    if type(value) is not int or value < 0:
        raise ValueError(name + ".window_start must be a nonnegative Unicode character offset")
    return value


def _record_refs(value, name):
    rows = _objects(value, name, 32)
    keys = []
    for row in rows:
        if set(row) - {"source_record_id", "version_id", "window_start"} or not {"source_record_id", "version_id"} <= set(row):
            raise ValueError(name + " requires source_record_id/version_id and optional window_start only")
        keys.append((_text(row.get("source_record_id"), name + ".source_record_id", 400),
                     _text(row.get("version_id"), name + ".version_id", 400), _window_start(row, name)))
    if not keys or len(set(keys)) != len(keys):
        raise ValueError(name + " requires nonempty unique record versions")
    return set(keys)


def _scope_matches(action_scope, scope, direction):
    return (isinstance(action_scope, dict) and action_scope.get("direction") == direction
            and all(action_scope.get(key) == scope[key] for key in ("as_of", "mode", "is_demo") if key in scope))


def semantic_key(kind, arguments):
    if kind not in KINDS or not isinstance(arguments, dict):
        raise ValueError("invalid KDD action kind or arguments")
    # Reference identity is only (id, version). Ordering or optional display
    # metadata must never turn an already attempted operation into a new one.
    normalized = _canonical_arguments(arguments)
    if "input_refs" in normalized:
        references = _refs(normalized["input_refs"], "arguments.input_refs")
        normalized["input_refs"] = [{"id": identity, "version": version}
                                    for identity, version in sorted(references)]
    if kind == "document_read":
        # Display annotations cannot turn one exact read into another operation.
        refs = _record_refs(normalized.get("records"), "arguments.records")
        # Omitted start and explicit zero denote the same first window. A later
        # host-derived locator is a different read, never a new source/study.
        normalized["records"] = [{"source_record_id": rid, "version_id": version, "window_start": start}
                                 for rid, version, start in sorted(refs)]
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
    checks, warnings, coverage_issues = {}, [], []
    def coverage_id(cid, source):
        _text(cid, source + ".check_id", 200)
        if cid not in checks:
            issue = {"check_id": cid, "source": source, "status": "unknown"}
            if issue not in coverage_issues:
                coverage_issues.append(issue)
                warnings.append("unknown_coverage_id:" + source + ":" + cid)
            return False
        return True
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
    manifest = {}
    manifest_rows = _objects(request.get("reading_manifest", []), "reading_manifest", 256)
    visible_refs = evidence | {r.get("source_record_id") for r in manifest_rows if isinstance(r.get("source_record_id"), str)}
    for row in manifest_rows:
        identity = (*(_text(row.get(k), "reading_manifest." + k, 400) for k in ("source_id", "source_record_id", "version_id")),
                    _window_start(row, "reading_manifest"))
        if identity in manifest or row.get("status") not in {"read", "unread", "unavailable"}:
            raise ValueError("duplicate reading manifest record or invalid status")
        if row.get("priority", "normal") not in {"critical", "normal"} or row.get("reason", "scope") not in {"correction", "counterevidence", "scope"}:
            raise ValueError("invalid reading manifest priority/reason")
        for cid in _ids(row.get("covers", []), "reading_manifest.covers"):
            coverage_id(cid, "reading_manifest:" + identity[1])
        basis = row.get("basis")
        if basis is not None or row.get("priority") == "critical":
            if not isinstance(basis, dict) or basis.get("kind") not in {"visible_source_relation", "observed_gap", "declared_scope"}:
                raise ValueError("critical reading requires a visible priority basis")
            refs = set(_ids(basis.get("reference_ids"), "reading_manifest.basis.reference_ids", 256))
            permitted = visible_refs if basis["kind"] == "visible_source_relation" else set(checks)
            if basis["kind"] == "declared_scope" and request.get("question_id"):
                permitted = permitted | {request["question_id"]}
            if not refs or not refs <= permitted:
                raise ValueError("reading priority basis must reference visible sources, observed checks or declared question scope")
        manifest[identity] = row
    action_limits = request.get("action_limits", {})
    if not isinstance(action_limits, dict) or set(action_limits) - KINDS:
        raise ValueError("action_limits must map registered action kinds to maximum attempts")
    for kind, limit in action_limits.items():
        _number(limit, "action_limits." + kind, 256)
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
    attempted, history_ids, pending, outcomes, attempt_counts = set(), set(), [], [], {}
    for entry in history:
        aid = _text(entry.get("action_id"), "history.action_id", 200)
        if aid in history_ids or entry.get("status") not in HISTORY_STATUSES:
            raise ValueError("one latest valid receipt per action_id required")
        history_ids.add(aid)
        attempted.add(semantic_key(entry.get("kind"), entry.get("arguments")))
        attempt_counts[entry["kind"]] = attempt_counts.get(entry["kind"], 0) + 1
        new_ids = _ids(entry.get("new_evidence_ids", []), "new_evidence_ids", 256)
        new_refs = _refs(entry.get("new_artifact_refs", []), "new_artifact_refs")
        updates = _objects(entry.get("check_updates", []), "check_updates", 64)
        update_ids = set()
        for update in updates:
            cid = _text(update.get("id"), "check_updates.id", 200)
            if cid in update_ids or update.get("status") not in {"passed", "failed", "unknown"}:
                raise ValueError("duplicate history check update or invalid status")
            update_ids.add(cid)
            coverage_id(cid, "history.check_updates:" + aid)
        information = entry.get("observed_information", {})
        if not isinstance(information, dict):
            raise ValueError("observed_information must be an object")
        for cid, gain in information.items():
            _unit(gain, "observed_information." + str(cid))
            coverage_id(cid, "history.observed_information:" + aid)
        # Coverage decoration is not permission or an executable action. Preserve
        # the actual receipt verbatim, but unknown labels cannot create reward.
        information = {cid: gain for cid, gain in information.items() if cid in checks}
        updates = [update for update in updates if update["id"] in checks]
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
        for cid in covers:
            coverage_id(cid, "candidate.covers:" + aid)
        if kind in action_limits and attempt_counts.get(kind, 0) >= action_limits[kind]:
            reason = reason or "action_limit_reached"
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
        reading = []
        if kind == "document_read":
            if set(args) != {"provider", "source_id", "records", "scope"} or args.get("provider") != "fabric":
                raise ValueError("document_read requires fabric provider/source_id/records/scope only")
            source_id = _text(args.get("source_id"), "arguments.source_id", 400)
            refs = _record_refs(args.get("records"), "arguments.records")
            if costs.get("tool_calls") != 1:
                raise ValueError("document_read denotes one host batch request with tool_calls=1")
            if not _scope_matches(args.get("scope"), scope, direction):
                reason = reason or "scope_mismatch"
            if "applicability" not in raw:
                reason = reason or "missing_applicability"
            for rid, version, start in sorted(refs):
                row = manifest.get((source_id, rid, version, start))
                if row is None:
                    reason = reason or "record_version_not_in_visible_manifest"
                elif row["status"] != "unread":
                    reason = reason or "record_not_unread"
                else:
                    reading.append(row)
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
            if not _scope_matches(action_scope, scope, direction):
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
        expected, estimated_ids = {}, set()
        for estimate in estimates:
            cid = _text(estimate.get("check_id"), "expected_information.check_id", 200)
            if cid in estimated_ids:
                raise ValueError("expected_information must uniquely reference covers")
            estimated_ids.add(cid)
            values = (_unit(estimate.get("probability"), "probability"), _unit(estimate.get("impact"), "impact"))
            if not coverage_id(cid, "candidate.expected_information:" + aid):
                continue
            if cid not in covers:
                raise ValueError("expected_information must uniquely reference covers")
            expected[cid] = values
        if reason:
            excluded.append({"id": aid, "reason": reason})
        else:
            candidates.append({"action": raw, "expected": expected, "scope_match": match, "prior_source": prior_source, "reading": reading})
    return dict(checks=checks, required=required, candidates=candidates, excluded=excluded, warnings=warnings,
                remaining=remaining, pending=pending, outcomes=outcomes, history=history, max_steps=max_steps,
                evidence_count=len(evidence), artifact_count=len(artifacts), coverage_issues=coverage_issues,
                reading_manifest=list(manifest.values()), action_limits=action_limits, attempt_counts=attempt_counts)


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
        gain = sum(state["checks"][cid]["status"] != "passed" for cid in action.get("covers", []) if cid in state["checks"])
        score = gain / (1 + pressure)
    else:
        gain = sum(p["expected_contribution"] for p in parts)
        score = gain / (1 + 0.25 * pressure)
    # A declared, still-unread visible record is a concrete information task even
    # when an untrained model estimates zero coverage gain. This is a routing
    # obligation, not measured gain or a hidden evaluator's must-read list.
    reading_floor = min(1.0, 0.25 * len(candidate["reading"])) * candidate["scope_match"]
    score = max(score, reading_floor / (1 + pressure))
    return {"id": action["id"], "score": round(score, 8), "expected_contribution": round(gain, 6),
            "reading_priority": int(any(r.get("priority") == "critical" for r in candidate["reading"]) and reading_floor > 0),
            "unread_record_count": len(candidate["reading"]), "visible_reading_floor": reading_floor,
            "cost_pressure": round(pressure, 6), "prior_source": candidate["prior_source"], "checks": parts}


def decide(request, *, selector="information_gain_v1", version=METHOD_VERSION):
    if selector not in SELECTORS:
        raise ValueError("unsupported KDD selector")
    state = _validate(request)
    affordable = [c for c in state["candidates"] if all(c["action"]["estimated_cost"][d] <= state["remaining"][d] for d in state["remaining"])]
    ranked = sorted([_score(c, state, selector) for c in affordable], key=lambda r: (-r["reading_priority"], -r["score"], r["id"]))
    unmet = [cid for cid in state["required"] if state["checks"][cid]["status"] != "passed"]
    unread = [row for row in state["reading_manifest"] if row["status"] != "read"]
    selected = None
    if state["pending"]:
        decision, reason = "insufficient", "pending_execution_or_reconciliation"
    elif state["required"] and not unmet and not unread:
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
            "policy": {"version": version, "method_version": METHOD_VERSION, "selector": selector, "stop": STOP_VERSION,
                       "training_status": "heuristic_untrained", "encoder": "kdd_observed_information_v1",
                       "scope": "host_observation_coverage_only_not_scientific_or_business_approval"},
            "state_summary": {"direction": request["direction"], "checks": list(state["checks"].values()),
                "evidence_count": state["evidence_count"], "artifact_count": state["artifact_count"],
                "step": len(state["history"]), "legal_action_count": len(state["candidates"]),
                "affordable_action_count": len(affordable), "remaining": copy.deepcopy(state["remaining"]),
                "pending_action_ids": state["pending"], "excluded_candidates": state["excluded"],
                "warnings": state["warnings"], "coverage_issues": state["coverage_issues"],
                 "unmet_reading": copy.deepcopy(unread), "unread_window_count": len(unread),
                 "unread_version_count": len({(r["source_id"], r["source_record_id"], r["version_id"]) for r in unread}),
                 "action_limits": copy.deepcopy(state["action_limits"]),
                 "action_attempt_counts": state["attempt_counts"], "unmet_gaps": gaps, "candidate_scores": ranked}}
