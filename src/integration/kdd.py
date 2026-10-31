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
METHOD_VERSION = "kdd-information-gain-1.4"
MAX_PAGES_PER_CHAIN = 5
# Pending receipts precede every stop; visible unread obligations prevent a
# coverage-only completion. A stop never certifies scientific/business validity.
STOP_VERSION = "observed_checks_reading_pages_objects_v4"
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


def _catalog_identity(args):
    """A host-unlocked directory action is one bounded first page, never a query rewrite."""
    required = {"provider", "source_id", "catalog_discovery", "terms", "limit", "mode", "as_of", "is_demo", "scope"}
    if (not required.issubset(args) or set(args) - required - {"phase", "index", "plan"}
            or args.get("provider") != "fabric" or args.get("catalog_discovery") is not True
            or args.get("terms") != [] or "query" in args):
        raise ValueError("catalog_discovery requires an explicit Fabric first-page action with terms=[] and no query")
    source = _text(args["source_id"], "catalog_discovery.source_id", 400)
    for key in ("mode", "as_of"):
        _text(args[key], "catalog_discovery." + key, 100)
    if type(args["is_demo"]) is not bool:
        raise ValueError("catalog_discovery.is_demo must be boolean")
    _number(args["limit"], "catalog_discovery.limit", 20)
    if args["limit"] < 1:
        raise ValueError("catalog_discovery.limit must be in [1, 20]")
    scope = args["scope"]
    if (not isinstance(scope, dict) or set(scope) != {"direction", "mode", "as_of", "is_demo"}
            or not isinstance(scope.get("direction"), str) or scope["direction"] not in DIRECTIONS
            or type(scope["is_demo"]) is not bool
            or any(scope[key] != args[key] for key in ("mode", "as_of", "is_demo"))):
        raise ValueError("catalog_discovery requires matching explicit flat and nested scope")
    if "phase" in args and args["phase"] != "initial":
        raise ValueError("catalog_discovery is only an initial retrieval action")
    if "index" in args:
        _number(args["index"], "catalog_discovery.index", 256)
    if "plan" in args:
        plan = args["plan"]
        if (not isinstance(plan, dict) or set(plan) - {"source_id", "terms", "reason"}
                or plan.get("source_id") != source or plan.get("terms") != []):
            raise ValueError("catalog_discovery plan must preserve the same source and empty terms")
    # One attempt per source/scope. A new ID, page length or host annotation
    # cannot turn a received, failed or unknown directory operation into a retry.
    return {"provider": "fabric", "source_id": source, "catalog_discovery": True, "scope": scope}



_PAGE_ARGUMENTS = {"cursor", "previous_action_id", "page_number", "max_seq"}


def _search_binding(args):
    """Public query/scope binding; cursor authentication and permissions stay at the host."""
    required = {"provider", "source_id", "terms", "limit", "mode", "as_of", "is_demo", "scope"}
    if not required.issubset(args) or args.get("provider") != "fabric" or "query" in args:
        raise ValueError("paginated search requires explicit Fabric terms, limit and complete scope; no query")
    _text(args["source_id"], "search.source_id", 400)
    for key in ("mode", "as_of"):
        _text(args[key], "search." + key, 100)
    if type(args["is_demo"]) is not bool or type(args["limit"]) is not int or not 1 <= args["limit"] <= 20:
        raise ValueError("paginated search requires boolean is_demo and integer limit in [1, 20]")
    scope = args["scope"]
    if (not isinstance(scope, dict) or set(scope) != {"direction", "mode", "as_of", "is_demo"}
            or scope.get("direction") not in DIRECTIONS or type(scope["is_demo"]) is not bool
            or any(scope[key] != args[key] for key in ("mode", "as_of", "is_demo"))):
        raise ValueError("paginated search requires matching explicit flat and nested scope")
    terms = _ids(args["terms"], "search.terms", 3)
    catalog = args.get("catalog_discovery", False)
    if type(catalog) is not bool or (bool(terms) if catalog else not terms):
        raise ValueError("empty terms require catalog_discovery=true; normal Fabric search requires 1-3 terms")
    if any(not 2 <= len(term.strip()) <= 120 for term in terms):
        raise ValueError("Fabric terms require 2-120 characters")
    # Match the actual provider binding: terms are strip/sorted/set, not casefolded.
    # Display ID, reason and host index never identify a page.
    return {"provider": "fabric", "source_id": args["source_id"], "terms": sorted({term.strip() for term in terms}),
            "limit": args["limit"], "scope": scope, "catalog_discovery": catalog}


def _continuation_identity(args):
    allowed = {"provider", "source_id", "terms", "limit", "mode", "as_of", "is_demo", "scope", "catalog_discovery", "phase", "index", "plan"} | _PAGE_ARGUMENTS
    if not _PAGE_ARGUMENTS.issubset(args) or set(args) - allowed:
        raise ValueError("continuation requires cursor/previous_action_id/page_number/max_seq only with the declared search fields")
    binding = _search_binding(args)
    _text(args["cursor"], "cursor", 4096)
    _text(args["previous_action_id"], "previous_action_id", 200)
    if type(args["page_number"]) is not int or not 2 <= args["page_number"] <= MAX_PAGES_PER_CHAIN:
        raise ValueError("continuation page_number must be an integer in [2, 5]")
    if type(args["max_seq"]) is not int or args["max_seq"] < 0:
        raise ValueError("continuation max_seq must be a nonnegative integer")
    if "phase" in args and args["phase"] not in {"initial", "followup"}:
        raise ValueError("invalid continuation host phase")
    if "index" in args:
        _number(args["index"], "continuation.index", 256)
    if "plan" in args:
        plan = args["plan"]
        if (not isinstance(plan, dict) or plan.get("source_id") != args["source_id"]
                or plan.get("terms") != args["terms"]):
            raise ValueError("continuation host plan must preserve source and terms")
    return {**binding, "cursor": args["cursor"]}


def _pagination(entry):
    if "pagination" not in entry:
        return None
    page = entry["pagination"]
    if (entry.get("kind") != "document_search" or entry.get("status") != "completed"
            or not isinstance(page, dict) or set(page) != {"has_more", "next_cursor", "max_seq", "page_number"}):
        raise ValueError("pagination is an exact completed Fabric search receipt")
    _search_binding(entry["arguments"])
    if (type(page["has_more"]) is not bool or type(page["max_seq"]) is not int or page["max_seq"] < 0
            or type(page["page_number"]) is not int or not 1 <= page["page_number"] <= MAX_PAGES_PER_CHAIN
            or page["page_number"] != entry["arguments"].get("page_number", 1)):
        raise ValueError("invalid search pagination receipt")
    if page["has_more"]:
        _text(page["next_cursor"], "pagination.next_cursor", 4096)
    elif page["next_cursor"] is not None:
        raise ValueError("exhausted search page must have next_cursor=null")
    if "max_seq" in entry["arguments"] and entry["arguments"]["max_seq"] != page["max_seq"]:
        raise ValueError("pagination changed the frozen max_seq")
    return page


def _continuation_reason(args, by_id):
    previous = by_id.get(args["previous_action_id"])
    if previous is None or previous.get("status") != "completed" or previous.get("kind") != "document_search":
        return "missing_completed_previous_page"
    page = previous.get("pagination")
    if not page or not page["has_more"]:
        return "previous_page_has_no_continuation"
    if (page["next_cursor"] != args["cursor"] or page["max_seq"] != args["max_seq"]
            or page["page_number"] + 1 != args["page_number"]
            or _search_binding(previous["arguments"]) != _search_binding(args)):
        return "continuation_receipt_binding_mismatch"
    return None


def _incomplete_search_chains(history):
    # Count unresolved query chains only, never infer a number of unseen pages or records.
    continued = {entry["arguments"]["previous_action_id"] for entry in history
                 if entry.get("status") == "completed" and entry.get("pagination")
                 and entry.get("kind") == "document_search" and "cursor" in entry["arguments"]}
    return [{"previous_action_id": entry["action_id"], "source_id": entry["arguments"]["source_id"],
             "page_number": entry["pagination"]["page_number"], "max_seq": entry["pagination"]["max_seq"],
             "next_cursor_sha256": hashlib.sha256(entry["pagination"]["next_cursor"].encode()).hexdigest(),
             "has_more": True, "qualification": "uncompleted_search_chain_not_unseen_page_or_record_count"}
            for entry in history if entry.get("pagination", {}).get("has_more")
            and entry["action_id"] not in continued]


def semantic_key(kind, arguments):
    if kind not in KINDS or not isinstance(arguments, dict):
        raise ValueError("invalid KDD action kind or arguments")
    # Reference identity is only (id, version). Ordering or optional display
    # metadata must never turn an already attempted operation into a new one.
    normalized = _canonical_arguments(arguments)
    if kind == "document_search" and _PAGE_ARGUMENTS.intersection(arguments):
        normalized = _continuation_identity(arguments)
    elif kind == "document_search" and "catalog_discovery" in arguments:
        if type(arguments["catalog_discovery"]) is not bool:
            raise ValueError("catalog_discovery must be boolean")
        if arguments["catalog_discovery"]:
            normalized = _catalog_identity(arguments)
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


def _named_objects(request, evidence):
    """Only visible question spans, never inferred evaluator targets or hidden IDs."""
    rows, seen = [], set()
    for raw in _objects(request.get("named_objects", []), "named_objects", 64):
        row = copy.deepcopy(raw)
        if set(row) - {"id", "name", "question_span", "status", "evidence_ids", "candidate_ids", "reason"}:
            raise ValueError("unknown named_object field")
        oid = _text(row.get("id"), "named_object.id", 200)
        name = _text(row.get("name"), "named_object.name", 400)
        span = row.get("question_span")
        if (oid in seen or not isinstance(span, dict) or set(span) != {"start", "end"}
                or any(type(span[k]) is not int for k in ("start", "end"))
                or not 0 <= span["start"] < span["end"] <= len(request["question"])
                or request["question"][span["start"]:span["end"]] != name):
            raise ValueError("named objects require unique IDs and an exact visible question span")
        seen.add(oid)
        if row.get("status") not in {"read", "unread", "unavailable"}:
            raise ValueError("invalid named_object status")
        ids = _ids(row.get("evidence_ids", []), "named_object.evidence_ids", 256)
        _ids(row.get("candidate_ids", []), "named_object.candidate_ids", 128)
        if "reason" in row:
            _text(row["reason"], "named_object.reason", 400)
        if row["status"] == "unavailable" and not row.get("reason"):
            raise ValueError("unavailable named object requires a reason")
        # The host verifies that the name occurs in actually read text. An absent
        # current evidence binding cannot be promoted to a completed obligation.
        if row["status"] == "read" and (not ids or not set(ids) <= evidence):
            row.update(status="unread", reason="read_without_current_evidence")
        rows.append(row)
    return rows


def _search_frontier(history, candidates, excluded, remaining, max_steps):
    """Account for actual unfinished chains without manufacturing provider actions."""
    excluded_by_id = {item["id"]: item["reason"] for item in excluded}
    chains = _incomplete_search_chains(history)
    for chain in chains:
        previous = chain["previous_action_id"]
        attempts = [row for row in history if row["kind"] == "document_search"
                    and row["arguments"].get("previous_action_id") == previous]
        offered = sorted([row for row in candidates if row["kind"] == "document_search"
                          and row["arguments"].get("previous_action_id") == previous], key=lambda row: row["id"])
        reason = None
        if chain["page_number"] >= MAX_PAGES_PER_CHAIN:
            reason = "search_chain_page_limit"
        elif attempts:
            reason = "continuation_already_attempted"
        elif len(history) >= max_steps:
            reason = "max_steps_reached"
        elif remaining.get("tool_calls", 1) < 1:
            reason = "budget_exhausted"
        elif offered:
            legal = [row for row in offered if row["id"] not in excluded_by_id]
            if not legal:
                reason = excluded_by_id[offered[0]["id"]]
            elif not any(all(row["estimated_cost"][d] <= remaining[d] for d in remaining) for row in legal):
                reason = "budget_exhausted"
        else:
            reason = "continuation_candidate_required"
        chain.update(max_pages_per_chain=MAX_PAGES_PER_CHAIN, candidate_ids=[r["id"] for r in offered],
                     unavailable_reason=reason)
    return chains


def _required_gap_catalog(request, action, checks, history):
    """Check visible routing receipts; the host verifies returned original versions."""
    basis = action.get("discovery_basis")
    if (request.get("discovery_version") != "bounded-required-gap-discovery-v1"
            or not isinstance(basis, dict)
            or set(basis) != {"version", "kind", "query_index", "check_ids"}
            or basis.get("version") != request["discovery_version"]
            or basis.get("kind") != "required_gap_repeat_search"):
        return False
    index, ids = basis.get("query_index"), basis.get("check_ids")
    if (type(index) is not int or not 0 <= index < len(history)
            or not isinstance(ids, list) or not ids
            or any(not isinstance(cid, str) for cid in ids) or len(ids) != len(set(ids))
            or not set(ids) <= set(action.get("covers", []))):
        return False
    if any(cid not in checks or not checks[cid].get("required", True)
           or checks[cid]["importance"] <= 0 or checks[cid]["status"] == "passed" for cid in ids):
        return False
    target = action["arguments"]
    chosen = history[index]
    if (chosen["kind"] != "document_search"
            or chosen["arguments"].get("provider") != "fabric"
            or chosen["arguments"].get("source_id") != target["source_id"]
            or not chosen["arguments"].get("terms")
            or chosen["arguments"].get("catalog_discovery")):
        return False
    for row in history[index:]:
        args = row["arguments"]
        source = args.get("source_id") or (args.get("parameters") or {}).get("source_id")
        if source != target["source_id"]:
            continue
        cost = (row.get("actual_cost") or {}).get("tool_calls")
        page = row.get("pagination") or {}
        if row["kind"] == "document_search" and any(args.get(key) != request["scope"][key]
                                                    for key in ("mode", "as_of", "is_demo")):
            return False
        if (row["status"] != "completed" or row.get("new_evidence_ids") != []
                or type(cost) is not int or cost < 1
                or not _scope_matches(args.get("scope"), request["scope"], request["direction"])
                or page.get("has_more") is True or page.get("next_cursor")):
            return False
    return True


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
    named_objects = _named_objects(request, evidence)
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
    history_by_id, search_phase_counts = {}, {}
    new_queries = continuations = 0
    for entry in history:
        aid = _text(entry.get("action_id"), "history.action_id", 200)
        if aid in history_ids or entry.get("status") not in HISTORY_STATUSES:
            raise ValueError("one latest valid receipt per action_id required")
        history_ids.add(aid)
        attempted.add(semantic_key(entry.get("kind"), entry.get("arguments")))
        if entry["kind"] == "document_search":
            if _PAGE_ARGUMENTS.intersection(entry["arguments"]):
                mismatch = _continuation_reason(entry["arguments"], history_by_id)
                if mismatch:
                    raise ValueError("history continuation: " + mismatch)
            page = _pagination(entry)
            if page and (not {"direction", "mode", "as_of", "is_demo"} <= set(scope)
                         or type(scope["is_demo"]) is not bool
                         or not _scope_matches(entry["arguments"]["scope"], scope, direction)):
                raise ValueError("pagination history requires the complete fixed request scope")
            continuations += "cursor" in entry["arguments"]
            new_queries += "cursor" not in entry["arguments"]
            phase = str(entry["arguments"].get("phase", "unspecified"))
            search_phase_counts[phase] = search_phase_counts.get(phase, 0) + 1
        elif "pagination" in entry:
            raise ValueError("pagination only belongs to document_search")
        history_by_id[aid] = entry
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
        # The search allowance counts new first-page queries. Every continuation
        # still consumes its real tool budget and step; its separate bound is five
        # total pages per receipt-bound chain, including the first page.
        counted = new_queries if kind == "document_search" else attempt_counts.get(kind, 0)
        if kind in action_limits and not (kind == "document_search" and "cursor" in args) and counted >= action_limits[kind]:
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
            if _PAGE_ARGUMENTS.intersection(args):
                # The pure entrypoint must enforce the same complete request scope
                # as the schema; absent task fields cannot silently waive binding.
                if (not {"direction", "mode", "as_of", "is_demo"}.issubset(scope)
                        or type(scope["is_demo"]) is not bool):
                    raise ValueError("continuation requires the complete fixed request scope")
                if not _scope_matches(args["scope"], scope, direction):
                    reason = reason or "scope_mismatch"
                if "applicability" not in raw:
                    reason = reason or "missing_applicability"
                reason = reason or _continuation_reason(args, history_by_id)
            elif args.get("catalog_discovery") is True:
                # Shape and first-page identity were validated by semantic_key.
                # The host alone unlocks this after actual empty language alternatives,
                # authorizes the source, performs the request and charges its receipt.
                if (not {"direction", "mode", "as_of", "is_demo"}.issubset(scope)
                        or type(scope["is_demo"]) is not bool):
                    raise ValueError("catalog_discovery requires the complete fixed request scope")
                if not _scope_matches(args["scope"], scope, direction):
                    reason = reason or "scope_mismatch"
                if "applicability" not in raw:
                    reason = reason or "missing_applicability"
                if evidence:
                    if request.get("discovery_version") != "bounded-required-gap-discovery-v1":
                        reason = reason or "catalog_discovery_requires_empty_evidence"
                    elif not _required_gap_catalog(request, raw, checks, history):
                        reason = reason or "catalog_discovery_basis_invalid"
            else:
                query, terms = args.get("query"), args.get("terms")
                if bool(query) == bool(terms):
                    raise ValueError("document_search requires query or terms")
                if query:
                    _text(query, "query", 1000)
                elif not 1 <= len(_ids(terms, "terms", 8)) <= 8:
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
            candidates.append({"action": raw, "expected": expected, "scope_match": match, "prior_source": prior_source, "reading": reading,
                               "continuation": kind == "document_search" and "cursor" in args,
                               "named_object_ids": [row["id"] for row in named_objects
                                   if row["status"] != "read" and aid in row.get("candidate_ids", [])
                                   and kind in RETRIEVAL_KINDS | {"source_fetch"}]})
    return dict(checks=checks, required=required, candidates=candidates, excluded=excluded, warnings=warnings,
                remaining=remaining, pending=pending, outcomes=outcomes, history=history, max_steps=max_steps,
                evidence_count=len(evidence), artifact_count=len(artifacts), coverage_issues=coverage_issues,
                reading_manifest=list(manifest.values()), action_limits=action_limits, attempt_counts=attempt_counts,
                incomplete_search_chains=_search_frontier(history, request.get("candidates", []), excluded, remaining, max_steps),
                search_phase_counts=search_phase_counts, new_query_attempt_count=new_queries,
                continuation_attempt_count=continuations, named_objects=named_objects)


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
    page_floor = 0.25 * candidate["scope_match"] if candidate["continuation"] else 0.0
    named_floor = 0.25 * candidate["scope_match"] if candidate["named_object_ids"] else 0.0
    score = max(score, reading_floor / (1 + pressure), page_floor / (1 + pressure), named_floor / (1 + pressure))
    return {"id": action["id"], "score": round(score, 8), "expected_contribution": round(gain, 6),
            "reading_priority": int(any(r.get("priority") == "critical" for r in candidate["reading"]) and reading_floor > 0),
            "unread_record_count": len(candidate["reading"]), "visible_reading_floor": reading_floor,
            "visible_continuation_floor": page_floor, "continuation_priority": int(candidate["continuation"]),
            "named_object_priority": int(bool(candidate["named_object_ids"])),
            "named_object_ids": candidate["named_object_ids"], "visible_named_object_floor": named_floor,
            "cost_pressure": round(pressure, 6), "prior_source": candidate["prior_source"], "checks": parts}


def decide(request, *, selector="information_gain_v1", version=METHOD_VERSION):
    if selector not in SELECTORS:
        raise ValueError("unsupported KDD selector")
    state = _validate(request)
    affordable = [c for c in state["candidates"] if all(c["action"]["estimated_cost"][d] <= state["remaining"][d] for d in state["remaining"])]
    ranked = sorted([_score(c, state, selector) for c in affordable], key=lambda r: (-r["continuation_priority"], -r["named_object_priority"], -r["reading_priority"], -r["score"], r["id"]))
    unmet = [cid for cid in state["required"] if state["checks"][cid]["status"] != "passed"]
    unread = [row for row in state["reading_manifest"] if row["status"] != "read"]
    unread_objects = [row for row in state["named_objects"] if row["status"] != "read"]
    requests = [chain for chain in state["incomplete_search_chains"]
                if chain["unavailable_reason"] == "continuation_candidate_required"]
    selected = None
    if state["pending"]:
        decision, reason = "insufficient", "pending_execution_or_reconciliation"
    elif state["required"] and not unmet and not unread and not unread_objects and not state["incomplete_search_chains"]:
        decision, reason = "complete", "required_observations_satisfied"
    elif len(state["history"]) >= state["max_steps"]:
        decision, reason = "insufficient", "max_steps_reached"
    elif requests:
        decision, reason = "insufficient", "continuation_candidate_required"
    elif not state["candidates"]:
        reasons = {chain["unavailable_reason"] for chain in state["incomplete_search_chains"]}
        reason = ("named_objects_unread" if unread_objects else
                  next(iter(reasons)) if len(reasons) == 1 else
                  "retrieval_incomplete" if reasons else "no_eligible_action")
        decision = "insufficient"
    elif not affordable:
        decision, reason = "insufficient", "budget_exhausted"
    elif not ranked or ranked[0]["score"] <= 0:
        decision, reason = "insufficient", "no_expected_information_gain"
    else:
        decision, reason = "continue", "selected_by_" + selector
        selected = next(c["action"] for c in affordable if c["action"]["id"] == ranked[0]["id"])
    terminal = decision != "continue" and reason not in {"continuation_candidate_required", "pending_execution_or_reconciliation"}
    execution_state = ("awaiting_host_candidates" if reason == "continuation_candidate_required" else
                       "awaiting_receipt_reconciliation" if state["pending"] else
                       "terminal" if terminal else "action_selected")
    affordable_ids = {c["action"]["id"] for c in affordable}
    excluded = {row["id"]: row["reason"] for row in state["excluded"]}
    unread_objects = [{**row, "available_candidate_ids": sorted(set(row.get("candidate_ids", [])) & affordable_ids),
                       "unavailable_reason": row.get("reason") or
                       (reason if terminal or state["pending"] else
                        None if set(row.get("candidate_ids", [])) & affordable_ids else
                        next((excluded[c] for c in row.get("candidate_ids", []) if c in excluded), "no_legal_candidate"))}
                      for row in unread_objects]
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
                 "execution_state": execution_state, "terminal": terminal,
                 "requested_continuations": requests if reason == "continuation_candidate_required" else [],
                 "max_pages_per_chain": MAX_PAGES_PER_CHAIN,
                 "new_query_attempt_count": state["new_query_attempt_count"],
                 "continuation_attempt_count": state["continuation_attempt_count"],
                 "unmet_named_objects": unread_objects,
                 "action_attempt_counts": state["attempt_counts"], "search_attempt_counts_by_phase": state["search_phase_counts"],
                  "incomplete_search_chains": state["incomplete_search_chains"], "unmet_gaps": gaps, "candidate_scores": ranked}}
