# 0002 Sources and channels are separate; every response carries a decision record

Status: accepted (2026-11)

## Context

The planner used `source` for two different things. WeKnora semantic search was a "source" called `weknora`, so
material the investigation may use (who_don, gdelt_doc) could not be told from a way of finding candidates in it.
And a response held only the choice and the scores: nothing said why a candidate scored 0, which channels were
in play, or how the planner had read the checks when it stopped. Task book 04 (sections 2.2, 2.5, 2.6) asks for
both to be fixed.

## Decision

- A candidate and a history item have a `source` (the material) and an optional `channel` (how it was found:
  `fabric_literal`, `weknora_semantic`, `gdelt_leads`, `graph`). A request may declare `sources` (the allowed set)
  and `channels` (each one's state, with a reason when it is unavailable). A candidate outside the sources, or on
  an unavailable channel, cannot be chosen: a channel never widens the sources. Dedup and the posterior are kept
  per (kind, source, channel).
- The response keeps `decision`, `candidate`, `reason`, `scores` and adds `reasons` (one per candidate),
  `channels` (offered / chosen / not_needed / unavailable), `checks` (as the planner read them) and
  `planner_version`. The planner stays stateless: the radar stores the request and the response, and the same
  request always gives the same response, so the pair rebuilds the decision. How the radar hashes and bounds what
  it stores is the radar's business.
- The state of 05, 06 and 07 reaches the planner as `checks[].status` (open / answered / conflict); the radar folds
  it. "answered" counts only with evidence and means a grounded candidate answer, not a verified one. Nothing here
  lets the planner write a check or decide that a report may be published.
- Scoring, obligations, priorities, budget and stop rules are unchanged. A request without the new fields gets the
  same decision and scores as before.

## Consequences

- The request contract is `extra="forbid"`: a radar that sends the new fields to a planner without them gets 422.
  Deploy the planner first; `/health` reports `version`, `kinds` and `channels` so the radar can check.
- A caller that sends channels must send them on the candidates and the history alike, or an action done through a
  channel is not recognised as done.
- The two graph actions (new kinds) are not part of this change.
