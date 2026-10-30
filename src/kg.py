"""执行层：玩具医学知识图谱 + 四个动作的真实检索实现。

对应循环里的"执行()"。这一层属于**固定工具**，不是本课题的研究内容；
将来整体替换为真实 KG / 文献检索 / A / E 接口时，上层代码不需要改动。

职责边界（重要）：
  * 本模块只负责 **合法性**（动作能不能做、预计成本多少、执行结果是什么）；
  * "偏好哪个动作"属于策略，在 policy.py 里，不在本模块。

动作语义与成本口径（第一版占位，需在正式环境中冻结）：
  entity_lookup(entity)             返回该实体的关系列表；成本 = 返回的关系种类数
  relation_expand(entity, relation) 沿该关系取 tail 三元组；成本 = 1
  bounded_path(start, target, hops) 在图上找最短路径；成本 = 实际遍历跳数（未找到按上限计）
  passage_retrieve(keywords)        返回命中文段；成本 = 0 关系访问，记 token
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from .constants import MAX_PATH_HOPS
from .schema import Action, ActionCost, Evidence, Passage, Triple
from .state_construction import PolicyState, evidence_triples


class ToyKG:
    def __init__(self, entities: dict, triples: list[Triple], passages: list[Passage]) -> None:
        self.entity_types = {name: spec.get("type", "") for name, spec in entities.items()}
        self.aliases: dict[str, str] = {}
        for name, spec in entities.items():
            self.aliases[name] = name
            for alias in spec.get("aliases", []) or []:
                self.aliases[alias] = name
        self.triples = triples
        self.passages = passages
        self._by_head: dict[str, list[Triple]] = {}
        for t in triples:
            self._by_head.setdefault(t.head, []).append(t)

    # ---------- 加载 ----------
    @classmethod
    def from_json(cls, path: str | Path) -> "ToyKG":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        entities = payload["entities"]

        def _etype(name: str) -> str:
            return str(entities.get(name, {}).get("type", ""))

        triples = [
            Triple(
                t["head"], t["relation"], t["tail"],
                source=t.get("source", ""),
                confidence=float(t.get("confidence", 1.0)),
                source_id=str(t.get("source_id", "")),
                head_type=str(t.get("head_type", "") or _etype(t["head"])),
                tail_type=str(t.get("tail_type", "") or _etype(t["tail"])),
                metadata=dict(t.get("metadata", {}) or {}),
            )
            for t in payload["triples"]
        ]
        passages = []
        for p in payload.get("passages", []):
            supports = tuple(Triple(*s) for s in p.get("supports", []))
            passages.append(Passage(p["pid"], p["text"], supports, tuple(p.get("keywords", []))))
        return cls(payload["entities"], triples, passages)

    # ---------- 基本查询 ----------
    def canonical(self, name: str) -> str | None:
        return self.aliases.get(name)

    def entity_names(self) -> list[str]:
        return list(self.entity_types.keys())

    def relation_names(self) -> list[str]:
        out: list[str] = []
        for t in self.triples:
            if t.relation not in out:
                out.append(t.relation)
        return out

    def relations_of(self, entity: str) -> list[str]:
        name = self.canonical(entity) or entity
        out: list[str] = []
        for t in self._by_head.get(name, []):
            if t.relation not in out:
                out.append(t.relation)
        return out

    def expand(self, entity: str, relation: str) -> list[Triple]:
        name = self.canonical(entity) or entity
        return [t for t in self._by_head.get(name, []) if t.relation == relation]

    def shortest_path(self, start: str, target: str, max_hops: int = MAX_PATH_HOPS) -> list[Triple] | None:
        s = self.canonical(start) or start
        t = self.canonical(target) or target
        if s == t:
            return []
        queue: deque[tuple[str, list[Triple]]] = deque([(s, [])])
        seen = {s}
        while queue:
            node, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for tri in self._by_head.get(node, []):
                if tri.tail in seen:
                    continue
                newpath = path + [tri]
                if tri.tail == t:
                    return newpath
                seen.add(tri.tail)
                queue.append((tri.tail, newpath))
        return None

    def passages_for(self, keywords: list[str], top_k: int = 2) -> list[Passage]:
        scored = []
        for p in self.passages:
            hits = len(set(keywords) & set(p.keywords))
            if hits:
                scored.append((hits, p))
        scored.sort(key=lambda x: (-x[0], x[1].pid))
        return [p for _, p in scored[:top_k]]

    # ---------- 动作成本（预计） ----------
    def estimated_cost(self, kind: str, args: tuple[str, ...]) -> ActionCost:
        if kind == "entity_lookup":
            return ActionCost(relation_access=len(self.relations_of(args[0])), tool_calls=1)
        if kind == "relation_expand":
            return ActionCost(relation_access=1, tool_calls=1)
        if kind == "bounded_path":
            return ActionCost(relation_access=MAX_PATH_HOPS, tool_calls=1)
        if kind == "passage_retrieve":
            return ActionCost(relation_access=0, tool_calls=1)
        raise ValueError(f"unknown action kind: {kind}")

    # ---------- 合法动作空间 ----------
    def legal_actions(self, state: PolicyState) -> list[Action]:
        """返回当前**合法且尚未尝试过**的动作（含预计成本）。

        只做合法性判断（含重复动作过滤），不做任何偏好排序——
        排序与选择是策略的职责。
        """
        out: list[Action] = []
        seen_keys = {t.key() for t in evidence_triples(state.evidence)}

        for s in state.seeds:
            for t in state.targets:
                if (s, t) in state.tried_paths:
                    continue
                cost = self.estimated_cost("bounded_path", (s, t))
                out.append(Action(f"bounded_path:{s}->{t}", "bounded_path", (s, t), cost))

        for e in state.known_entities:
            if e in state.looked_up:
                continue
            cost = self.estimated_cost("entity_lookup", (e,))
            out.append(Action(f"entity_lookup:{e}", "entity_lookup", (e,), cost))

        for e, rels in state.relations_seen.items():
            for r in rels:
                if (e, r) in state.expanded:
                    continue
                if any(k[0] == e and k[1] == r for k in seen_keys):
                    continue  # 已有该 (head, relation) 的证据，属于重复动作
                cost = self.estimated_cost("relation_expand", (e, r))
                out.append(Action(f"relation_expand:{e}|{r}", "relation_expand", (e, r), cost))

        # 文段检索的语义是“为已建立的证据链补文本支撑”，
        # 因此在没有任何结构化证据时视为非法动作（避免过早检索文段）。
        has_triple = any(k[1] for k in seen_keys)
        if not state.tried_passages and has_triple:
            kws = tuple(dict.fromkeys(state.seeds + state.targets + state.relation_keywords)) or state.seeds
            cost = self.estimated_cost("passage_retrieve", kws)
            out.append(Action("passage_retrieve:" + "|".join(kws), "passage_retrieve", kws, cost))

        return out

    # ---------- 执行 ----------
    def execute(self, action: Action, step: int) -> tuple[list[Evidence], ActionCost, str]:
        aid = action.aid
        if action.kind == "entity_lookup":
            entity = action.args[0]
            rels = self.relations_of(entity)
            cost = ActionCost(relation_access=len(rels), tool_calls=1)
            ev = [Evidence(f"{step}:{aid}:0", "relation_list", aid, step, entity=entity, relations=tuple(rels))]
            return ev, cost, f"返回 {len(rels)} 个关系"

        if action.kind == "relation_expand":
            entity, relation = action.args
            triples = self.expand(entity, relation)
            cost = ActionCost(relation_access=1, tool_calls=1)
            ev = [Evidence(f"{step}:{aid}:{i}", "triple", aid, step, triple=t) for i, t in enumerate(triples)]
            return ev, cost, f"返回 {len(triples)} 条证据"

        if action.kind == "bounded_path":
            start, target = action.args[0], action.args[1]
            path = self.shortest_path(start, target, MAX_PATH_HOPS)
            if path is None:
                return [], ActionCost(relation_access=MAX_PATH_HOPS, tool_calls=1), f"{MAX_PATH_HOPS} 跳内未找到路径"
            cost = ActionCost(relation_access=max(len(path), 1), tool_calls=1)
            ev = [Evidence(f"{step}:{aid}:{i}", "triple", aid, step, triple=t) for i, t in enumerate(path)]
            return ev, cost, f"找到 {len(path)} 跳路径"

        if action.kind == "passage_retrieve":
            keywords = list(action.args)
            hits = self.passages_for(keywords)
            tokens = sum(len(p.text) // 2 for p in hits)
            cost = ActionCost(relation_access=0, tool_calls=1, tokens=tokens)
            ev = [Evidence(f"{step}:{aid}:{i}", "passage", aid, step, passage=p) for i, p in enumerate(hits)]
            return ev, cost, f"返回 {len(hits)} 个文段（约 {tokens} token）"

        raise ValueError(f"unknown action kind: {action.kind}")