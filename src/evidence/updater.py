"""⑦ 更新证据策略：把新证据并入状态，并维护查询历史（用于状态编码与动作掩码）。

  StandardUpdater  标准实现：证据入列 + 去重集合 + 查询历史记录

历史上的每一步都记 (动作类型, 新增证据数, 成本)，
供 ② 状态编码的「历史块」使用（对应 RareAgent 的 stagnation 信号）。
"""
from __future__ import annotations

from typing import Sequence

from ..schema import Action, ActionCost, Evidence, HistoryItem
from ..common.stage_contracts import EvidenceStore


class StandardUpdater:
    name = "standard"

    def update(
        self,
        store: EvidenceStore,
        action: Action,
        new_evidence: Sequence[Evidence],
        cost: ActionCost | None = None,
    ) -> None:
        # 先去重计数：只有"新"三元组才算产出
        before = {(e.triple.head, e.triple.relation, e.triple.tail)
                  for e in store.evidence if e.triple is not None}
        store.evidence.extend(new_evidence)
        after = {(e.triple.head, e.triple.relation, e.triple.tail)
                 for e in store.evidence if e.triple is not None}
        yielded = len(after) - len(before)

        # 查询历史（供动作掩码与状态编码使用）
        if action.kind == "entity_lookup":
            store.looked_up.add(action.args[0])
            for ev in new_evidence:
                if ev.kind == "relation_list":
                    store.relations_seen[ev.entity] = ev.relations
        elif action.kind == "relation_expand":
            store.expanded.add((action.args[0], action.args[1]))
        elif action.kind == "bounded_path":
            store.tried_paths.add((action.args[0], action.args[1]))
        elif action.kind == "passage_retrieve":
            store.tried_passages = True

        store.history.append(HistoryItem(
            kind=action.kind,
            yielded=yielded,
            cost_ra=(cost.relation_access if cost is not None else action.cost.relation_access),
        ))


UPDATERS = {StandardUpdater.name: StandardUpdater}
