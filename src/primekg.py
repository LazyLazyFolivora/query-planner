"""PrimeKG 的按需查询适配层。

第一版使用 SQLite 索引：
    kg.csv -> primekg.sqlite

不需要把 936MB CSV 一次性读进内存。
"""

from __future__ import annotations

import csv
import sqlite3
from collections import deque
from pathlib import Path
from typing import Iterable

from .constants import MAX_PATH_HOPS
from .schema import Action, ActionCost, Evidence, Triple
from .state_construction import PolicyState, evidence_triples


class PrimeKG:
    """PrimeKG 的 SQLite-backed 最小接口。"""

    def __init__(
        self,
        csv_path: str | Path,
        db_path: str | Path | None = None,
        *,
        build_if_missing: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.db_path = Path(db_path) if db_path else self.csv_path.with_suffix(".sqlite")
        if build_if_missing and not self.db_path.exists():
            self.build_index()
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"PrimeKG SQLite 索引不存在：{self.db_path}；请先运行 build_index()"
            )
        self._aliases: dict[str, str] | None = None
        self._entity_types: dict[str, str] | None = None
        self._relation_names: tuple[str, ...] | None = None

    # ---------------------------------------------------------------- 建库
    def build_index(self, batch_size: int = 100_000) -> None:
        """从 kg.csv 构建 SQLite 索引。一次执行，之后直接查询。"""
        if self.db_path.exists():
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("PRAGMA journal_mode=OFF")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute(
                """
                CREATE TABLE edges (
                    head TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    tail TEXT NOT NULL,
                    head_id TEXT,
                    tail_id TEXT,
                    head_type TEXT,
                    tail_type TEXT,
                    source TEXT
                )
                """
            )
            batch: list[tuple] = []
            with self.csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    batch.append(
                        (
                            row["x_name"],
                            row["relation"],
                            row["y_name"],
                            row.get("x_id", ""),
                            row.get("y_id", ""),
                            row.get("x_type", ""),
                            row.get("y_type", ""),
                            row.get("x_source", ""),
                        )
                    )
                    if len(batch) >= batch_size:
                        conn.executemany(
                            "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch
                        )
                        batch.clear()
                if batch:
                    conn.executemany(
                        "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?)", batch
                    )
            conn.execute("CREATE INDEX idx_edges_head ON edges(head)")
            conn.execute("CREATE INDEX idx_edges_tail ON edges(tail)")
            conn.execute("CREATE INDEX idx_edges_head_rel ON edges(head, relation)")
            conn.execute("CREATE INDEX idx_edges_tail_rel ON edges(tail, relation)")
            conn.commit()
        finally:
            conn.close()

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ---------------------------------------------------------------- 只读视图
    @property
    def aliases(self) -> dict[str, str]:
        if self._aliases is None:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT head FROM edges UNION SELECT DISTINCT tail FROM edges"
                ).fetchall()
            self._aliases = {str(row[0]): str(row[0]) for row in rows if row[0]}
        return self._aliases

    @property
    def entity_types(self) -> dict[str, str]:
        if self._entity_types is None:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT head, head_type FROM edges "
                    "UNION SELECT DISTINCT tail, tail_type FROM edges"
                ).fetchall()
            self._entity_types = {str(name): str(tp) for name, tp in rows if name}
        return self._entity_types

    def entity_names(self) -> list[str]:
        return list(self.aliases.keys())

    def relation_names(self) -> list[str]:
        if self._relation_names is None:
            with self._conn() as conn:
                rows = conn.execute("SELECT DISTINCT relation FROM edges ORDER BY relation").fetchall()
            self._relation_names = tuple(str(row[0]) for row in rows if row[0])
        return list(self._relation_names)

    def canonical(self, name: str) -> str | None:
        return self.aliases.get(name)

    # ---------------------------------------------------------------- 查询
    def relations_of(self, entity: str) -> list[str]:
        with self._conn() as conn:
            outgoing = [row[0] for row in conn.execute(
                "SELECT DISTINCT relation FROM edges WHERE head = ? ORDER BY relation", (entity,)
            ).fetchall()]
            incoming = [row[0] for row in conn.execute(
                "SELECT DISTINCT relation FROM edges WHERE tail = ? ORDER BY relation", (entity,)
            ).fetchall()]
        out: list[str] = []
        for rel in outgoing:
            if rel not in out:
                out.append(rel)
        for rel in incoming:
            marked = "~" + rel
            if marked not in out:
                out.append(marked)
        return out

    @staticmethod
    def _triple(
        head: str,
        relation: str,
        tail: str,
        head_id: str,
        tail_id: str,
        head_type: str,
        tail_type: str,
        source: str,
    ) -> Triple:
        return Triple(
            head=head,
            relation=relation,
            tail=tail,
            source=source,
            source_id=f"{head_id}->{tail_id}",
            head_type=head_type,
            tail_type=tail_type,
        )

    def expand(self, entity: str, relation: str) -> list[Triple]:
        reverse = relation.startswith("~")
        raw = relation[1:] if reverse else relation
        with self._conn() as conn:
            if reverse:
                rows = conn.execute(
                    "SELECT head, tail, head_id, tail_id, head_type, tail_type, source "
                    "FROM edges WHERE tail = ? AND relation = ?",
                    (entity, raw),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT head, tail, head_id, tail_id, head_type, tail_type, source "
                    "FROM edges WHERE head = ? AND relation = ?",
                    (entity, raw),
                ).fetchall()
        out: list[Triple] = []
        for head, tail, head_id, tail_id, head_type, tail_type, source in rows:
            if reverse:
                out.append(self._triple(
                    entity, "~" + raw, str(head), str(tail_id), str(head_id),
                    str(tail_type), str(head_type), str(source),
                ))
            else:
                out.append(self._triple(
                    str(head), raw, str(tail), str(head_id), str(tail_id),
                    str(head_type), str(tail_type), str(source),
                ))
        return out

    def shortest_path(self, start: str, target: str, max_hops: int = MAX_PATH_HOPS) -> list[Triple] | None:
        """先实现有向 1～3 跳最短路径；无结果返回 None。"""
        with self._conn() as conn:
            if max_hops >= 1:
                rows = conn.execute(
                    "SELECT head, relation, tail, head_id, tail_id, head_type, tail_type, source "
                    "FROM edges WHERE head = ? AND tail = ? LIMIT 1",
                    (start, target),
                ).fetchall()
                if rows:
                    r = rows[0]
                    return [self._triple(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7])]
            if max_hops >= 2:
                rows = conn.execute(
                    """
                    SELECT e1.head, e1.relation, e1.tail, e2.relation, e2.tail,
                           e1.head_id, e1.tail_id, e2.tail_id,
                           e1.head_type, e1.tail_type, e2.tail_type,
                           e1.source, e2.source
                    FROM edges e1 JOIN edges e2 ON e2.head = e1.tail
                    WHERE e1.head = ? AND e2.tail = ? LIMIT 1
                    """,
                    (start, target),
                ).fetchall()
                if rows:
                    r = rows[0]
                    return [
                        self._triple(r[0], r[1], r[2], r[5], r[6], r[8], r[9], r[11]),
                        self._triple(r[2], r[3], r[4], r[6], r[7], r[9], r[10], r[12]),
                    ]
            if max_hops >= 3:
                rows = conn.execute(
                    """
                    SELECT e1.head, e1.relation, e1.tail, e2.relation, e2.tail, e3.relation, e3.tail,
                           e1.head_id, e1.tail_id, e2.tail_id, e3.tail_id,
                           e1.head_type, e1.tail_type, e2.tail_type, e3.tail_type,
                           e1.source, e2.source, e3.source
                    FROM edges e1
                    JOIN edges e2 ON e2.head = e1.tail
                    JOIN edges e3 ON e3.head = e2.tail
                    WHERE e1.head = ? AND e3.tail = ? LIMIT 1
                    """,
                    (start, target),
                ).fetchall()
                if rows:
                    r = rows[0]
                    return [
                        self._triple(r[0], r[1], r[2], r[7], r[8], r[11], r[12], r[15]),
                        self._triple(r[2], r[3], r[4], r[8], r[9], r[12], r[13], r[16]),
                        self._triple(r[4], r[5], r[6], r[9], r[10], r[13], r[14], r[17]),
                    ]
        return None

    # ---------------------------------------------------------------- 动作层
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

    def legal_actions(self, state: PolicyState) -> list[Action]:
        out: list[Action] = []
        seen_keys = {t.key() for t in evidence_triples(state.evidence)}

        for seed in state.seeds:
            for target in state.targets:
                if (seed, target) in state.tried_paths:
                    continue
                cost = self.estimated_cost("bounded_path", (seed, target))
                out.append(Action(f"bounded_path:{seed}->{target}", "bounded_path", (seed, target), cost))

        for entity in state.known_entities:
            if entity in state.looked_up:
                continue
            cost = self.estimated_cost("entity_lookup", (entity,))
            out.append(Action(f"entity_lookup:{entity}", "entity_lookup", (entity,), cost))

        for entity, relations in state.relations_seen.items():
            for relation in relations:
                if (entity, relation) in state.expanded:
                    continue
                if any(k[0] == entity and k[1] == relation for k in seen_keys):
                    continue
                cost = self.estimated_cost("relation_expand", (entity, relation))
                out.append(Action(
                    f"relation_expand:{entity}|{relation}",
                    "relation_expand",
                    (entity, relation),
                    cost,
                ))
        return out

    def execute(self, action: Action, step: int) -> tuple[list[Evidence], ActionCost, str]:
        if action.kind == "entity_lookup":
            entity = action.args[0]
            relations = self.relations_of(entity)
            cost = ActionCost(relation_access=len(relations), tool_calls=1)
            return [Evidence(
                f"{step}:{action.aid}:0", "relation_list", action.aid, step,
                entity=entity, relations=tuple(relations),
            )], cost, f"返回 {len(relations)} 个关系"
        if action.kind == "relation_expand":
            entity, relation = action.args
            triples = self.expand(entity, relation)
            cost = ActionCost(relation_access=1, tool_calls=1)
            return [
                Evidence(f"{step}:{action.aid}:{i}", "triple", action.aid, step, triple=t)
                for i, t in enumerate(triples)
            ], cost, f"返回 {len(triples)} 条证据"
        if action.kind == "bounded_path":
            start, target = action.args
            path = self.shortest_path(start, target, MAX_PATH_HOPS)
            if path is None:
                return [], ActionCost(relation_access=MAX_PATH_HOPS, tool_calls=1), f"{MAX_PATH_HOPS} 跳内未找到路径"
            cost = ActionCost(relation_access=max(len(path), 1), tool_calls=1)
            return [
                Evidence(f"{step}:{action.aid}:{i}", "triple", action.aid, step, triple=t)
                for i, t in enumerate(path)
            ], cost, f"找到 {len(path)} 跳路径"
        if action.kind == "passage_retrieve":
            return [], ActionCost(relation_access=0, tool_calls=1), "PrimeKG 暂无文段检索"
        raise ValueError(f"unknown action kind: {action.kind}")
