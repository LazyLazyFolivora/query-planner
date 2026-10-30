"""证据视图与图算法：把"已获得的证据"变成可检查的结构。

这一层是纯传统算法（邻接表 + DFS/BFS + 集合运算），不含 LLM，
也不含深度学习模型。它只回答"证据图里有没有某种结构"，
不做任何打分、不做学习。

输入：Evidence 列表（来自 ⑦ EvidenceStore）
输出：邻接表 / 路径存在性 / 解释性路径 / 关系链匹配结果 / 目标节点匹配
"""
from __future__ import annotations

from typing import Sequence

from ..constants import CAUSAL_RELATIONS, entity_type_group
from ..schema import Evidence, Triple

# ---------------------------------------------------------------- 证据拆分


def evidence_triples(evidence: tuple[Evidence, ...]) -> list[Triple]:
    """取出全部三元组证据（kind=="triple"）。"""
    return [e.triple for e in evidence if e.kind == "triple" and e.triple is not None]


def evidence_passages(evidence: tuple[Evidence, ...]):
    """取出全部文段证据（kind=="passage"）。"""
    return [e.passage for e in evidence if e.kind == "passage" and e.passage is not None]


# ---------------------------------------------------------------- 图构建


def build_graph(triples: Sequence[Triple]) -> dict[str, list[Triple]]:
    """正向邻接表：head -> 从它出发的三元组。"""
    graph: dict[str, list[Triple]] = {}
    for t in triples:
        graph.setdefault(t.head, []).append(t)
    return graph


# ---------------------------------------------------------------- 目标匹配


def node_matches_target(
    node: str,
    triples: Sequence[Triple],
    targets: tuple[str, ...],
    target_types: tuple[str, ...],
) -> bool:
    """节点是否算"到达了目标"：命中目标实体名，或其粗粒度类型落在目标类型组里。"""
    if node in targets:
        return True
    if not target_types:
        return False
    wanted = set(target_types)
    for t in triples:
        if t.head == node and entity_type_group(t.head_type) in wanted:
            return True
        if t.tail == node and entity_type_group(t.tail_type) in wanted:
            return True
    return False


# ---------------------------------------------------------------- 路径判定


def any_path_exists(
    triples: Sequence[Triple],
    seeds: tuple[str, ...],
    targets: tuple[str, ...],
    target_types: tuple[str, ...] = (),
) -> bool:
    """证据图中是否存在任一 seed -> target 的有向路径（DFS，带 visited 去重）。"""
    if not targets and not target_types:
        return False
    graph = build_graph(triples)
    for seed in seeds:
        stack = [seed]
        seen = {seed}
        while stack:
            node = stack.pop()
            if node_matches_target(node, triples, targets, target_types):
                return True
            for t in graph.get(node, []):
                if t.tail not in seen:
                    seen.add(t.tail)
                    stack.append(t.tail)
    return False


def find_explanatory_path(
    triples: Sequence[Triple],
    seeds: tuple[str, ...],
    targets: tuple[str, ...],
    target_types: tuple[str, ...] = (),
    min_hops: int = 2,
    max_hops: int = 8,
) -> list[Triple] | None:
    """在已有证据里找一条 >= min_hops 跳、且至少含一条因果关系的 seed->target 路径。"""
    if not targets and not target_types:
        return None
    graph = build_graph(triples)
    for seed in seeds:
        stack: list[tuple[str, list[Triple]]] = [(seed, [])]
        seen_paths: set[tuple[str, int]] = set()
        while stack:
            node, path = stack.pop()
            key = (node, len(path))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            if len(path) >= min_hops and node_matches_target(node, triples, targets, target_types):
                if any(t.relation in CAUSAL_RELATIONS for t in path):
                    return path
            if len(path) >= max_hops:
                continue
            for t in graph.get(node, []):
                if any(t.tail == prev.head for prev in path):  # 避免环
                    continue
                stack.append((t.tail, path + [t]))
    return None


def matches_relation_chain(
    triples: Sequence[Triple],
    seeds: tuple[str, ...],
    chain: Sequence[Sequence[str]],
) -> bool:
    """证据图中是否存在按顺序匹配每跳关系集合的路径（逐层前沿推进）。"""
    if not chain or not seeds:
        return False
    graph = build_graph(triples)
    frontier = set(seeds)
    for allowed in chain:
        next_frontier: set[str] = set()
        for node in frontier:
            for t in graph.get(node, []):
                if t.relation in allowed:
                    next_frontier.add(t.tail)
        if not next_frontier:
            return False
        frontier = next_frontier
    return True
