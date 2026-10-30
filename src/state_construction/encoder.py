"""② 状态编码策略：PolicyState -> 定长向量。

  FeatureEncoder  规则实现：**五块 42 维**手工归一化特征
  ModelEncoder    模型实现（默认）：
                    [tanh(Wq·问题嵌入)] ⊕ [证据池化向量] ⊕ [42 维手工特征]

设计原则：
  * 只保留会改变“三态停止”或“动作选择”的信息；
  * 能从其他特征无损推出的量不单独编码；
  * 不使用动态分母、自造指标和未校准置信度；
  * 状态向量里的“维度”不等于“状态变量”——一个 5 类变量用 one-hot 表示就是 5 维。

五块划分：
    ① 问题块 (0-15)    16 维   我在解什么题、要覆盖什么
    ② 证据块 (16-18)    3 维   已拿到多少结构化/文本证据
    ③ 缺口块 (19-33)   15 维   还差什么、是否已无路可走
    ④ 历史块 (34-38)    5 维   最近做过什么、是否卡住
    ⑤ 预算块 (39-41)    3 维   还剩多少预算

说明：
  * “检查项是否适用”和“检查项是否通过”分开编码；
    非机制型问题的因果链/文本支撑是 not_applicable，不会被误当成 failed。
  * 关系组、目标实体类型是集合，所以用 multi-hot；
    推理模式是互斥单值，所以用 one-hot。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..constants import CAUSAL_RELATIONS, ENTITY_TYPE_GROUPS, RELATION_GROUPS, relation_group
from . import (
    CHECK_CAUSAL_CHAIN,
    CHECK_IDS,
    CHECK_NOT_APPLICABLE,
    CHECK_PASSED,
    CHECK_PATH,
    CHECK_REQUIRED_RELATIONS,
    CHECK_TEXT_SUPPORT,
    PolicyState,
    evidence_triples,
    observable_sufficiency_status,
)
from ..infrastructure.embedder import DEFAULT_EMBED_MODEL, Embedder, SentenceEmbedder


# ============================================================ 归一化配置
@dataclass
class Normalizers:
    """计数类特征的归一化上界（占位，需按真实轨迹标定）。"""

    method: str = "log"
    n_triples: float = 20.0
    n_passages: float = 10.0
    max_causal_chain: float = 6.0
    no_progress_streak: float = 5.0
    n_legal_actions: float = 16.0

    def scale(self, x: float, cap: float) -> float:
        """把计数 x 归一到 [0, 1]。"""
        if cap <= 0:
            return 0.0
        value = max(float(x), 0.0)
        if self.method == "log":
            return float(min(math.log1p(value) / math.log1p(cap), 1.0))
        return float(min(value / cap, 1.0))


NORM = Normalizers()

# ---------------- ① 问题块（16 维） ----------------
REASONING_MODES: tuple[str, ...] = (
    "mechanism", "chain", "intersection", "counting", "aggregation",
)

PROBLEM_FEATURES: tuple[str, ...] = (
    *(f"reasoning_mode_{m}" for m in REASONING_MODES),   # 0-4   推理模式 one-hot
    *(f"relgroup_{g}" for g in RELATION_GROUPS),         # 5-10  必需关系组 multi-hot
    *(f"targettype_{g}" for g in ENTITY_TYPE_GROUPS),    # 11-14 目标实体类型 multi-hot
    "has_negative_constraint",                            # 15    否定约束
)

# ---------------- ② 证据块（3 维） ----------------
EVIDENCE_FEATURES: tuple[str, ...] = (
    "n_triples_norm",          # 0 结构化证据条数
    "n_passages_norm",         # 1 文本证据条数（未启用文本检索时恒为 0）
    "max_causal_chain_norm",   # 2 最长因果链；非机制型问题恒为 0
)

# ---------------- ③ 缺口块（15 维） ----------------
# 每项检查用两个 bit：applicable / passed。
# not_applicable -> (0, 0), failed -> (1, 0), passed -> (1, 1)。
GAP_FEATURES: tuple[str, ...] = (
    "check_path_applicable", "check_path_passed",
    "check_required_relations_applicable", "check_required_relations_passed",
    "check_causal_chain_applicable", "check_causal_chain_passed",
    "check_text_support_applicable", "check_text_support_passed",
    *(f"uncovered_relgroup_{g}" for g in RELATION_GROUPS),
    "affordable_action_count_norm",
)

# ---------------- ④ 历史块（5 维） ----------------
HISTORY_FEATURES: tuple[str, ...] = (
    "recent_action_lookup_ratio", "recent_action_expand_ratio",
    "recent_action_path_ratio", "recent_action_passage_ratio",
    "no_progress_streak_norm",
)

# ---------------- ⑤ 预算块（3 维） ----------------
BUDGET_FEATURES: tuple[str, ...] = (
    "rem_relation_access_ratio",
    "rem_tool_calls_ratio",
    "rem_tokens_ratio",
)

FEATURE_NAMES: tuple[str, ...] = (
    PROBLEM_FEATURES + EVIDENCE_FEATURES + GAP_FEATURES + HISTORY_FEATURES + BUDGET_FEATURES
)
N_FEATURES = len(FEATURE_NAMES)

_OFF_PROBLEM = 0
_OFF_EVIDENCE = _OFF_PROBLEM + len(PROBLEM_FEATURES)
_OFF_GAP = _OFF_EVIDENCE + len(EVIDENCE_FEATURES)
_OFF_HISTORY = _OFF_GAP + len(GAP_FEATURES)
_OFF_BUDGET = _OFF_HISTORY + len(HISTORY_FEATURES)

_ACTION_KINDS = ("entity_lookup", "relation_expand", "bounded_path", "passage_retrieve")
RECENT_WINDOW = 3
_CHECK_INDEX = {cid: i for i, cid in enumerate(CHECK_IDS)}


def _reasoning_mode(state: PolicyState) -> str:
    """机制型优先；否则使用解析层给出的推理算子。"""
    if state.qtype == "mechanism":
        return "mechanism"
    return state.operator if state.operator in REASONING_MODES else "chain"


def _causal_edges(triples) -> list:
    return [t for t in triples if t.relation in CAUSAL_RELATIONS]


def _longest_causal_chain(triples) -> int:
    """证据图里最长因果链的边数（沿因果边做 DFS）。"""
    causal = _causal_edges(triples)
    if not causal:
        return 0
    adj: dict[str, list] = {}
    for t in causal:
        adj.setdefault(t.head, []).append(t)
    best = 0

    def walk(node: str, depth: int, visited: set[str]) -> None:
        nonlocal best
        best = max(best, depth)
        for t in adj.get(node, []):
            if t.tail in visited:
                continue
            visited.add(t.tail)
            walk(t.tail, depth + 1, visited)
            visited.discard(t.tail)

    for start in list(adj):
        walk(start, 0, {start})
    return best


class FeatureEncoder:
    """规则实现：五块 42 维手工归一化特征。"""

    name = "feature"
    dim = N_FEATURES

    def __init__(self, norm: Normalizers | None = None) -> None:
        self.norm = norm or NORM

    def encode(self, state: PolicyState) -> np.ndarray:
        N = self.norm
        triples = evidence_triples(state.evidence)
        passages = [e for e in state.evidence if e.kind == "passage"]
        limits = state.budget_limits or {}
        rem = state.remaining
        f = np.zeros(N_FEATURES, dtype=np.float32)

        # ================= ① 问题块 =================
        b = _OFF_PROBLEM
        mode = _reasoning_mode(state)
        if mode in REASONING_MODES:
            f[b + REASONING_MODES.index(mode)] = 1.0

        for g in state.required_relation_groups:
            if g in RELATION_GROUPS:
                f[b + len(REASONING_MODES) + RELATION_GROUPS.index(g)] = 1.0

        b_target = b + len(REASONING_MODES) + len(RELATION_GROUPS)
        for g in state.target_types:
            if g in ENTITY_TYPE_GROUPS:
                f[b_target + ENTITY_TYPE_GROUPS.index(g)] = 1.0

        f[b + len(REASONING_MODES) + len(RELATION_GROUPS) + len(ENTITY_TYPE_GROUPS)] = (
            1.0 if state.excluded_entities else 0.0
        )

        # ================= ② 证据块 =================
        b = _OFF_EVIDENCE
        f[b + 0] = N.scale(len(triples), N.n_triples)
        f[b + 1] = N.scale(len(passages), N.n_passages)
        if state.qtype == "mechanism":
            f[b + 2] = N.scale(_longest_causal_chain(triples), N.max_causal_chain)

        # ================= ③ 缺口块 =================
        b = _OFF_GAP
        statuses = {cid: status for cid, status in observable_sufficiency_status(state)}
        for cid in CHECK_IDS:
            status = statuses.get(cid, CHECK_NOT_APPLICABLE)
            i = _CHECK_INDEX[cid]
            f[b + 2 * i + 0] = 0.0 if status == CHECK_NOT_APPLICABLE else 1.0
            f[b + 2 * i + 1] = 1.0 if status == CHECK_PASSED else 0.0

        b_rel = b + 2 * len(CHECK_IDS)
        required_groups = set(state.required_relation_groups)
        covered_groups = {relation_group(t.relation) for t in triples}
        for i, g in enumerate(RELATION_GROUPS):
            f[b_rel + i] = 1.0 if g in required_groups and g not in covered_groups else 0.0

        feasible_count = state.affordable_action_count
        if feasible_count is None:
            feasible_count = state.legal_action_count or 0
        f[b_rel + len(RELATION_GROUPS)] = N.scale(max(feasible_count, 0), N.n_legal_actions)

        # ================= ④ 历史块 =================
        b = _OFF_HISTORY
        recent = state.recent_actions[-RECENT_WINDOW:]
        denom = max(len(recent), 1)
        for i, kind in enumerate(_ACTION_KINDS):
            f[b + i] = recent.count(kind) / denom
        f[b + 4] = N.scale(state.no_progress_streak, N.no_progress_streak)

        # ================= ⑤ 预算块 =================
        b = _OFF_BUDGET

        def ratio(dim: str) -> float:
            limit = limits.get(dim, 0)
            if limit <= 0:
                return 0.0
            return float(min(max(rem.get(dim, 0), 0) / limit, 1.0))

        f[b + 0] = ratio("relation_access")
        f[b + 1] = ratio("tool_calls")
        f[b + 2] = ratio("tokens")
        return f


class ModelEncoder:
    """模型实现（默认）：问题语义 ⊕ 证据语义 ⊕ 手工特征。"""

    name = "model"

    def __init__(
        self,
        embedder: Embedder | None = None,
        proj_dim: int = 16,
        evidence_dim: int = 16,
        max_evidence_encoded: int = 12,
        seed: int = 42,
    ) -> None:
        self.embedder = embedder or SentenceEmbedder(DEFAULT_EMBED_MODEL)
        self.proj_dim = proj_dim
        self.evidence_dim = evidence_dim
        self.max_evidence_encoded = max_evidence_encoded
        self.feature_encoder = FeatureEncoder()
        self.dim = proj_dim + evidence_dim + self.feature_encoder.dim

        rng = np.random.default_rng(seed)
        scale = 1.0 / max(self.embedder.dim, 1) ** 0.5
        self.W_q = self._ortho(rng.normal(0.0, scale, size=(proj_dim, self.embedder.dim)))
        self.W_e = self._ortho(rng.normal(0.0, scale, size=(evidence_dim, self.embedder.dim)))

    @staticmethod
    def _ortho(w: np.ndarray) -> np.ndarray:
        q, _ = np.linalg.qr(w.T)
        return q.T.astype(np.float32)

    def encode_evidence(self, state: PolicyState) -> np.ndarray:
        """证据集合 -> 定长向量（线性化 + 批量编码 + 平均池化）。"""
        triples = evidence_triples(state.evidence)
        if not triples:
            return np.zeros(self.evidence_dim, dtype=np.float32)
        texts = [t.linearize() for t in triples[: self.max_evidence_encoded]]
        if hasattr(self.embedder, "encode_batch"):
            emb = self.embedder.encode_batch(texts)
        else:
            emb = np.stack([self.embedder.encode(text) for text in texts])
        vecs = np.tanh(emb @ self.W_e.T)
        return np.mean(vecs, axis=0).astype(np.float32)

    def encode(self, state: PolicyState) -> np.ndarray:
        q_proj = np.tanh(self.W_q @ self.embedder.encode(state.question)).astype(np.float32)
        e_proj = self.encode_evidence(state)
        feats = self.feature_encoder.encode(state)
        return np.concatenate([q_proj, e_proj, feats])

    def fit(self, embeddings: np.ndarray, targets: np.ndarray) -> None:
        """预留：最小二乘更新问题投影矩阵（行为克隆 / 监督）。"""
        n_q = self.proj_dim
        sol, *_ = np.linalg.lstsq(embeddings[:, :n_q], targets, rcond=None)
        self.W_q = sol.T.astype(np.float32)


ENCODERS = {FeatureEncoder.name: FeatureEncoder, ModelEncoder.name: ModelEncoder}
