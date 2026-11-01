"""④ 动作选择策略。

  HeuristicSelector   规则实现：BioWeave 四维打分（缺口语义手工编码）
  LinearModelSelector 线性基线：效用头 − λ(剩余预算) × 成本预测头
  ModelSelector       模型实现（默认）：状态-动作联合 MLP 效用/成本头

训练接口：
    selector.fit(X, y_utility, y_cost)   ← 行为克隆 / 监督
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..constants import CAUSAL_RELATIONS, MECHANISM_RELATIONS, RELATION_PRIORITY, relation_rank
from ..schema import Action, StopReason
from ..state_construction import PolicyState, observable_sufficiency
from ..common.stage_contracts import ActionSelector, Selection
from ..infrastructure.nets import as_tensor, make_trunk, require_torch, zero_init_head

KINDS = ("entity_lookup", "relation_expand", "bounded_path", "passage_retrieve")
N_FEAT = 18

FEATURE_NAMES = (
    "kind_entity_lookup", "kind_relation_expand", "kind_bounded_path", "kind_passage",
    "cost_ra_norm", "cost_calls_norm",
    "relation_prior", "is_causal",
    "head_is_seed", "head_is_target", "bias",
    "closes_reachability",          # bounded_path 且 缺可达性
    "advances_chain",               # relation_expand 且 缺因果链
    "fills_text_support",           # passage 且 缺文本支撑（且链已建）
    "advances_mechanism_chain",     # relation_expand + 机制类关系 + 缺因果链
    "prerequisite_lookup",           # entity_lookup 且 缺链/缺可达 且有未查询实体
    "frontier_recency",             # entity_lookup 的"最近发现优先"
    "premature_passage",            # passage 但链未建立
)


def _failing_checks(state: PolicyState) -> list[str]:
    _, checks = observable_sufficiency(state)
    return [n for n, ok in checks if not ok]


def action_features(action: Action, state: PolicyState, failing: Sequence[str]) -> np.ndarray:
    """候选动作的 18 维特征。Heuristic 与 Model 共用同一套特征。"""
    # Host supplied coverage is an observation, while action.coverage_ids only
    # describes what a candidate might help obtain. Never mark checks passed here.
    external = getattr(state, "external_checks", None)
    if external is not None:
        unmet = {key for key, status in external if status != "passed"}
        gain = len(unmet.intersection(action.coverage_ids)) / max(len(unmet), 1)
        f = np.zeros(N_FEAT, dtype=np.float32)
        if action.kind in KINDS:
            f[KINDS.index(action.kind)] = 1.0
        f[4] = action.cost_pressure
        f[5] = min(action.cost.tool_calls / 2.0, 1.0)
        f[10] = 1.0
        f[11] = gain
        f[12] = gain if action.kind in {"relation_expand", "bounded_path"} else 0.0
        f[13] = gain if action.kind == "passage_retrieve" else 0.0
        f[15] = gain if action.kind == "entity_lookup" else 0.0
        return f
    kind = action.kind
    cost = action.cost
    relation = action.args[1] if kind == "relation_expand" and len(action.args) > 1 else ""
    head = action.args[0] if action.args else ""

    need_reach = "seed->target 存在证据路径" in failing
    need_chain = "存在 >=2 跳的因果链" in failing
    need_text = "因果链有文本证据支撑" in failing
    chain_built = not need_chain

    f = np.zeros(N_FEAT, dtype=np.float32)
    for i, k in enumerate(KINDS):
        f[i] = 1.0 if kind == k else 0.0
    f[4] = min(cost.relation_access / 8.0, 1.0)
    f[5] = min(cost.tool_calls / 2.0, 1.0)
    f[6] = (relation_rank(relation) / len(RELATION_PRIORITY)) if relation else 0.0
    f[7] = 1.0 if relation in CAUSAL_RELATIONS else 0.0
    f[8] = 1.0 if head in state.seeds else 0.0
    f[9] = 1.0 if head in state.targets else 0.0
    f[10] = 1.0

    f[11] = 1.0 if (kind == "bounded_path" and need_reach) else 0.0
    f[12] = 1.0 if (kind == "relation_expand" and need_chain) else 0.0
    f[13] = 1.0 if (kind == "passage_retrieve" and need_text and chain_built) else 0.0
    f[14] = 1.0 if (kind == "relation_expand" and need_chain and relation in MECHANISM_RELATIONS) else 0.0
    has_unlooked = any(e not in state.looked_up for e in state.known_entities)
    f[15] = 1.0 if (kind == "entity_lookup" and (need_chain or need_reach) and has_unlooked) else 0.0
    if kind == "entity_lookup" and state.known_entities:
        idx = state.known_entities.index(head) if head in state.known_entities else 0
        f[16] = idx / max(len(state.known_entities) - 1, 1)   # 越新越大（深度优先）
    f[17] = 1.0 if (kind == "passage_retrieve" and not (need_text and chain_built)) else 0.0
    return f


def action_feature_vector(action: Action, state: PolicyState) -> np.ndarray:
    """公开的动作特征入口，供轨迹记录复用。"""
    return action_features(action, state, _failing_checks(state))


def _affordable(action: Action, state: PolicyState) -> bool:
    c = action.cost.as_dict()
    if getattr(state, "external_checks", None) is not None:
        return all(c[d] <= state.remaining[d] for d in state.remaining)
    return all(c[d] <= state.remaining.get(d, 0) for d in c)


class HeuristicSelector:
    """规则实现：缺口覆盖 + 关系先验 + 因果优先，除以成本。"""

    name = "heuristic"

    def select(self, state: PolicyState, candidates: Sequence[Action]) -> Selection:
        if not candidates:
            return Selection(None, StopReason.NO_LEGAL_ACTION.value)
        usable = [a for a in candidates if _affordable(a, state)]
        if not usable:
            return Selection(None, StopReason.BUDGET_EXHAUSTED.value)
        failing = _failing_checks(state)
        best, best_score = None, -1e9
        for a in usable:
            f = action_features(a, state, failing)
            gain = (1.0 * f[11] + 0.8 * f[12] + 1.0 * f[13] + 1.2 * f[14]
                    + 0.6 * f[15] + 0.4 * f[16] - 1.2 * f[17]
                    + 0.3 * (1.0 - f[6]) + 0.4 * f[7])
            score = (0.5 + gain) / (1.0 + 1.2 * f[4])
            if score > best_score:
                best, best_score = a, score
        return Selection(best)


class LinearModelSelector:
    """线性基线：效用头 − λ(剩余预算) × 成本预测头。

    λ(b) = λ0 · (1 + 2·(1 − 剩余比例)) / 总预算
      → 预算越少越保守；按总预算归一化保证与效用同量纲。
    这是与 BCR-RAG 固定 λ=1.0 的关键差异。
    """

    name = "linear"

    def __init__(self, lam0: float = 3.0, adaptive: bool = True, encoder=None) -> None:
        self.lam0 = lam0
        self.adaptive = adaptive
        from ..state_construction.encoder import ModelEncoder as _ME
        self.encoder = encoder or _ME()
        # 状态向量到效用/成本头的连接权重：初始 0，训练后生效
        self.w_state_u = np.zeros(self.encoder.dim, dtype=np.float32)
        self.w_state_c = np.zeros(self.encoder.dim, dtype=np.float32)
        w = np.zeros(N_FEAT, dtype=np.float32)
        # 动作类型偏好
        w[0], w[1], w[2], w[3] = 0.2, 1.0, 0.6, 0.4
        # 成本（负权重）
        w[4], w[5] = -0.5, -0.2
        # 关系先验 / 因果
        w[6], w[7] = -0.3, 0.6
        # 触及种子 / 目标
        w[8], w[9] = 0.3, 0.5
        w[10] = 0.0
        # 缺口相关（核心）
        w[11], w[12], w[13] = 2.0, 1.0, 1.0
        w[14] = 2.5   # 机制类关系优先
        w[15] = 1.5   # 前置查询
        w[16] = 0.8   # 最近发现优先
        w[17] = -1.5  # 过早检索文段惩罚
        self.w_utility = w
        # 成本预测头：≈ 恒等映射到预计成本
        c = np.zeros(N_FEAT, dtype=np.float32)
        c[4], c[5] = 8.0, 2.0
        self.w_cost = c

    def lam(self, state: PolicyState) -> float:
        limits = state.budget_limits or {}
        dimension = getattr(state, "budget_axis", "relation_access")
        total = max(limits.get(dimension, 1), 1)
        ratio = min(state.remaining.get(dimension, 0) / total, 1.0)
        if not self.adaptive:
            return self.lam0 / total
        return self.lam0 * (1.0 + 2.0 * (1.0 - ratio)) / total

    def utility(self, feats: np.ndarray) -> np.ndarray:
        return feats @ self.w_utility

    def cost_pred(self, feats: np.ndarray) -> np.ndarray:
        return np.maximum(feats @ self.w_cost, 0.0)

    def scores(self, state: PolicyState, candidates: Sequence[Action]) -> np.ndarray:
        failing = _failing_checks(state)
        feats = np.stack([action_features(a, state, failing) for a in candidates])
        z = self.encoder.encode(state)
        u = self.utility(feats) + float(z @ self.w_state_u)
        c = np.maximum(self.cost_pred(feats) + float(z @ self.w_state_c), 0.0)
        return u - self.lam(state) * c

    def select(self, state: PolicyState, candidates: Sequence[Action]) -> Selection:
        if not candidates:
            return Selection(None, StopReason.NO_LEGAL_ACTION.value)
        usable = [a for a in candidates if _affordable(a, state)]
        if not usable:
            return Selection(None, StopReason.BUDGET_EXHAUSTED.value)
        scores = self.scores(state, usable)
        return Selection(usable[int(np.argmax(scores))])

    def fit(self, X: np.ndarray, y_utility: np.ndarray, y_cost: np.ndarray) -> None:
        """预留：最小二乘拟合两个头（行为克隆 / 监督）。"""
        sol, *_ = np.linalg.lstsq(X, y_utility, rcond=None)
        self.w_utility = sol.astype(np.float32)
        sol2, *_ = np.linalg.lstsq(X, y_cost, rcond=None)
        self.w_cost = np.maximum(sol2.astype(np.float32), 0.0)


class ModelSelector:
    """默认实现：状态-动作联合 MLP 的效用/成本策略。

    utility = base_utility(a) + MLP_u([z_s, a])
    cost    = base_cost(a)    + ReLU(MLP_c([z_s, a]))
    score   = utility - λ(剩余预算) * cost

    两个 MLP 输出头零初始化；未训练时行为与线性启发式基线一致。
    """

    name = "model"

    def __init__(
        self,
        lam0: float = 3.0,
        adaptive: bool = True,
        encoder=None,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        device: str | None = None,
    ) -> None:
        self.lam0 = lam0
        self.adaptive = adaptive
        from ..state_construction.encoder import ModelEncoder as _ME
        self.encoder = encoder or _ME()
        self.hidden_dim = int(hidden_dim)
        self._torch = require_torch()
        self.device = self._torch.device(device or ("cuda" if self._torch.cuda.is_available() else "cpu"))
        self.in_dim = self.encoder.dim + N_FEAT
        self.trunk = make_trunk(self._torch, self.in_dim, self.hidden_dim, dropout=dropout).to(self.device)
        self.utility_head = zero_init_head(self._torch, self.hidden_dim, 1).to(self.device)
        self.cost_head = zero_init_head(self._torch, self.hidden_dim, 1).to(self.device)
        self.use_heuristic_base = True

        w = np.zeros(N_FEAT, dtype=np.float32)
        w[0], w[1], w[2], w[3] = 0.2, 1.0, 0.6, 0.4
        w[4], w[5] = -0.5, -0.2
        w[6], w[7] = -0.3, 0.6
        w[8], w[9] = 0.3, 0.5
        w[11], w[12], w[13] = 2.0, 1.0, 1.0
        w[14], w[15], w[16], w[17] = 2.5, 1.5, 0.8, -1.5
        self.base_utility = w

        c = np.zeros(N_FEAT, dtype=np.float32)
        c[4], c[5] = 8.0, 2.0
        self.base_cost = c

    def parameters(self):
        return (
            list(self.trunk.parameters())
            + list(self.utility_head.parameters())
            + list(self.cost_head.parameters())
        )

    def lam(self, state: PolicyState) -> float:
        limits = state.budget_limits or {}
        dimension = getattr(state, "budget_axis", "relation_access")
        total = max(limits.get(dimension, 1), 1)
        ratio = min(state.remaining.get(dimension, 0) / total, 1.0)
        if not self.adaptive:
            return self.lam0 / total
        return self.lam0 * (1.0 + 2.0 * (1.0 - ratio)) / total

    def _joint_features(self, state: PolicyState, candidates: Sequence[Action]) -> tuple[np.ndarray, np.ndarray]:
        failing = _failing_checks(state)
        feats = np.stack([action_features(a, state, failing) for a in candidates])
        z = self.encoder.encode(state)
        z_tiled = np.repeat(z[None, :], len(candidates), axis=0)
        return feats, np.concatenate([z_tiled, feats], axis=1).astype(np.float32)

    def scores(self, state: PolicyState, candidates: Sequence[Action]) -> np.ndarray:
        feats, joint = self._joint_features(state, candidates)
        if self.use_heuristic_base:
            base_u = feats @ self.base_utility
            base_c = feats @ self.base_cost
        else:
            base_u = np.zeros(len(candidates), dtype=np.float32)
            base_c = np.zeros(len(candidates), dtype=np.float32)

        z = as_tensor(self._torch, joint).to(self.device)
        with self._torch.no_grad():
            hidden = self.trunk(z)
            residual_u = self.utility_head(hidden).squeeze(-1).cpu().numpy()
            residual_c = self._torch.nn.functional.softplus(
                self.cost_head(hidden).squeeze(-1)
            ).cpu().numpy()
            if self.use_heuristic_base:
                residual_c = residual_c - float(np.log(2.0))

        utility = base_u + residual_u
        cost = np.maximum(base_c + residual_c, 0.0)
        return utility - self.lam(state) * cost

    def select(self, state: PolicyState, candidates: Sequence[Action]) -> Selection:
        if not candidates:
            return Selection(None, StopReason.NO_LEGAL_ACTION.value)
        usable = [a for a in candidates if _affordable(a, state)]
        if not usable:
            return Selection(None, StopReason.BUDGET_EXHAUSTED.value)
        scores = self.scores(state, usable)
        return Selection(usable[int(np.argmax(scores))])

    def fit(
        self,
        X: np.ndarray,
        y_utility: np.ndarray,
        y_cost: np.ndarray,
        *,
        epochs: int = 300,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
    ) -> None:
        """行为克隆：X 为 [state_embedding, action_features] 的批量矩阵。"""
        X_t = self._torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=self._torch.float32).to(self.device)
        y_u = self._torch.as_tensor(np.asarray(y_utility, dtype=np.float32), dtype=self._torch.float32).to(self.device)
        y_c = self._torch.as_tensor(np.asarray(y_cost, dtype=np.float32), dtype=self._torch.float32).to(self.device)
        optimizer = self._torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        n = X_t.shape[0]
        for _ in range(epochs):
            perm = self._torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                hidden = self.trunk(X_t[idx])
                pred_u = self.utility_head(hidden).squeeze(-1)
                pred_c = self._torch.nn.functional.softplus(self.cost_head(hidden).squeeze(-1))
                loss = self._torch.nn.functional.mse_loss(pred_u, y_u[idx]) + self._torch.nn.functional.mse_loss(
                    pred_c, y_c[idx]
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
        self.use_heuristic_base = False


SELECTORS = {
    HeuristicSelector.name: HeuristicSelector,
    LinearModelSelector.name: LinearModelSelector,
    ModelSelector.name: ModelSelector,
}