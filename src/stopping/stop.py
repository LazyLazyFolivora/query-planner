"""③ 终止判断策略：三态（continue / complete / insufficient）。

实现分层：

  RuleStop              规则三态基线
  FlatModelStop         旧版扁平三分类头基线（线性）
  HierarchicalModelStop 默认：MLP + p_stop + p_complete|stop
  NaiveTargetStop       过早停止基线

默认 HierarchicalModelStop 的结构：

    h = MLP(s)
    p_stop     = σ(z_stop(s) + MLP_stop(h))
    p_complete = σ(z_complete(s) + MLP_complete(h))

硬规则负责安全底线；MLP 负责学习残差。未训练时 MLP 输出头零初始化，
因此默认行为退化为启发式三态规则。
"""
from __future__ import annotations

import numpy as np

from ..schema import Decision, StopReason
from ..state_construction import PolicyState, observable_sufficiency
from ..common.stage_contracts import StopRule
from ..state_construction.encoder import ModelEncoder
from ..infrastructure.nets import as_tensor, make_trunk, require_torch, zero_init_head

NO_PROGRESS_LIMIT = 4
DEFAULT_COMPLETE_THRESHOLD = 0.65


def _budget_ratio(state: PolicyState) -> float:
    limits = state.budget_limits or {}
    total = max(limits.get("relation_access", 1), 1)
    return min(state.remaining.get("relation_access", 0) / total, 1.0)


def _budget_exhausted(state: PolicyState) -> bool:
    return (
        state.remaining.get("relation_access", 0) <= 0
        or state.remaining.get("tool_calls", 0) <= 0
    )


def _affordable_action_count(state: PolicyState) -> int | None:
    """返回可负担动作数；旧状态没有该字段时回退到合法动作数。"""
    if state.affordable_action_count is not None:
        return state.affordable_action_count
    if state.legal_action_count is not None:
        return state.legal_action_count
    return None


def _no_feasible_action(state: PolicyState) -> bool:
    count = _affordable_action_count(state)
    return count is not None and count == 0


def _hard_insufficient_reason(state: PolicyState) -> str:
    """安全底线：这些条件不允许交给学习模型自由选择。"""
    if _budget_exhausted(state):
        return StopReason.BUDGET_EXHAUSTED.value
    if _no_feasible_action(state):
        return StopReason.NO_LEGAL_ACTION.value
    if state.step_index >= state.max_steps:
        return StopReason.MAX_STEPS.value
    if state.no_progress_streak >= NO_PROGRESS_LIMIT:
        return StopReason.NO_PROGRESS.value
    return ""


def _hard_complete(state: PolicyState) -> bool:
    """四项可观测检查已达标时，直接判 complete，避免预算低时被误判 insufficient。"""
    score, _ = observable_sufficiency(state)
    return score >= state.sufficiency_threshold


def _decision_index(label) -> int:
    if isinstance(label, Decision):
        return {
            Decision.CONTINUE: 0,
            Decision.COMPLETE: 1,
            Decision.INSUFFICIENT: 2,
        }[label]
    text = str(label).lower()
    if text in {"0", "continue"}:
        return 0
    if text in {"1", "complete"}:
        return 1
    if text in {"2", "insufficient"}:
        return 2
    raise ValueError(f"未知停止标签：{label!r}")


class RuleStop:
    """基线：三态规则。"""

    name = "rule"

    def decide(self, state: PolicyState) -> tuple[Decision, str]:
        if _hard_complete(state):
            return Decision.COMPLETE, StopReason.SUFFICIENT.value
        reason = _hard_insufficient_reason(state)
        if reason:
            return Decision.INSUFFICIENT, reason
        return Decision.CONTINUE, ""


class FlatModelStop:
    """旧版基线：扁平三分类线性打分头，用于消融。"""

    name = "flat"

    def __init__(self, encoder: ModelEncoder | None = None) -> None:
        self.encoder = encoder or ModelEncoder()
        self.a = np.asarray([10.0, 0.5, -9.5], dtype=np.float32)          # complete
        self.b = np.asarray([8.0, 4.0, 3.0, -6.0], dtype=np.float32)      # insufficient
        self.c = np.asarray([6.0, 1.0, -1.0], dtype=np.float32)           # continue
        self.w_state = np.zeros((3, self.encoder.dim), dtype=np.float32)

    def features(self, state: PolicyState) -> np.ndarray:
        suff, _ = observable_sufficiency(state)
        thr = state.sufficiency_threshold
        stall = min(state.no_progress_streak / max(NO_PROGRESS_LIMIT, 1), 1.0)
        no_action = 1.0 if _no_feasible_action(state) else 0.0
        return np.asarray(
            [suff, thr, 1.0 - _budget_ratio(state), stall, no_action],
            dtype=np.float32,
        )

    def scores(self, state: PolicyState) -> np.ndarray:
        suff, thr, budget_gap, stall, no_action = self.features(state)
        s_complete = float(self.a @ np.asarray([suff, 1.0 - budget_gap, 1.0], dtype=np.float32))
        s_insufficient = float(
            self.b @ np.asarray([1.0 - suff, budget_gap, stall + no_action, 1.0], dtype=np.float32)
        )
        s_continue = float(self.c @ np.asarray([thr - suff, 1.0 - budget_gap, 1.0], dtype=np.float32))
        base = np.asarray([s_continue, s_complete, s_insufficient], dtype=np.float32)
        return base + self.w_state @ self.encoder.encode(state)

    def probabilities(self, state: PolicyState) -> np.ndarray:
        logits = self.scores(state)
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def decide(self, state: PolicyState) -> tuple[Decision, str]:
        if _hard_complete(state):
            return Decision.COMPLETE, StopReason.SUFFICIENT.value
        reason = _hard_insufficient_reason(state)
        if reason:
            return Decision.INSUFFICIENT, reason

        idx = int(np.argmax(self.scores(state)))
        if idx == 1:
            return Decision.COMPLETE, StopReason.SUFFICIENT.value
        if idx == 2:
            _, _, budget_gap, _, _ = self.features(state)
            reason = (
                StopReason.BUDGET_EXHAUSTED.value
                if budget_gap >= 0.5
                else StopReason.LOW_CONTINUATION_VALUE.value
            )
            return Decision.INSUFFICIENT, reason
        return Decision.CONTINUE, ""

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """预留：最小二乘拟合三分类线性头（行为克隆）。"""
        labels = np.asarray([_decision_index(v) for v in y], dtype=np.int64)
        for k, w in enumerate((self.c, self.a, self.b)):
            sel = labels == k
            if sel.sum() >= 2:
                sol, *_ = np.linalg.lstsq(X[sel], np.ones(int(sel.sum()), dtype=np.float32), rcond=None)
                w[:] = sol.astype(np.float32)


class HierarchicalModelStop:
    """默认实现：预算感知的分层三态 MLP 停止器。

    启发式基线负责提供先验，MLP 输出头零初始化负责学习残差。
    这样未训练时行为可解释，训练后才逐步改变规则。
    """

    name = "model"

    def __init__(
        self,
        encoder: ModelEncoder | None = None,
        complete_threshold: float = DEFAULT_COMPLETE_THRESHOLD,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        device: str | None = None,
    ) -> None:
        self.encoder = encoder or ModelEncoder()
        self.complete_threshold = float(complete_threshold)
        self.hidden_dim = int(hidden_dim)
        self._torch = require_torch()
        self.device = self._torch.device(device or ("cuda" if self._torch.cuda.is_available() else "cpu"))
        self.trunk = make_trunk(self._torch, self.encoder.dim, self.hidden_dim, dropout=dropout).to(self.device)
        self.stop_head = zero_init_head(self._torch, self.hidden_dim, 1).to(self.device)
        self.complete_head = zero_init_head(self._torch, self.hidden_dim, 1).to(self.device)
        self.use_heuristic_base = True

    def parameters(self):
        return (
            list(self.trunk.parameters())
            + list(self.stop_head.parameters())
            + list(self.complete_head.parameters())
        )

    def _residual_logits(self, z):
        hidden = self.trunk(z)
        stop = self.stop_head(hidden).squeeze(-1)
        complete = self.complete_head(hidden).squeeze(-1)
        return stop, complete

    def _residual_stop(self, state: PolicyState) -> float:
        z = as_tensor(self._torch, self.encoder.encode(state)).to(self.device)
        with self._torch.no_grad():
            stop, _ = self._residual_logits(z)
        return float(stop)

    def _residual_complete(self, state: PolicyState) -> float:
        z = as_tensor(self._torch, self.encoder.encode(state)).to(self.device)
        with self._torch.no_grad():
            _, complete = self._residual_logits(z)
        return float(complete)

    def heuristic_features(self, state: PolicyState) -> np.ndarray:
        suff, _ = observable_sufficiency(state)
        thr = state.sufficiency_threshold
        budget_gap = 1.0 - _budget_ratio(state)
        stall = min(state.no_progress_streak / max(NO_PROGRESS_LIMIT, 1), 1.0)
        no_action = 1.0 if _no_feasible_action(state) else 0.0
        return np.asarray([suff, thr, budget_gap, stall, no_action], dtype=np.float32)

    def stop_logit(self, state: PolicyState) -> float:
        residual = self._residual_stop(state)
        if not self.use_heuristic_base:
            return residual
        suff, _, budget_gap, stall, no_action = self.heuristic_features(state)
        base = 10.0 * (suff - 0.9) + 1.5 * budget_gap + 2.0 * stall + 2.0 * no_action
        return float(base + residual)

    def complete_logit(self, state: PolicyState) -> float:
        residual = self._residual_complete(state)
        if not self.use_heuristic_base:
            return residual
        suff, thr, _, _, _ = self.heuristic_features(state)
        base = 12.0 * (suff - thr) + 0.5
        return float(base + residual)

    def stop_probability(self, state: PolicyState) -> float:
        logit = np.clip(self.stop_logit(state), -30.0, 30.0)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def complete_probability(self, state: PolicyState) -> float:
        """P(complete | stop, s)。"""
        logit = np.clip(self.complete_logit(state), -30.0, 30.0)
        return float(1.0 / (1.0 + np.exp(-logit)))

    def probabilities(self, state: PolicyState) -> np.ndarray:
        """返回联合三态概率 [continue, complete, insufficient]。"""
        p_stop = self.stop_probability(state)
        p_complete = self.complete_probability(state)
        return np.asarray(
            [1.0 - p_stop, p_stop * p_complete, p_stop * (1.0 - p_complete)],
            dtype=np.float32,
        )

    def decide(self, state: PolicyState) -> tuple[Decision, str]:
        if _hard_complete(state):
            return Decision.COMPLETE, StopReason.SUFFICIENT.value

        reason = _hard_insufficient_reason(state)
        if reason:
            return Decision.INSUFFICIENT, reason

        p_stop = self.stop_probability(state)
        if p_stop < 0.5:
            return Decision.CONTINUE, ""

        p_complete = self.complete_probability(state)
        if p_complete >= self.complete_threshold:
            return Decision.COMPLETE, StopReason.SUFFICIENT.value

        if _budget_ratio(state) <= 0.25:
            reason = StopReason.BUDGET_EXHAUSTED.value
        elif state.no_progress_streak > 0:
            reason = StopReason.NO_PROGRESS.value
        else:
            reason = StopReason.LOW_CONTINUATION_VALUE.value
        return Decision.INSUFFICIENT, reason

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        epochs: int = 300,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
    ) -> None:
        """行为克隆：分别学习 p_stop 与 p_complete|stop。

        X 为 `encoder.encode(state)` 的批量矩阵；
        y 支持 `0/1/2` 或 `continue/complete/insufficient`。
        fit 后关闭启发式 base，完全使用 MLP logits。
        """
        labels = np.asarray([_decision_index(v) for v in y], dtype=np.int64)
        X_t = self._torch.as_tensor(np.asarray(X, dtype=np.float32), dtype=self._torch.float32).to(self.device)
        stop_y = self._torch.as_tensor((labels != 0).astype(np.float32), dtype=self._torch.float32).to(self.device)
        complete_y = self._torch.as_tensor((labels == 1).astype(np.float32), dtype=self._torch.float32).to(self.device)

        optimizer = self._torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_fn = self._torch.nn.BCEWithLogitsLoss()
        n = X_t.shape[0]
        for _ in range(epochs):
            perm = self._torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                stop_logits, complete_logits = self._residual_logits(X_t[idx])
                loss = loss_fn(stop_logits, stop_y[idx]) + loss_fn(complete_logits, complete_y[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self._torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
        self.use_heuristic_base = False


class NaiveTargetStop:
    """基线：只要结构化证据里出现过目标实体就停（暴露"过早停止"）。"""

    name = "naive_target"

    def decide(self, state: PolicyState) -> tuple[Decision, str]:
        if state.targets:
            from ..state_construction import evidence_triples
            if any(t.tail in state.targets for t in evidence_triples(state.evidence)):
                return Decision.COMPLETE, StopReason.TARGET_FOUND.value
        if state.step_index >= state.max_steps:
            return Decision.INSUFFICIENT, StopReason.MAX_STEPS.value
        return Decision.CONTINUE, ""


STOP_RULES = {
    RuleStop.name: RuleStop,
    FlatModelStop.name: FlatModelStop,
    HierarchicalModelStop.name: HierarchicalModelStop,
    NaiveTargetStop.name: NaiveTargetStop,
}
