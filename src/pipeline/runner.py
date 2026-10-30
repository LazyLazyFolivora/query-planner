"""运行时调度器：把主循环跑起来。

每一步都通过**策略接口**调用，runner 本身不含任何具体算法：

    parsed  = pipeline.parser.parse(q, kg)            ① 解析（只在开头跑一次）
    state   = build_policy_state(...)                 ② 状态构建（state_construction）
    cands   = executor.legal_actions(state)          ⑤ 枚举候选动作
    annotate_action_space(state, cands)              ② 回填"还能做几个动作"
    d, why  = pipeline.stop.decide(state)             ③ 终止判断
    cands   = pipeline.executor.legal_actions(state)  ⑤ 执行层给出合法动作
    sel     = pipeline.selector.select(state, cands)  ④ 动作选择
    ev, c   = pipeline.executor.execute(sel.action)   ⑤ 执行
    ledger.charge(c)                                  ⑥ 记账
    pipeline.updater.update(store, a, ev, c)          ⑦ 更新证据 + 记录历史

换任何一步的实现都不需要改本文件。
"""
from __future__ import annotations

from ..kg import ToyKG
from ..schema import ActionCost, Decision, Question, RunTrace, Step
from ..state_construction import (
    PolicyState,
    candidates_of,
    build_state,
    observable_sufficiency,
)
from .assembly import Pipeline
from ..common.stage_contracts import EvidenceStore


class Session:
    def __init__(self, question: Question, kg: ToyKG, pipeline: Pipeline,
                 record_transitions: bool = False) -> None:
        self.q = question
        self.kg = kg
        self.pipe = pipeline
        self.ledger = pipeline.ledger_factory(question.budget)
        self.store = EvidenceStore()
        self.parsed = pipeline.parser.parse(question.question, kg)
        self.record_transitions = bool(record_transitions)


    # ---------------- 主循环 ----------------
    def _record_stop(self, trace: RunTrace, step_no: int, state: PolicyState,
                     decision: Decision, reason: str) -> None:
        suff, _ = observable_sufficiency(state)
        trace.steps.append(Step(step_no, decision, reason, None, ActionCost(),
                                self.ledger.remaining(), suff, [], ""))
        trace.final_decision = decision
        trace.final_reason = reason

    def run(self) -> RunTrace:
        trace = RunTrace(qid=self.q.qid, question=self.q.question, policy=self.pipe.name())
        step_no = 0
        guard = self.q.max_steps + 3
        while step_no <= guard:
            # 2. 状态构建：一次调用完成七步（含枚举候选动作并回填动作空间摘要）。
            # legal_actions 以回调注入，所以它在这里只被调用一次。
            state = build_state(
                self.q,
                self.parsed,
                self.store,
                self.ledger,
                step_no,
                kg_entity_types=self.kg.entity_types,
                enumerate_candidates=self.pipe.executor.legal_actions,
            )

            state_vector = self.pipe.encoder.encode(state).tolist() if self.record_transitions else []
            stop_probabilities = None
            if self.record_transitions and hasattr(self.pipe.stop, "probabilities"):
                try:
                    stop_probabilities = self.pipe.stop.probabilities(state).tolist()
                except Exception:
                    stop_probabilities = None

            ## 3. 决定是否停止
            decision, reason = self.pipe.stop.decide(state)
            if decision is not Decision.CONTINUE:
                if self.record_transitions:
                    trace.transitions.append({
                        "qid": self.q.qid,
                        "step": step_no,
                        "state_vector": state_vector,
                        "decision": decision.value,
                        "stop_reason": reason,
                        "stop_probabilities": stop_probabilities,
                        "action": None,
                        "action_features": [],
                        "estimated_cost": {},
                        "actual_cost": {},
                        "new_evidence_ids": [],
                        "note": "",
                    })
                self._record_stop(trace, step_no, state, decision, reason)
                break

            ## 4. 下一步选择什么动作
            selection = self.pipe.selector.select(state, candidates_of(state))
            if selection.action is None:
                if self.record_transitions:
                    trace.transitions.append({
                        "qid": self.q.qid,
                        "step": step_no,
                        "state_vector": state_vector,
                        "decision": Decision.INSUFFICIENT.value,
                        "stop_reason": selection.blocked_reason,
                        "stop_probabilities": stop_probabilities,
                        "action": None,
                        "action_features": [],
                        "estimated_cost": {},
                        "actual_cost": {},
                        "new_evidence_ids": [],
                        "note": "",
                    })
                self._record_stop(trace, step_no, state, Decision.INSUFFICIENT,
                                  selection.blocked_reason)
                break


            ## 轨迹记录
            step_no += 1
            action = selection.action
            transition = None
            if self.record_transitions:
                from ..action_selection.pipeline import action_feature_vector
                transition = {
                    "qid": self.q.qid,
                    "step": step_no - 1,
                    "state_vector": state_vector,
                    "decision": Decision.CONTINUE.value,
                    "stop_reason": "",
                    "stop_probabilities": stop_probabilities,
                    "action": action.aid,
                    "action_kind": action.kind,
                    "action_args": list(action.args),
                    "action_features": action_feature_vector(action, state).tolist(),
                    "estimated_cost": action.cost.as_dict(),
                    "actual_cost": {},
                    "new_evidence_ids": [],
                    "note": "",
                }
                trace.transitions.append(transition)

            ## 5. 执行：到 数据库/知识图谱 执行相应的操作，返回结果和真实成本
            new_evidence, cost, note = self.pipe.executor.execute(action, step_no)
            
            ## 6. 记账
            self.ledger.charge(cost, action.aid)

            ## 7. 更新证据
            self.pipe.updater.update(self.store, action, new_evidence, cost)
            if transition is not None:
                transition["actual_cost"] = cost.as_dict()
                transition["new_evidence_ids"] = [e.eid for e in new_evidence]
                transition["note"] = note


            ## 记录trace
            suff_after, _ = observable_sufficiency(build_state(
                self.q, self.parsed, self.store, self.ledger, step_no,
                kg_entity_types=self.kg.entity_types,
            ))
            trace.steps.append(
                Step(step_no, Decision.CONTINUE, "-", action, cost,
                     self.ledger.remaining(), suff_after, new_evidence, note)
            )

        
        trace.spent = self.ledger.spent()
        trace.evidence_count = len(self.store.evidence)
        trace.final_evidence = list(self.store.evidence)
        return trace


def run_question(question: Question, kg: ToyKG, pipeline: Pipeline,
                 record_transitions: bool = False) -> RunTrace:
    return Session(question, kg, pipeline, record_transitions=record_transitions).run()

