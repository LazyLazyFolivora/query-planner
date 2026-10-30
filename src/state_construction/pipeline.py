"""② 状态构建的单一入口：外层只调这一个函数，② 的七步全在里面。

对标第一阶段 src/parsing/pipeline.py：那里 `ParsingPipeline.parse()` 的
函数体就是① 的完整流程；这里 `build_state()` 的函数体就是② 的完整流程。

    build_state(question, parsed, store, ledger, step_index,
                kg_entity_types=..., enumerate_candidates=...)
        │
        ├─ ① 目标类型组   entity_type_groups()    终点实体名 -> 粗粒度类型组（查表）
        ├─ ② 必需关系组   relation_group_list()   细粒度关系 -> 粗粒度关系组（查表）
        ├─ ③ 已知实体集   known_entity_list()     起点实体 + 证据里出现过的 head/tail
        ├─ ④ 进度信号     recent_actions()        最近 K 个动作类型
        │                 no_progress_streak()   连续几步没有新增证据
        ├─ ⑤ 预算快照     ledger.remaining()      各维度剩余额度
        ├─ ⑥ 汇总快照     -> PolicyState          把上面五组填进状态结构
        └─ ⑦ 动作空间摘要 annotate_action_space()还能做几个动作、其中几个付得起
                ↑
                ⑦ 依赖⑤ 执行层枚举候选动作，所以 enumerate_candidates
                由调用方（主循环）以回调注入。这样② 不 import 执行层，
                不会形成 state_construction -> strategies -> kg 的循环依赖；
                也保证 legal_actions 只被调用一次。

主循环里的完整用法（就是一行）：

    state = build_state(question, parsed, store, ledger, step_no,
                        kg_entity_types=kg.entity_types,
                        enumerate_candidates=executor.legal_actions)

技术性质：全流程是**确定性算法**（查表 / 集合运算 / 数数），
不含 LLM，也不含深度学习模型，也不学习任何参数。
"""
from __future__ import annotations

from typing import Callable, Sequence

from ..schema import Action, Question
from ..constants import entity_type_group, relation_group
from .action_space import annotate_action_space  # noqa: F401  第七步入口，供 runner 一同导入
from .coverage import evidence_triples
from .models import PolicyState
from .progress import no_progress_streak, recent_actions


# ---------------------------------------------------------------- 派生量


def entity_type_groups(entity_types: dict[str, str], names) -> tuple[str, ...]:
    """实体名 -> 粗粒度类型组（查 constants 表，去重保序）。"""
    out: list[str] = []
    for n in names:
        g = entity_type_group(entity_types.get(n, ""))
        if g and g not in out:
            out.append(g)
    return tuple(out)


def relation_group_list(relations) -> tuple[str, ...]:
    """细粒度关系名 -> 粗粒度关系组（查 constants 表，去重保序）。"""
    out: list[str] = []
    for r in relations:
        g = relation_group(r)
        if g not in out:
            out.append(g)
    return tuple(out)


def known_entity_list(parsed, evidence) -> tuple[str, ...]:
    """当前"已知的实体"：起点实体 + 证据里出现过的 head/tail（去重保序）。

    用途：⑤ 执行层据此判断还能对谁做 entity_lookup；
    ④ 动作选择也用它算"最近发现优先"。
    """
    triples = evidence_triples(tuple(evidence))
    out: list[str] = []
    for e in list(parsed.seeds) + [t.head for t in triples] + [t.tail for t in triples]:
        if e not in out:
            out.append(e)
    return tuple(out)


def build_state(
    question: Question,
    parsed,
    store,
    ledger,
    step_index: int,
    *,
    kg_entity_types: dict[str, str] | None = None,
    enumerate_candidates: Callable[[PolicyState], Sequence[Action]] | None = None,
) -> PolicyState:
    """② 状态构建：把整轮上下文汇总成一份确定性状态快照（七步一次做完）。

    输入（● 每步变化，○ 整轮不变）：
        ○ question         问题参数：qtype / max_steps / sufficiency_threshold
        ○ parsed           ① 解析结果：起点、终点、必需关系、推理模式、目标类型
        ● store            ⑦ 证据容器：证据、查过的实体、试过的路径、动作历史
        ● ledger           ⑥ 预算账本：各维度已花与剩余
        ● step_index       当前步序号
        ○ kg_entity_types  KG 加载期建好的"实体名 -> 类型"表
        ● enumerate_candidates  ⑤ 执行层枚举合法动作的回调（可为 None，此时动作空间两个字段留空）

    输出：
        PolicyState —— ③ 终止判断与 ④ 动作选择唯一读取的状态对象。
        七个步骤全部做完，含动作空间摘要（legal_action_count /
        affordable_action_count）；只有 enumerate_candidates 为 None 时这两项才是 None。

    不计算任何 gold；不含 LLM 与深度学习模型。
    """

    # 保存 证据快照
    evidence = tuple(store.evidence)

    # ---- ① 目标类型组：终点实体名 -> 粗粒度类型组（查 constants 表）----
    # 用于"类型目标"问题：终点没链接到具体实体时，按类型组判定是否到达。
    target_types = tuple(dict.fromkeys(
        entity_type_groups(kg_entity_types or {}, parsed.targets) + parsed.target_types
    ))

    # ---- ② 必需关系组：细粒度关系 -> 粗粒度关系组（查 constants 表）----
    # 用于"关系覆盖"检查：把 30+ 种细关系降维成6 个组做集合覆盖判定。
    required_groups = relation_group_list(parsed.required_relations)

    # ---- ③ 已知实体集：起点实体 + 证据里出现过的一切实体 ----
    # ⑤ 据此判断还能对谁做 entity_lookup；④ 据此算"最近发现优先"。
    known_entities = known_entity_list(parsed, evidence)

    # ---- ④ 进度信号：最近做过什么、是不是卡住了 ----
    history = store.history
    recent = recent_actions(history)
    streak = no_progress_streak(history)

    # ---- ⑤ 预算快照：各维度还剩多少 ----
    remaining = ledger.remaining()
    limits = dict(ledger.limits)

    # ---- ⑥ 汇总快照：把上面五组填进 PolicyState ----
    state = PolicyState(
        # --- 问题与解析结果（整轮不变）---
        qid=question.qid,
        question=question.question,
        qtype=question.qtype,
        seeds=parsed.seeds,
        targets=parsed.targets,
        relation_keywords=parsed.required_relations,
        operator=parsed.operator,
        excluded_entities=parsed.excluded_entities,
        target_types=target_types,
        required_relation_groups=required_groups,
        required_relation_chain=parsed.required_relation_chain,
        query_intent=getattr(parsed, "query_intent", None),
        evidence_spec=getattr(parsed, "evidence_spec", None),
        parse_status=getattr(parsed, "parse_status", "legacy"),
        unlinked_mentions=getattr(parsed, "unlinked_mentions", ()),
        # --- 证据与覆盖痕迹（每步变化）---
        evidence=evidence,
        known_entities=known_entities,
        relations_seen=dict(store.relations_seen),
        looked_up=frozenset(store.looked_up),
        expanded=frozenset(store.expanded),
        tried_paths=frozenset(store.tried_paths),
        tried_passages=store.tried_passages,
        # --- 进度与预算（每步变化）---
        recent_actions=recent,
        no_progress_streak=streak,
        remaining=remaining,
        budget_limits=limits,
        # --- 步序与停止参数 ---
        step_index=step_index,
        sufficiency_threshold=question.sufficiency_threshold,
        max_steps=question.max_steps,
        # --- 动作空间摘要：第 ⑦ 步算，见下 ---
        legal_action_count=None,
        affordable_action_count=None,
    )

    # ---- ⑦ 动作空间摘要：还能做几个动作、其中几个还付得起 ----
    # ③ 终止判断要回答"现在还有没有路可走"，所以这一步不能省。
    # 候选动作由 ⑤ 执行层定义（legal_actions 住在 kg.py / primekg.py），
    # 以回调注入而不是直接 import，避免 state_construction -> strategies -> kg 的循环依赖。
    if enumerate_candidates is not None:
        candidates = tuple(enumerate_candidates(state))
        annotate_action_space(state, candidates)
        state.candidate_actions = candidates

    return state


def candidates_of(
    state: PolicyState,
    enumerate_candidates: Callable[[PolicyState], Sequence[Action]] | None = None,
) -> list[Action]:
    """取出当前状态的候选动作列表（④ 动作选择用）。

    正常路径下 build_state 已经枚举过并缓存，这里直接读缓存；
    只有在状态不是由 build_state 产出（测试里手搓的 state）时，
    才现场调一次回调。
    """
    if state.candidate_actions is not None:
        return list(state.candidate_actions)
    if enumerate_candidates is None:
        return []
    return list(enumerate_candidates(state))
