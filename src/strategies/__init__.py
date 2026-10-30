"""**兼容门面（deprecated）**——真实实现已按阶段拆分到独立包。

历史上七个阶段的实现全挤在 ``src/strategies/`` 一个目录里，
现已重构为「每个阶段一个包」：

    ① 解析         src/parsing/            pipeline.py + adapter.py + legacy.py
    ② 状态整合     src/state_construction/  build_state / sufficiency / encoder.py
    ③ 终止判断     src/stopping/            stop.py
    ④ 动作选择     src/action_selection/    selector.py
    ⑤ 执行         src/execution/           executor.py
    ⑥ 记账         src/accounting/          ledger.py
    ⑦ 证据更新     src/evidence/            updater.py
       装配与主循环  src/pipeline/           assembly.py + runner.py
       阶段契约      src/common/stage_contracts.py
       共用地基      src/infrastructure/     embedder.py / llm.py / nets.py

本文件只做 re-export，保证 ``from src.strategies import build_pipeline``
和 ``from src.strategies.selector import ModelSelector`` 这类旧 import 仍可用。
新代码请直接 import 对应阶段的包。
"""

from __future__ import annotations

from ..accounting import LEDGERS
from ..accounting import BudgetLedger as BudgetLedgerImpl
from ..action_selection import (
    SELECTORS,
    HeuristicSelector,
    LinearModelSelector,
    ModelSelector,
)
from ..common.stage_contracts import (
    ActionExecutor,
    ActionSelector,
    BudgetLedger,
    EvidenceStore,
    EvidenceUpdater,
    ParsedQuestion,
    QuestionParser,
    Selection,
    StateEncoder,
    StopRule,
)
from ..evidence import UPDATERS, StandardUpdater
from ..execution import EXECUTORS, PrimeKGExecutor, ToyKGExecutor
from ..infrastructure.embedder import DEFAULT_EMBED_MODEL, SentenceEmbedder
from ..parsing.adapter import PipelineParserAdapter
from ..parsing.legacy import BioHopRParser, HybridParser, ModelParser, PARSERS, RuleParser
from ..pipeline import DEFAULTS, REGISTRY, STEPS, Pipeline, build_pipeline
from ..state_construction.encoder import ENCODERS, FeatureEncoder, ModelEncoder
from ..stopping import (
    STOP_RULES,
    FlatModelStop,
    HierarchicalModelStop,
    NaiveTargetStop,
    RuleStop,
)

__all__ = [
    "Pipeline", "build_pipeline", "REGISTRY", "DEFAULTS", "STEPS",
    "PARSERS", "ENCODERS", "STOP_RULES", "SELECTORS", "EXECUTORS", "LEDGERS", "UPDATERS",
    "ModelParser", "RuleParser", "HybridParser", "BioHopRParser", "ModelEncoder", "FeatureEncoder",
    "HierarchicalModelStop", "FlatModelStop", "RuleStop", "NaiveTargetStop", "ModelSelector", "LinearModelSelector",
    "HeuristicSelector", "ToyKGExecutor", "PrimeKGExecutor", "BudgetLedgerImpl", "StandardUpdater", "PipelineParserAdapter",
    # 契约层
    "ParsedQuestion", "QuestionParser", "StateEncoder", "StopRule", "ActionSelector", "Selection",
    "ActionExecutor", "BudgetLedger", "EvidenceStore", "EvidenceUpdater",
    # 基础设施
    "DEFAULT_EMBED_MODEL", "SentenceEmbedder",
]