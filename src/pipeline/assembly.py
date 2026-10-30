"""Pipeline 装配：把七个阶段的实现按名字拼成一条可运行的链路。

    Pipeline = ①解析 + ②状态整合/编码 + ③终止判断 + ④动作选择 + ⑤执行 + ⑥记账 + ⑦证据更新

**默认全部走模型实现**（①pipeline 解析 / ②model 编码 / ③model 三态头 / ④model 选择器），
规则实现作为基线保留，用于消融对比。

① 解析这一步的默认值是 **pipeline**：它不是又一个同链路的"模型策略"，
而是新版解析层（SemanticParser + KG grounding + EvidenceSpec）的入口，
走 LLM 改写 + 词表链接。需要完全离线、可复现的基线时用 `--parser rule`。

装配只做三件事：选名字、实例化、把依赖注入进去。它不含任何算法。
主循环在 runner.py，阶段实现分别在各自包里。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable

from ..accounting import LEDGERS, BudgetLedger as BudgetLedgerImpl
from ..action_selection import SELECTORS, ModelSelector
from ..common.stage_contracts import (
    ActionExecutor, ActionSelector, BudgetLedger, EvidenceUpdater, QuestionParser, StateEncoder, StopRule,
)
from ..evidence import UPDATERS
from ..execution import EXECUTORS
from ..infrastructure.embedder import DEFAULT_EMBED_MODEL, SentenceEmbedder
from ..kg import ToyKG
from ..parsing.adapter import PipelineParserAdapter
from ..parsing.legacy import BioHopRParser, HybridParser, ModelParser, PARSERS, RuleParser
from ..state_construction.encoder import ENCODERS, ModelEncoder
from ..stopping import STOP_RULES

PARSERS[PipelineParserAdapter.name] = PipelineParserAdapter

REGISTRY: dict[str, dict] = {
    "parser": PARSERS,
    "encoder": ENCODERS,
    "stop": STOP_RULES,
    "selector": SELECTORS,
    "executor": EXECUTORS,
    "ledger": LEDGERS,
    "updater": UPDATERS,
}

# 每一步的默认实现：**优先模型**
DEFAULTS = {
    # LLM 改写 + KG 链接（新版解析层）。它不是同一条链路上的另一个策略，
    # 而是把解析层整体换成 LLM 驱动的实现。
    "parser": "pipeline",
    "encoder": "model",
    "stop": "model",
    "selector": "model",
    "executor": "toy_kg",
    "ledger": "budget",
    "updater": "standard",
}

STEPS = tuple(DEFAULTS)


@dataclass
class Pipeline:
    parser: QuestionParser
    encoder: StateEncoder
    stop: StopRule
    selector: ActionSelector
    executor: ActionExecutor
    ledger_factory: Callable[[dict], BudgetLedger]
    updater: EvidenceUpdater
    names: dict[str, str]

    def name(self) -> str:
        return "|".join(f"{k}={self.names[k]}" for k in STEPS)

    def short_name(self) -> str:
        return f"{self.names['selector']}/{self.names['stop']}"


def build_pipeline(kg: ToyKG, llm=None, embedder=None, **overrides: str) -> Pipeline:
    names = {**DEFAULTS, **{k: v for k, v in overrides.items() if v}}
    for step in STEPS:
        if names[step] not in REGISTRY[step]:
            raise KeyError(f"未知的 {step} 实现：{names[step]}（可选 {sorted(REGISTRY[step])}）")

    needs_embedder = names["encoder"] == ModelEncoder.name or names["parser"] in {
        ModelParser.name,
        HybridParser.name,
        PipelineParserAdapter.name,
    }
    if needs_embedder and embedder is None:
        try:
            embedder = SentenceEmbedder(DEFAULT_EMBED_MODEL)
        except RuntimeError:
            # 缺依赖时不抛：该步骤支持无依赖的 feature 实现，降级比整条链路拒绝运行更有用
            embedder = None

    def _build_with_kwargs(cls, **kwargs):
        params = inspect.signature(cls).parameters
        selected = {k: v for k, v in kwargs.items() if k in params and v is not None}
        return cls(**selected)

    encoder = _build_with_kwargs(
        REGISTRY["encoder"][names["encoder"]],
        embedder=embedder,
    )
    stop = _build_with_kwargs(
        REGISTRY["stop"][names["stop"]],
        encoder=encoder,
    )
    selector = _build_with_kwargs(
        REGISTRY["selector"][names["selector"]],
        encoder=encoder,
    )
    parser = _build_with_kwargs(
        REGISTRY["parser"][names["parser"]],
        llm=llm,
        embedder=embedder,
    )

    return Pipeline(
        parser=parser,
        encoder=encoder,
        stop=stop,
        selector=selector,
        executor=REGISTRY["executor"][names["executor"]](kg),
        ledger_factory=REGISTRY["ledger"][names["ledger"]],
        updater=REGISTRY["updater"][names["updater"]](),
        names=names,
    )


__all__ = [
    "Pipeline", "build_pipeline", "REGISTRY", "DEFAULTS", "STEPS",
    "PARSERS", "ENCODERS", "STOP_RULES", "SELECTORS", "EXECUTORS", "LEDGERS", "UPDATERS",
]