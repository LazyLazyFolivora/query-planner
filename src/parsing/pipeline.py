"""Step 1 facade: compose semantic parsing and KG grounding.

    question ──► SemanticParser ──► IntentDraft
                                     │
    KG vocabulary ───────────────────┴──► IntentGrounder ──► QueryIntent
                                                              │
                                              EvidenceSpecBuilder ──► EvidenceSpec

Note: the knowledge base is a *vocabulary resource*, not part of the parsing
input. The real input is the user question.

The vocabulary is built once when the knowledge base is loaded and reused for
every question (see ``grounding.vocabulary.get_vocabulary``).
"""
from __future__ import annotations

from ..common.contracts import ParsingResult
from ..grounding.builder import EvidenceSpecBuilder
from ..grounding.grounder import IntentGrounder
from ..grounding.vocabulary import get_vocabulary
from .llm_intent import IntentRewriter
from .semantic import SemanticParser


class ParsingPipeline:
    """Compose semantic parsing, KG grounding, and evidence-spec building."""

    def __init__(
        self,
        rewriter: IntentRewriter | None = None,
        *,
        fuzzy_threshold: float = 0.72,
        grounder: IntentGrounder | None = None,
        spec_builder: EvidenceSpecBuilder | None = None,
    ) -> None:
        self.semantic = SemanticParser(rewriter=rewriter)
        self.fuzzy_threshold = fuzzy_threshold
        self.grounder = grounder or IntentGrounder(fuzzy_threshold=fuzzy_threshold)
        self.spec_builder = spec_builder or EvidenceSpecBuilder()

    # ---- entrance of step 1: question -> ParsingResult ----
    def parse(self, question: str, kg) -> ParsingResult:
        # 获取词表，词表在 KG 加载期构建一次，这里直接复用
        vocabulary = get_vocabulary(kg)
        # 规范化问题，并确定所使用的问题改写器并改写问题, 拆实体和关系；规范化只在语义层做一次，原始问题另存以便追溯
        draft, normalized, diagnostics = self.semantic.parse(question, vocabulary)
        # 从词表中 把上一步拆出来的实体和关系，链接到kg中规范的QueryIntent
        intent, ground_diagnostics = self.grounder.ground(question, normalized, draft, kg)
        
        # 把QueryIntent 补上对应问题的 template_id
        spec = self.spec_builder.build(intent)
        metadata = tuple(intent.metadata) + (
            ("rewriter", getattr(self.semantic.rewriter, "name", "unknown")),
        )
        return ParsingResult(
            intent=_with_metadata(intent, metadata),
            evidence_spec=spec,
            diagnostics=tuple(diagnostics) + tuple(ground_diagnostics),
        )


def _with_metadata(intent, metadata):
    from dataclasses import replace

    return replace(intent, metadata=metadata)