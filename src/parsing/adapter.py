"Compatibility adapter exposing ParsingPipeline through the legacy parser API."
from __future__ import annotations


from .llm_intent import LLMIntentRewriter, NullIntentRewriter
from .pipeline import ParsingPipeline


class PipelineParserAdapter:
    """Legacy adapter for the current Pipeline registry.

    New code should call ``parse_intent`` and consume ``ParsingResult``.
    ``parse`` preserves compatibility with the existing runner.
    """

    name = "pipeline"

    def __init__(self, llm=None, embedder=None, fuzzy_threshold: float = 0.72) -> None:
        self.embedder = embedder
        # 传入的客户端若声明 is_available()，以它的判断为准；
        # 不可用时保持 NullIntentRewfitef，让诊断记为"没配置"而不是"调用失败"。
        if llm is not None and hasattr(llm, "complete"):
            probe = getattr(llm, "is_available", None)
            usable = bool(probe()) if callable(probe) else True
        else:
            usable = False
        rewriter = LLMIntentRewriter(llm) if usable else NullIntentRewriter()
        self.llm = llm if usable else None
        self.pipeline = ParsingPipeline(rewriter=rewriter, fuzzy_threshold=fuzzy_threshold)
        self.last_result = None

    def parse_intent(self, question: str, kg):
        self.last_result = self.pipeline.parse(question, kg)
        return self.last_result

    def parse(self, question: str, kg):
        from ..strategies.base import ParsedQuestion
        result = self.parse_intent(question, kg)
        intent = result.intent
        operator = {
            "mechanism": "chain",
            "chain": "chain",
            "intersection": "intersection",
            "counting": "counting",
            "aggregation": "aggregation",
        }.get(intent.reasoning_mode, "chain")
        return ParsedQuestion(
            seeds=intent.start_entities,
            targets=intent.target_entities,
            required_relations=tuple(c.relation for c in intent.relation_constraints),
            excluded_entities=intent.exclusions,
            operator=operator,
            target_types=intent.target_types,
            required_relation_chain=intent.path_template,
            unlinked_mentions=intent.unlinked_mentions,
            query_intent=intent,
            evidence_spec=result.evidence_spec,
            parse_status=intent.parse_status,
        )
