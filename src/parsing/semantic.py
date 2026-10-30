"""Semantic parsing: user question -> ungrounded IntentDraft.

This layer owns text normalization. ``parse`` is the single place where the
raw question is normalized; every downstream stage receives the already
normalized text and must not normalize again.
"""
from __future__ import annotations

from ..common.normalization import normalize_text
from .lexicon import Lexicon
from .llm_intent import LLMIntentRewriter, NullIntentRewriter
from .rule_intent import RuleIntentRewriter
from .schema import IntentDraft


def _is_available(rewriter) -> bool:
    """改写器是否真的具备调用能力。

    不能只判断"不是 NullIntentRewriter"——传入一个未配置的 LLM 客户端时，
    它照样不是 Null 类型，但调用必然失败。这里区分两种情况：

        没配置 -> is_available() 为 False -> 记llm_rewriter_not_configured
        配了但挂了         -> 调用抛异常     -> 记 llm_rewriter_error:...

    前者是能力上限，后者是基础设施问题，消融实验里结论完全不同。
    """
    check = getattr(rewriter, "is_available", None)
    if callable(check):
        return bool(check())
    return True


class SemanticParser:
    """Rewrite a natural-language question into an intent draft."""

    def __init__(self, rewriter: LLMIntentRewriter | NullIntentRewriter | None = None) -> None:
        self.rewriter = rewriter or NullIntentRewriter()

    def parse(
        self,
        question: str,
        lexicon: Lexicon | None = None,
    ) -> tuple[IntentDraft, str, tuple[str, ...]]:
        """Normalize once, then rewrite.

        Returns ``(draft, normalized_question, diagnostics)``.
        """
        normalized = normalize_text(question)
        diagnostics: list[str] = []
        draft = self._rewrite(normalized, lexicon, diagnostics)
        return draft, normalized, tuple(diagnostics)

    def _rewrite(
        self,
        question: str,
        lexicon: Lexicon | None,
        diagnostics: list[str],
    ) -> IntentDraft:
        """决定由谁来把问题改写成 IntentDraft，并保证任何情况下都能返回草稿。

        本函数**不含任何解析规则**，只做三件事：
            选实现（LLM 还是规则）、失败兜底、记录诊断。

        第一段：优先走 LLM 改写器
            配置了 LLM 且调用成功 -> 直接返回它的草稿。
            LLM 是网络依赖，会超时、限流、返回坏 JSON，
            所以整个调用被 try 包住，任何异常都在这里被吞掉并降级，
            保证解析步骤永远不会因为 LLM 挂掉而失败。

        第二段：降级到确定性规则
            LLM 不可用、返回 None 或抛异常时，用 RuleIntentRewriter 兜底。
            规则版本可复现、无外部依赖，但只能处理表达较规范的问题。
            连规则都解析不出来（空文本）时，造一个最小草稿，
            保证下游拿到的一定是 IntentDraft，不会是 None。

        诊断信息按失败原因分类，是为了让评测能区分不同的问题：
            llm_rewriter_not_configured  本来就没配 LLM，属于能力上限问题
            llm_rewriter_returned_none   LLM 答了但没给草稿，属于提示词问题
            llm_rewriter_error:...       LLM 调用失败，属于基础设施问题
            rule_fallback_used           走了规则兜底
        这四种在消融实验里是完全不同的失败原因，不能混为一谈。

        diagnostics 以可变 list 传入、由 parse() 冻结成 tuple，
        这样本函数只管追加，不用管返回。
        """
        if _is_available(self.rewriter):
            try:
                draft = self.rewriter.rewrite(question)
                if draft is not None:
                    return draft
                diagnostics.append("llm_rewriter_returned_none")
            except Exception as exc:
                diagnostics.append(f"llm_rewriter_error:{type(exc).__name__}:{exc}")
        else:
            diagnostics.append("llm_rewriter_not_configured")

        draft = RuleIntentRewriter(lexicon).rewrite(question)
        if draft is None:
            draft = IntentDraft(template_id="chain_path", canonical_question=question)
        diagnostics.append("rule_fallback_used")
        return draft
