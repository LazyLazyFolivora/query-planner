from __future__ import annotations

import json
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kg import ToyKG
from src.parsing.llm_intent import LLMIntentRewriter
from src.parsing.pipeline import ParsingPipeline


class NormalizationContractTest(unittest.TestCase):
    """Step 1 must normalize the user question exactly once."""

    @classmethod
    def setUpClass(cls):
        cls.kg = ToyKG.from_json(ROOT / "data" / "toy_kg.json")

    def test_question_is_normalized_once_in_semantic_layer(self):
        from src.parsing import semantic as semantic_module

        calls = []
        original = semantic_module.normalize_text

        def counting(text):
            calls.append(text)
            return original(text)

        pipeline = ParsingPipeline()
        with unittest.mock.patch.object(semantic_module, "normalize_text", counting):
            pipeline.parse("　 二甲双胍为什么可能导致乳酸性酸中毒？ ", self.kg)

        self.assertEqual(len(calls), 1)

    def test_rewriter_receives_normalized_text(self):
        captured = {}

        payload = {
            "template_id": "mechanism_path",
            "canonical_question": "求二甲双胍到乳酸性酸中毒的因果机制路径",
            "entities": [
                {"role": "start", "mention": "二甲双胍"},
                {"role": "target", "mention": "乳酸性酸中毒"},
            ],
            "answer": {"variable": None, "type_hint": None, "cardinality": "single"},
            "relation_constraints": [],
        }

        class CapturingRewriter:
            name = "capture"

            def rewrite(self, question):
                captured["question"] = question
                return LLMIntentRewriter(FakeLLM({"二甲双胍": payload})).rewrite(question)

        pipeline = ParsingPipeline(rewriter=CapturingRewriter())
        pipeline.parse("　 二甲双胍为什么可能导致乳酸性酸中毒？ ", self.kg)

        self.assertEqual(captured["question"], "二甲双胍为什么可能导致乳酸性酸中毒?")

    def test_raw_and_canonical_question_are_distinct(self):
        raw = "　 二甲双胍为什么可能导致乳酸性酸中毒？ "
        result = ParsingPipeline().parse(raw, self.kg)

        self.assertEqual(result.intent.raw_question, raw)
        self.assertEqual(result.intent.canonical_question, "二甲双胍为什么可能导致乳酸性酸中毒?")

    def test_downstream_layers_do_not_normalize_the_question(self):
        """Grounding must reuse the normalized text instead of recomputing it."""
        from src.common import normalization as normalization_module

        original = normalization_module.normalize_text
        seen = []

        def spy(text):
            seen.append(text)
            return original(text)

        with unittest.mock.patch.object(normalization_module, "normalize_text", spy):
            result = ParsingPipeline().parse("二甲双胍为什么可能导致乳酸性酸中毒？", self.kg)

        # nothing else may receive the full question for re-normalization
        self.assertNotIn("二甲双胍为什么可能导致乳酸性酸中毒？", seen)
        self.assertEqual(result.intent.parse_status, "linked")


class FakeLLM:
    def __init__(self, payloads):
        self.payloads = payloads

    def complete(self, prompt: str) -> str:
        for question, payload in self.payloads.items():
            if question in prompt:
                return json.dumps(payload, ensure_ascii=False)
        return "{}"


if __name__ == "__main__":
    unittest.main()