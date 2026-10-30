from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.kg import ToyKG
from src.parsing.adapter import PipelineParserAdapter
from src.parsing.llm_intent import LLMIntentRewriter
from src.parsing.pipeline import ParsingPipeline


class FakeLLM:
    def __init__(self, payloads):
        self.payloads = payloads

    def complete(self, prompt: str) -> str:
        for question, payload in self.payloads.items():
            if question in prompt:
                return json.dumps(payload, ensure_ascii=False)
        return "{}"


class ParsingPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kg = ToyKG.from_json(ROOT / "data" / "toy_kg.json")

    def test_mechanism_question(self):
        payload = {
            "template_id": "mechanism_path",
            "canonical_question": "求二甲双胍到乳酸性酸中毒的因果机制路径",
            "entities": [
                {"role": "start", "mention": "二甲双胍"},
                {"role": "target", "mention": "乳酸性酸中毒"},
            ],
            "answer": {"variable": None, "type_hint": None, "cardinality": "single"},
            "relation_constraints": [
                {
                    "source": "二甲双胍",
                    "predicate_hint": "为什么导致",
                    "target": "乳酸性酸中毒",
                    "direction": "forward",
                    "strength": "required",
                }
            ],
        }
        pipeline = ParsingPipeline(rewriter=LLMIntentRewriter(FakeLLM({"二甲双胍": payload})))
        result = pipeline.parse("二甲双胍为什么可能导致乳酸性酸中毒？", self.kg)

        self.assertEqual(result.intent.template_id, "mechanism_path")
        self.assertEqual(result.intent.reasoning_mode, "mechanism")
        self.assertEqual(result.intent.start_entities, ("二甲双胍",))
        self.assertEqual(result.intent.target_entities, ("乳酸性酸中毒",))
        self.assertEqual(result.intent.relation_constraints[0].relation, "导致")
        self.assertIn("causal_chain", {c.check_id for c in result.evidence_spec.checks})

    def test_interaction_set_question(self):
        payload = {
            "template_id": "interaction_set",
            "canonical_question": "求与华法林相互作用并增加出血风险的药物集合",
            "entities": [
                {"role": "start", "mention": "华法林"},
                {"role": "target", "mention": "出血风险增加"},
            ],
            "answer": {"variable": "?drug", "type_hint": "drug", "cardinality": "set"},
            "relation_constraints": [
                {
                    "source": "华法林",
                    "predicate_hint": "相互作用",
                    "target": "?drug",
                    "direction": "forward",
                    "strength": "required",
                },
                {
                    "source": "?drug",
                    "predicate_hint": "增加出血风险",
                    "target": "出血风险增加",
                    "direction": "forward",
                    "strength": "required",
                },
            ],
        }
        pipeline = ParsingPipeline(rewriter=LLMIntentRewriter(FakeLLM({"华法林": payload})))
        result = pipeline.parse("华法林与哪些药物相互作用会增加出血风险？", self.kg)

        self.assertEqual(result.intent.template_id, "interaction_set")
        self.assertEqual(result.intent.answer_mode, "set")
        self.assertEqual(result.intent.answer_slots[0].variable, "?drug")
        self.assertEqual(result.intent.start_entities, ("华法林",))
        self.assertEqual(result.intent.target_entities, ("出血风险增加",))
        self.assertEqual(len(result.intent.relation_constraints), 2)
        self.assertEqual(result.intent.relation_constraints[0].relation, "相互作用")

    def test_unlinked_question(self):
        pipeline = ParsingPipeline()
        result = pipeline.parse("完全不存在的药物会不会引起不存在的疾病？", self.kg)
        self.assertEqual(result.intent.parse_status, "unlinked")
        self.assertEqual(result.intent.start_entities, ())

    def test_legacy_adapter(self):
        adapter = PipelineParserAdapter()
        parsed = adapter.parse("二甲双胍为什么可能导致乳酸性酸中毒？", self.kg)
        self.assertEqual(parsed.seeds, ("二甲双胍",))
        self.assertEqual(parsed.targets, ("乳酸性酸中毒",))
        self.assertTrue(adapter.last_result is not None)


if __name__ == "__main__":
    unittest.main()
