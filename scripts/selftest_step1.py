# -*- coding: utf-8 -*-
"""第一阶段（ParsingPipeline）自测脚本。

它验证的是"接口契约"，不是"某个 LLM 的具体输出"。这一点很重要：
LLM 的实体抽取会漂移，硬编码 "LLM 必须抽出某几个字" 的断言会天天假失败。
所以这里的断言全部针对**结构性质**，例如：

    · 三段是否都跑了（改写 -> grounding -> spec）
    · 诊断里是否出现回退（说明 LLM 没被真正用上）
    · 规格里的check 数量是否够不够支撑这个模板
    · EvidenceSpec 里的约束是否和 QueryIntent 一致

用法：
    python scripts/selftest_step1.py            # 离线：规则解析 + 单元测试
    python scripts/selftest_step1.py --llm       # 联网：真实 LLM
    python scripts/selftest_step1.py --llm -v# 联网 + 逐条明细
退出码 0 = 全通过；1 = 有失败。
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.llm_client import client_from_env
from src.grounding.vocabulary import get_vocabulary
from src.kg import ToyKG
from src.parsing.adapter import PipelineParserAdapter
from src.parsing.pipeline import ParsingPipeline

KG_PATH = ROOT / "data" / "toy_kg.json"

# 期望覆盖的模板。挑这几个是因为 toy_kg 的三元组真实支持它们，
# 这样"解析对了"才有意义——图谱里压根没有的东西，解析对了也回答不了。
QUESTIONS = [
    ("q1", "二甲双胍为什么可能导致乳酸性酸中毒？", "mechanism_path"),
    ("q2", "华法林与哪些药物相互作用会增加出血风险？", "interaction_set"),
    ("q4", "布洛芬的完整作用机制、相互作用和出血风险是什么？", "aggregation"),
]


class Check:
    """极简断言收集器：把"跑挂了"和"结果不对"分成两类。"""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose
        self.passed = 0
        self.failures: list[str] = []

    def ok(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            if self.verbose:
                print(f"    [OK]   {label}")
        else:
            self.failures.append(f"{label}  |  {detail}")
            print(f"    [FAIL] {label}  |  {detail}")

    def eq(self, label: str, actual, expected) -> None:
        self.ok(label, actual == expected, f"actual={actual!r} expected={expected!r}")


# ---------------------------------------------------------------- 离线部分

def offline_checks(c: Check) -> None:
    """不依赖 LLM 的部分：契约、降级路径、单元测试。"""
    kg = ToyKG.from_json(KG_PATH)

    print("\n[1] \u7eaf\u79bb\u7ebf\uff1a\u89c4\u5219\u89e3\u6790\u4f5c\u4e3a\u964d\u7ea7\u8def\u5f84")
    pipe = ParsingPipeline()  # \u4e0d\u6ce8\u5165 rewriter -> Null -> Rule
    for tag, question, _ in QUESTIONS:
        result = pipe.parse(question, kg)
        i = result.intent
        diags = result.diagnostics
        # \u89c4\u5219\u8def\u5f84\u5fc5\u987b\u81ea\u8bc1\u201c\u6211\u662f\u964d\u7ea7\u8def\u5f84\u201d\uff0c\u5426\u5219\u8fdb\u683c\u4e0b\u6765\u4f1a\u88ab\u8bef\u8bfb\u4e3a\u6b63\u5e38
        c.ok(f"{tag} \u964d\u7ea7\u8def\u5f84\u5e26\u6807\u8bb0", "rule_fallback_used" in diags, str(diags))
        c.ok(f"{tag} \u964d\u7ea7\u8def\u5f84\u4ecd\u6709 start entity", bool(i.start_entities), str(i.start_entities))
        c.ok(f"{tag} \u964d\u7ea7\u8def\u5f84\u4ecd\u6709 spec", bool(result.evidence_spec.checks), "")

    print("\n[2] \u7eaf\u79bb\u7ebf\uff1a\u8bca\u65ad\u80fd\u533a\u5206\u5931\u8d25\u539f\u56e0")
    # \u4e0d\u914d LLM -> \u5e94\u8be5\u662f not_configured\uff0c\u800c\u4e0d\u662f error
    c.ok("\u672a\u914d\u7f6e\u65f6\u8bb0 not_configured",
         "llm_rewriter_not_configured" in pipe.parse("x\u4ec0\u4e48", kg).diagnostics, "")
    # \u914d\u4e86\u4f46\u8c03\u7528\u5931\u8d25 -> \u5e94\u8be5\u662f error
    class Boom:
        name = "boom"
        def complete(self, prompt): raise RuntimeError("net down")
        def is_available(self): return True

    failing = ParsingPipeline(rewriter=_Adapter(Boom()))
    diags = failing.parse("\u4e8c\u7532\u53cc\u80cd\u4e3f\u4e3a\u4ec0\u4e48", kg).diagnostics
    c.ok("\u8c03\u7528\u5931\u8d25\u65f6\u8bb0 error \u4e0d\u662f not_configured",
         any(d.startswith("llm_rewriter_error") for d in diags) and
         "llm_rewriter_not_configured" not in diags, str(diags))

    print("\n[3] \u7eaf\u79bb\u7ebf\uff1a\u8bcd\u8868\u53ea\u6784\u5efa\u4e00\u6b21")
    v1 = get_vocabulary(kg)
    v2 = get_vocabulary(kg)
    c.eq("\u540c\u4e00 KG \u8bcd\u8868\u91cd\u590d\u53d6\u662f\u540c\u4e00\u5bf9\u8c61", v1 is v2, True)

    print("\n[4] \u5355\u5143\u6d4b\u8bd5")
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=0, stream=open("NUL", "w")).run(suite)
    c.ok(f"\u5355\u5143\u6d4b\u8bd5 {result.testsRun} \u4e2a", result.wasSuccessful(),
         "; ".join(str(x) for x in (result.failures + result.errors)[:2]))


class _Adapter:
    """\u5c06\u4efb\u610f\u5ba2\u6237\u7aef\u5305\u6210 SemanticParser \u80fd\u7528\u7684\u6539\u5199\u5668\u3002"""
    name = "adapter"

    def __init__(self, client) -> None:
        from src.parsing.llm_intent import LLMIntentRewriter
        self._r = LLMIntentRewriter(client)

    def rewrite(self, question):
        return self._r.rewrite(question)

    def is_available(self):
        return True


# ---------------------------------------------------------------- LLM 部分

def llm_checks(c: Check) -> None:
    kg = ToyKG.from_json(KG_PATH)
    client = client_from_env()
    c.ok("LLM \u914d\u7f6e\u53ef\u7528", client.is_available(),
         "\u8bf7\u914d\u7f6e OPENAI_API_KEY / QP_LLM_BASE_URL / QP_LLM_MODEL")

    if not client.is_available():
        return

    adapter = PipelineParserAdapter(llm=client)
    print(f"    (base_url={client.base_url} model={client.model} wire={client.wire_api})")

    for tag, question, expected_tpl in QUESTIONS:
        print(f"\n    {tag}: {question}")
        try:
            result = adapter.parse_intent(question, kg)
        except Exception as exc:
            c.ok(f"{tag} \u4e0d\u629b\u5f02\u5e38", False, f"{type(exc).__name__}: {exc}")
            continue
        c.ok(f"{tag} \u4e0d\u629b\u5f02\u5e38", True)

        i = result.intent
        diags = [d for d in result.diagnostics if not d.startswith("raw_draft")]
        # \u5173\u952e\uff1a\u786e\u8ba4\u771f\u7684\u7528\u4e0a\u4e86 LLM\uff0c\u4e0d\u662f\u89c4\u5219\u51c6\u9f50\u7406\u5b8c\u6210\u4e86\u4f1c\u7684
        c.ok(f"{tag} \u771f\u7684\u8d70 LLM\uff08\u975e\u964d\u7ea7\uff09",
             dict(i.metadata).get("rewriter") == "llm" and "rule_fallback_used" not in diags,
             f"rewriter={dict(i.metadata).get('rewriter')} diags={diags}")

        # \u6a21\u677f\u5206\u7c7b\uff1a\u5141\u8bb8\u5c0f\u5e45\u504f\u5dee\uff0c\u4f46\u4e0d\u5141\u8bb8\u6389\u5230\u5b8c\u5168\u4e0d\u540c\u7684\u5bb9\u5668
        c.ok(f"{tag} template \u5c5e\u4e8e\u9884\u671f\u5bb9\u5668",
             i.template_id in {expected_tpl, "chain_path"},
             f"actual={i.template_id} expected~{expected_tpl}")
        c.ok(f"{tag} start entity \u5df2\u94fe\u63a5", bool(i.start_entities), str(i.start_entities))
        c.ok(f"{tag} \u6709\u5173\u7cfb\u7cfb\u7edf", len(i.relation_constraints) >= 1, str(len(i.relation_constraints)))
        c.ok(f"{tag} spec \u5b8c\u6574", bool(result.evidence_spec.checks), "")

        if c.verbose:
            print(f"      template={i.template_id} status={i.parse_status}")
            print(f"      entities={i.start_entities} -> {i.target_entities}")
            print(f"      relations={[x.relation for x in i.relation_constraints]}")
            print(f"      checks={[x.check_id for x in result.evidence_spec.checks]}")
            print(f"      diags={diags}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="\u540c\u65f6\u8dd1\u771f\u5b9e LLM \u68c0\u67e5")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    c = Check(args.verbose or not args.llm)
    offline_checks(c)
    if args.llm:
        print("\n[5] \u8054\u7f51\uff1a\u771f\u5b9e LLM")
        llm_checks(c)

    print("\n" + "=" * 70)
    if c.failures:
        print(f"\u5931\u8d25 {len(c.failures)} \u4e2a\uff08\u901a\u8fc7 {c.passed} \u4e2a\uff09\uff1a")
        for f in c.failures:
            print(f"  - {f}")
        return 1
    print(f"\u5168\u90e8\u901a\u8fc7\uff1a{c.passed} \u4e2a\u68c0\u67e5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
