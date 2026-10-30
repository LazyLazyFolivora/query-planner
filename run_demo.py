#!/usr/bin/env python3
"""D 课题 demo：预算约束查询规划的循环骨架（全流程策略化）。

每一步都可以独立指定实现，**默认全部走模型**：

    ① --parser    pipeline | model | rule | biohopr | hybrid
    ② --encoder   model | feature
    ③ --stop      model | flat | rule | naive_target
    ④ --selector  model | linear | heuristic
    ⑤ --executor  toy_kg
    ⑥ --ledger    budget
    ⑦ --updater   standard

**① 解析默认走 `pipeline`：LLM 改写 + KG 链接**（新版解析层的第一阶段）。
LLM 配置从项目根目录 .env 读（OPENAI_API_KEY / QP_LLM_BASE_URL / QP_LLM_MODEL），
见 .env.example。没配成时不会崩，而是逐条退化为规则解析，并在诊断里标出来——
所以"是否真的用上了 LLM"这件事，要看输出里的 rewriter 字段，不是看默认参数。

常用：
    python run_demo.py --quiet                       # 默认：全模型（含 LLM 解析）
    python run_demo.py --preset rule                 # 一键切回全规则（基线，完全离线）
    python run_demo.py --parser rule                 # 只把① 换成规则解析，其余保持默认
    python run_demo.py --stop rule,model             # 固定其他，只对比终止规则
    python run_demo.py --selector heuristic,model    # 固定其他，只对比选择器

依赖：Python >= 3.10 + numpy。LLM 与嵌入模型都是可选依赖，缺了会自动降级。
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "本项目需要 Python >= 3.10，当前解释器为 "
        + sys.version.split()[0]
        + " ("
        + sys.executable
        + ")。请切换到 3.10+ 的解释器，例如 C:\\Python312\\python.exe run_demo.py"
    )

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.common.llm_client import client_from_env
from src.evaluator import evaluate
from src.kg import ToyKG
from src.pipeline import run_question
from src.schema import Question, RequirementGroup, Triple
from src.pipeline import DEFAULTS, REGISTRY, STEPS, build_pipeline
from src.infrastructure.embedder import DEFAULT_EMBED_MODEL, SentenceEmbedder

RULE_PRESET = {
    "parser": "rule",
    "encoder": "feature",
    "stop": "rule",
    "selector": "heuristic",
}


def load_questions(path: Path) -> list[Question]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: list[Question] = []
    for q in payload["questions"]:
        groups = []
        for g in q["requirement_groups"]:
            alts = tuple(Triple(a["head"], a["relation"], a["tail"]) for a in g["any_of"])
            groups.append(RequirementGroup(g["gid"], alts, float(g.get("weight", 1.0))))
        out.append(Question(
            qid=q["qid"], question=q["question"], qtype=q["qtype"], budget=dict(q["budget"]),
            requirement_groups=groups, max_steps=int(q.get("max_steps", 12)),
            sufficiency_threshold=float(q.get("sufficiency_threshold", 0.99)), note=q.get("note", ""),
        ))
    return out


def print_trace(q: Question, pipeline, trace, verbose: bool = True) -> None:
    print("=" * 104)
    print(f"[{q.qid}] {q.question}")
    print(f"  类型={q.qtype}")
    print("  流水线: " + "  ".join(f"{k}={pipeline.names[k]}" for k in STEPS))
    print(f"  预算 ra={q.budget['relation_access']} calls={q.budget['tool_calls']} "
          f"tok={q.budget['tokens']}  |  {q.note}")
    if not verbose:
        return
    print("-" * 104)
    print(f"  {'步':>2}  {'充分性':>6}  {'决策':<13}{'动作':<44}{'成本':<19}{'余额ra':>6}  新增")
    for s in trace.steps:
        action = s.action.describe() if s.action else "(无)"
        if len(action) > 42:
            action = action[:39] + "..."
        cost = s.cost.text() if s.action else "-"
        print(f"  {s.index:>2}  {s.sufficiency:>6.2f}  {s.decision.value:<13}{action:<44}"
              f"{cost:<19}{s.remaining['relation_access']:>6}  {len(s.new_evidence)}")
        for e in s.new_evidence:
            txt = e.label()
            if len(txt) > 86:
                txt = txt[:83] + "..."
            print(f"        + {txt}")
    print("-" * 104)
    print(f"  结束: {trace.final_decision.value} ({trace.final_reason}) | 累计 {trace.spent.text()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=str(ROOT / "data" / "toy_kg.json"))
    ap.add_argument("--questions", default=str(ROOT / "data" / "questions.json"))
    ap.add_argument("--preset", choices=["model", "rule"], default="model",
                    help="model=全模型（默认）；rule=全规则基线")
    for step in STEPS:
        opts = ", ".join(sorted(REGISTRY[step]))
        ap.add_argument(f"--{step}", dest=step, default=None, help=f"可选: {opts}")
    ap.add_argument("--stop-rules", dest="stop_rules", default=None,
                    help="③ 终止规则列表（做消融，如 rule,model）")
    ap.add_argument("--selectors", dest="selectors", default=None,
                    help="④ 选择器列表（做消融，如 heuristic,model）")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    base = dict(RULE_PRESET) if args.preset == "rule" else {}
    for step in STEPS:
        v = getattr(args, step)
        if v:
            base[step] = v

    kg = ToyKG.from_json(args.kg)
    questions = load_questions(Path(args.questions))

    parser_name = base.get("parser", DEFAULTS["parser"])
    encoder_name = base.get("encoder", DEFAULTS["encoder"])

    # ---- 依赖预检与降级 ----------------------------------------
    # 默认配置的部分步骤依赖 torch / sentence-transformers。缺份时整条链路无法装配，
    # 而错误会在装配阶段就抛出，用户看到的是一堆 traceback，很难猜到是“默认值换了”导致的。
    # 这里逐步降级到无依赖实现，并把降级说到重要地方，不把自己转给后面的报错。
    downgrades: list[str] = []

    shared_embedder: SentenceEmbedder | None = None
    if encoder_name == "model" or parser_name in {"model", "hybrid"}:
        try:
            shared_embedder: SentenceEmbedder | None = SentenceEmbedder(DEFAULT_EMBED_MODEL)
        except RuntimeError:
            shared_embedder = None
            encoder_name = "feature"
            base["encoder"] = encoder_name
            downgrades.append(
                "② 编码 model->feature（缺 sentence-transformers）"
            )

    # ③、④ 的模型实现都需要 torch；在确认之前就降级，避免装配时接回显式的堆栈
    if (base.get("stop", DEFAULTS["stop"]) == "model"
            or base.get("selector", DEFAULTS["selector"]) == "model"):
        try:
            import torch  # noqa: F401
        except ImportError:
            if base.get("stop", DEFAULTS["stop"]) == "model":
                base["stop"] = "rule"
                downgrades.append("③ 终止 model->rule（缺 torch）")
            if base.get("selector", DEFAULTS["selector"]) == "model":
                base["selector"] = "heuristic"
                downgrades.append("④ 选择 model->heuristic（缺 torch）")

    if downgrades:
        print("[提示] 部分模型组件缺依赖，已降级到无依赖实现：")
        for item in downgrades:
            print(f"       - {item}")
        print("       注：降级后的结果不代表真实模型能力。")

    # 只有① 解析真的要用 LLM 时才建客户端，避免无关配置报错
    shared_llm = client_from_env() if parser_name == "pipeline" else None
    if shared_llm is not None and not shared_llm.is_available():
        # 不抛异常：未配置 LLM 是合法状态，退化成规则解析并让诊断说话
        print("[提示] 未检测到可用的 LLM 配置（.env / 环境变量），① 解析将退化为规则实现。")
        print("       可用项：OPENAI_API_KEY、QP_LLM_BASE_URL、QP_LLM_MODEL")
        shared_llm = None

    # æ­¥éª¤åå¿é¡»å¨éçº§ä¹åå®ï¼å¦åé»è®¤å¼ä¼æå·²ç»éçº§è¿çå®ç°åæåæ¥
    stop_list = [s.strip() for s in (args.stop_rules or base.get("stop") or DEFAULTS["stop"]).split(",")]
    sel_list = [s.strip() for s in (args.selectors or base.get("selector") or DEFAULTS["selector"]).split(",")]

    pipelines = []
    for sel in sel_list:
        for stop in stop_list:
            overrides = dict(base)
            overrides["selector"] = sel
            overrides["stop"] = stop
            overrides["encoder"] = base.get("encoder", encoder_name)
            try:
                pipeline = build_pipeline(kg, llm=shared_llm, embedder=shared_embedder, **overrides)
            except KeyError as exc:
                print(f"[参数错误] {exc}")
                return 2
            pipelines.append(pipeline)

    rows = []
    for q in questions:
        for pipeline in pipelines:
            trace = run_question(q, kg, pipeline)
            result = evaluate(q, trace)
            print_trace(q, pipeline, trace, verbose=not args.quiet)
            if not args.quiet:
                print(f"  gold: 充分性={result['gold_sufficiency']:.2f} "
                      f"命中={len(result['groups_hit'])}/{len(q.requirement_groups)} "
                      f"未命中={result['groups_missed']} | 判定={result['verdict']}")
            rows.append((q, pipeline, trace, result))
        print()

    print("=" * 104)
    print("汇总表")
    print("=" * 104)
    print(f"{'问题':<26}{'选择器':<11}{'终止':<13}{'解析':<8}{'编码':<8}{'决策':<13}"
          f"{'停止原因':<20}{'ra':>4}{'调用':>5}{'步':>4}{'gold':>7}  判定")
    print("-" * 104)
    for q, pipeline, trace, result in rows:
        print(f"{q.qid[:25]:<26}{pipeline.names['selector']:<11}{pipeline.names['stop']:<13}"
              f"{pipeline.names['parser']:<8}{pipeline.names['encoder']:<8}"
              f"{trace.final_decision.value:<13}{trace.final_reason:<20}"
              f"{trace.spent.relation_access:>4}{trace.spent.tool_calls:>5}"
              f"{len(trace.steps):>4}{result['gold_sufficiency']:>7.2f}  {result['verdict']}")
    print("-" * 104)
    wrong = sum(1 for _, _, _, r in rows if r["wrong_complete"])
    false_ins = sum(1 for _, _, _, r in rows if r["false_insufficient"])
    print(f"错误完成(证据不足却报完成) {wrong} 次 | "
          f"错误不足(证据已足却报不足) {false_ins} 次 | 共 {len(rows)} 次运行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())