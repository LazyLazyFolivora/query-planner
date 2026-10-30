"""① 解析策略：问题文本 -> 证据需求。

  RuleParser    规则实现：实体词典 + 别名匹配（基线）
  ModelParser   模型实现（默认）：词典召回 + 嵌入消歧 + 滑窗模糊 + 嵌入原型算子分类
  HybridParser  PrimeKG 模式匹配 + LLM 补齐（需要注入 LLM 客户端）

HybridParser 的设计约束（重要）：
  * LLM **只在 PrimeKG 直接匹配失败时**被调用，主路径不依赖 LLM；
  * LLM 的产物必须**再过一次 PrimeKG 匹配**才能被采信——避免生成图中不存在的实体
    （若落不到图上，后续"关系访问数"无法计量）；
  * 最终输出三态：linked（命中）/ partial（部分命中）/ unlinked（起点未命中 -> insufficient）。
"""
from __future__ import annotations

import json
import re
from typing import Sequence

import numpy as np

from ..state_construction import find_entities, find_relation_keywords
from ..common.stage_contracts import ParsedQuestion
from ..infrastructure.embedder import DEFAULT_EMBED_MODEL, Embedder, SentenceEmbedder
from ..infrastructure.llm import LLMClient, NullLLM, build_resolve_prompt

OPERATOR_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "chain": ("为什么会导致", "作用机制是什么", "原因是什么", "如何引起"),
    "intersection": ("有什么共同点", "都相关的", "共同的", "同时满足"),
    "counting": ("有多少", "几种", "数量是多少", "哪些"),
    "aggregation": ("总结", "综述", "有哪些类型", "整体情况"),
}

NEGATION_PATTERNS = ("不能", "禁忌", "除外", "排除", "禁用")

# 用于切出"疑似提及"的停用词与标点（占位，可按语料扩充）
_SPLIT = re.compile(
    r"[，。？！、；：,.\?!;:（）()\s]|为什么|怎么办|有哪些|哪些|什么|如何|是否|"
    r"可能导致|导致|引起|引起|增加|减少|的|了|是|会|与|和|或|在|对|把"
)
_TYPE_HINTS = ("药", "病", "症", "基因", "蛋白", "表型", "机制", "综合征", "中毒", "反应")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def _char_ngrams(s: str, n: int = 2) -> set[str]:
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def _dice(a: str, b: str, n: int = 2) -> float:
    A, B = _char_ngrams(a, n), _char_ngrams(b, n)
    if not A or not B:
        return 0.0
    return 2.0 * len(A & B) / (len(A) + len(B))


class RuleParser:
    """基线：纯词典/别名匹配。"""

    name = "rule"

    def parse(self, question: str, kg) -> ParsedQuestion:
        found = find_entities(question, kg)
        return ParsedQuestion(
            seeds=tuple(found[:1]),
            targets=tuple(found[1:]),
            required_relations=tuple(find_relation_keywords(question, kg)),
            excluded_entities=(),
            operator=self._operator(question),
            unlinked_mentions=(),
        )

    @staticmethod
    def _operator(question: str) -> str:
        for op, kws in OPERATOR_PROTOTYPES.items():
            if any(k in question for k in kws):
                return op
        return "chain"


class BioHopRParser:
    """BioHopR 专用解析器：问答模板 + PrimeKG 实体链接。

    该解析器从问题文本中抽取 topic entity，并从问题中的 answer type
    词（disease / drug / effect 等）生成 target_types。它不读取 gold answer。
    """

    name = "biohopr"

    _TARGET_HINTS = (
        ("effect/phenotype", "effect"),
        ("phenotype", "effect"),
        ("gene/protein", "gene"),
        ("protein", "gene"),
        ("disease", "disease"),
        ("drug", "drug"),
        ("gene", "gene"),
    )

    def parse(self, question: str, kg) -> ParsedQuestion:
        found = find_entities(question, kg)
        seeds = tuple(found[:1])
        targets = tuple(found[1:])
        lowered = question.lower()
        target_phrase = lowered.split("associated with", 1)[0]
        match = re.search(r"name\s+(?:all|a|an)\s+(.+?)\s+that\s+(?:are|is)\s+related", lowered)
        if match:
            target_phrase = match.group(1)
        target_types: list[str] = []
        for hint, group in self._TARGET_HINTS:
            if hint in target_phrase and group not in target_types:
                target_types.append(group)
        if not target_types:
            for hint, group in self._TARGET_HINTS:
                if hint in lowered and group not in target_types:
                    target_types.append(group)
        chain: list[tuple[str, ...]] = []
        if "disease" in target_types and "effect" in lowered:
            if "drug" in lowered:
                chain.append(("drug_effect",))
            chain.append(("~disease_phenotype_positive", "~disease_phenotype_negative"))
        elif "effect" in target_types:
            chain.append(("drug_effect",))
        elif "gene" in target_types:
            chain.append(("drug_protein", "protein_protein"))

        return ParsedQuestion(
            seeds=seeds,
            targets=targets,
            required_relations=(),
            excluded_entities=(),
            operator="chain",
            target_types=tuple(target_types),
            required_relation_chain=tuple(chain),
            unlinked_mentions=(),
        )


class ModelParser:
    """模型实现（默认）：词典召回 + 嵌入消歧 + 滑窗模糊补充 + 嵌入原型算子分类。"""

    name = "model"

    def __init__(
        self,
        embedder: Embedder | None = None,
        fuzzy_threshold: float = 0.80,
        relation_threshold: float = 0.75,
    ) -> None:
        self.embedder = embedder or SentenceEmbedder(DEFAULT_EMBED_MODEL)
        self.fuzzy_threshold = fuzzy_threshold
        self.relation_threshold = relation_threshold

    def _encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedder.dim), dtype=np.float32)
        if hasattr(self.embedder, "encode_batch"):
            return self.embedder.encode_batch(texts)
        return np.stack([self.embedder.encode(text) for text in texts])

    def _entity_candidates(self, question: str, kg) -> list[tuple[int, float, str]]:
        out: list[tuple[int, float, str]] = []
        seen: set[str] = set()
        for surface, canonical in sorted(kg.aliases.items(), key=lambda kv: -len(kv[0])):
            pos = question.find(surface)
            if pos < 0 or canonical in seen:
                continue
            out.append((pos, 1.0, canonical))
            seen.add(canonical)

        n = len(question)
        unresolved = [
            (surface, canonical)
            for surface, canonical in kg.aliases.items()
            if canonical not in seen and len(surface) >= 2
        ]
        if not unresolved:
            return out

        surfaces = [surface for surface, _ in unresolved]
        targets = self._encode_batch(surfaces)
        for target, (surface, canonical) in zip(targets, unresolved):
            w = len(surface)
            windows = [question[i : i + w] for i in range(max(n - w + 1, 0))]
            if not windows:
                continue
            window_vecs = self._encode_batch(windows)
            sims = window_vecs @ target / (
                np.linalg.norm(window_vecs, axis=1) * max(float(np.linalg.norm(target)), 1e-12)
                + 1e-12
            )
            best_idx = int(np.argmax(sims))
            best = float(sims[best_idx])
            if best >= self.fuzzy_threshold:
                out.append((best_idx, best, canonical))
                seen.add(canonical)
        return out

    def _relation_candidates(self, question: str, kg) -> list[str]:
        rels = [r for r in kg.relation_names() if r in question]
        if not rels:
            names = kg.relation_names()
            if not names:
                return []
            qv = self.embedder.encode(question)
            rel_vecs = self._encode_batch(names)
            sims = rel_vecs @ qv / (
                np.linalg.norm(rel_vecs, axis=1) * max(float(np.linalg.norm(qv)), 1e-12)
                + 1e-12
            )
            rels = [name for name, score in zip(names, sims) if float(score) >= self.relation_threshold]
        return rels

    def _operator(self, question: str) -> str:
        qv = self.embedder.encode(question)
        flat_prototypes: list[str] = []
        owners: list[str] = []
        for op, prototypes in OPERATOR_PROTOTYPES.items():
            flat_prototypes.extend(prototypes)
            owners.extend([op] * len(prototypes))
        proto_vecs = self._encode_batch(flat_prototypes)
        sims = proto_vecs @ qv / (
            np.linalg.norm(proto_vecs, axis=1) * max(float(np.linalg.norm(qv)), 1e-12)
            + 1e-12
        )
        scores: dict[str, float] = {op: -1.0 for op in OPERATOR_PROTOTYPES}
        for owner, score in zip(owners, sims):
            scores[owner] = max(scores[owner], float(score))
        return max(scores, key=lambda k: scores[k])

    def parse(self, question: str, kg) -> ParsedQuestion:
        hits = self._entity_candidates(question, kg)
        hits.sort(key=lambda x: (x[0] if x[0] >= 0 else 10**6, -x[1]))
        ordered: list[str] = []
        for _, _, canonical in hits:
            if canonical not in ordered:
                ordered.append(canonical)
        excluded = tuple(e for e in ordered
                         if any(f"{neg}{e}" in question for neg in NEGATION_PATTERNS))
        return ParsedQuestion(
            seeds=tuple(ordered[:1]),
            targets=tuple(e for e in ordered[1:] if e not in excluded),
            required_relations=tuple(self._relation_candidates(question, kg)),
            excluded_entities=excluded,
            operator=self._operator(question),
            unlinked_mentions=(),
        )


class HybridParser:
    """PrimeKG 模式匹配 + LLM 补齐。

    流程：
        ① PrimeKG 直接匹配（词典/别名/滑窗模糊）
              命中 -> 直接采用
        ② 对未链接的可疑片段，调用 LLM 归一为规范名
        ③ LLM 产出的规范名**必须再过一次 PrimeKG 匹配**才采信
              命中 -> 记为补齐成功
              未命中 -> 记入 unlinked_mentions
    """

    name = "hybrid"

    def __init__(
        self,
        llm: LLMClient | None = None,
        embedder: Embedder | None = None,
        max_mentions: int = 4,
        fuzzy_for_resolution: float = 0.5,
    ) -> None:
        self.matcher = ModelParser(embedder=embedder)
        self.llm = llm or NullLLM()
        self.max_mentions = max_mentions
        self.fuzzy_for_resolution = fuzzy_for_resolution

    # ---------- ① PrimeKG 直接匹配 ----------
    def _exact_linked(self, question: str, kg) -> list[str]:
        out: list[str] = []
        for surface, canonical in sorted(kg.aliases.items(), key=lambda kv: -len(kv[0])):
            if surface in question and canonical not in out:
                out.append(canonical)
        return out

    # ---------- 可疑片段抽取（启发式，占位） ----------
    def _suspect_mentions(self, question: str, kg, linked: Sequence[str]) -> list[str]:
        linked_names = set(linked)
        covered = "".join(linked_names)
        out: list[str] = []
        for seg in _SPLIT.split(question):
            seg = seg.strip()
            if not (2 <= len(seg) <= 12):
                continue
            if seg in linked_names or seg in covered:
                continue
            # 已能被 PrimeKG 模糊命中，就不必问 LLM
            if self._best_primekg_match(seg, kg, threshold=self.matcher.fuzzy_threshold):
                continue
            out.append(seg)
        # 带类型后缀的片段优先（更像实体）
        out.sort(key=lambda s: (not any(h in s for h in _TYPE_HINTS), -len(s)))
        return out[: self.max_mentions]

    def _best_primekg_match(self, mention: str, kg, threshold: float = 0.5) -> str | None:
        best, best_score = None, 0.0
        for surface, canonical in kg.aliases.items():
            score = max(_dice(mention, surface), 1.0 if surface in mention else 0.0)
            if score > best_score:
                best, best_score = canonical, score
        return best if best_score >= threshold else None

    # ---------- ②③ LLM 归一 + 回图校验 ----------
    def _resolve_with_llm(self, mention: str, question: str, kg) -> str | None:
        # 给 LLM 一个按字面相似度排序的小候选集（bounded reranking）
        scored = sorted(
            ((_dice(mention, s), c) for s, c in kg.aliases.items()), reverse=True
        )[:8]
        candidates = [c for _, c in scored]
        prompt = build_resolve_prompt(mention, question, candidates)
        raw = (self.llm.complete(prompt) or "").strip()
        if not raw:
            return None
        try:
            choice = json.loads(raw).get("choice", "NONE")
        except (json.JSONDecodeError, AttributeError):
            m = re.search(r'"choice"\s*:\s*"([^"]+)"', raw)
            choice = m.group(1) if m else raw.strip().strip('"')
        if not choice or choice.upper() == "NONE":
            return None
        # 回图校验：LLM 给的名字必须能在 PrimeKG 里匹配到
        return self._best_primekg_match(choice, kg, threshold=self.fuzzy_for_resolution)

    def parse(self, question: str, kg) -> ParsedQuestion:
        base = self.matcher.parse(question, kg)
        linked: list[str] = list(base.seeds) + list(base.targets)

        misses = self._suspect_mentions(question, kg, linked)
        unlinked: list[str] = []
        for mention in misses:
            node = self._resolve_with_llm(mention, question, kg)
            if node and node not in linked:
                linked.append(node)
            elif not node:
                unlinked.append(mention)

        seeds = tuple(linked[:1])
        targets = tuple(linked[1:])
        return ParsedQuestion(
            seeds=seeds,
            targets=targets,
            required_relations=base.required_relations,
            excluded_entities=base.excluded_entities,
            operator=base.operator,
            unlinked_mentions=tuple(unlinked),
        )


PARSERS = {
    RuleParser.name: RuleParser,
    BioHopRParser.name: BioHopRParser,
    ModelParser.name: ModelParser,
    HybridParser.name: HybridParser,
}