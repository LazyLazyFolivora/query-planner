"""裸文本扫图谱：字符串匹配的实体/关系抽取（第一阶段旧实现的共用底座）。

严格字符串匹配，不做语义理解，也不含 LLM。
第一阶段新的 ParsingPipeline 不走这里（它走 LLM 改写 + 词表链接），
  parsing/legacy.py 里的规则/模型解析器仍在用这两个函数，
所以保留在本包里作为共用工具。

  find_entities()           文本中出现的图谱实体（长名优先，保持出现顺序）
  find_relation_keywords()  文本中出现的图谱关系名
  parse_question()         上面两者的组合：seeds / targets / 关系关键词
"""
from __future__ import annotations

from .models import KGView


def find_entities(text: str, kg: KGView) -> list[str]:
    """返回文本中出现的图谱实体（长名优先，保持出现顺序）。"""
    hits: list[tuple[int, int, str]] = []
    for surface, canonical in kg.aliases.items():
        idx = text.find(surface)
        if idx >= 0:
            hits.append((idx, -len(surface), canonical))
    hits.sort()
    out: list[str] = []
    seen: set[str] = set()
    for _, _, canonical in hits:
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out


def find_relation_keywords(text: str, kg: KGView) -> list[str]:
    """返回文本中出现的图谱关系名（图谱关系名通常是子串）。"""
    return [r for r in kg.relation_names() if r in text]


def parse_question(
    question: str, kg: KGView,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """裸解析：第一个命中实体当起点，其余当终点，外加关系关键词。"""
    found = find_entities(question, kg)
    seeds = tuple(found[:1])
    targets = tuple(found[1:])
    return seeds, targets, tuple(find_relation_keywords(question, kg))
