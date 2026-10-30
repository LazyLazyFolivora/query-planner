"""LLM 客户端接口：只在"PrimeKG 匹配失败"时被调用。

设计约束：
  * **默认关闭**（NullLLM）：主路径不依赖 LLM，保证可复现；
  * 需要时显式注入一个实现（OpenAI 兼容接口 / 本地模型）；
  * LLM 只用于 **在给定候选中选择**，不用于自由生成实体名——
    生成的名字若不在图中，后续关系访问成本无法计量。
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    name: str
    def complete(self, prompt: str) -> str: ...


class NullLLM:
    """默认实现：不可用。触发 LLM 补齐时直接返回 None（等价于"没补齐"）。"""

    name = "null"

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        return ""


class OpenAICompatClient:
    """OpenAI 兼容接口（本地 vLLM / Ollama / 云端皆可）。

    仅用标准库 urllib，避免强制依赖；密钥从环境变量读取，不硬编码。

        client = OpenAICompatClient(
            base_url="http://127.0.0.1:8000/v1",
            model="Qwen/Qwen2.5-7B-Instruct",
            api_key_env="OPENAI_API_KEY",
        )
    """

    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = timeout
        self.temperature = temperature

    def complete(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]


RESOLVE_PROMPT = """你是生物医学实体链接助手。请判断给定的【提及】最接近下列哪个候选实体。

【提及】{mention}
【上下文】{context}
【候选实体】
{candidates}

要求：
1. 只能从上面的候选中选一个，或回答 NONE；
2. 不要创造候选列表之外的名称；
3. 只输出一行 JSON：{{"choice": "候选实体名或NONE"}}
"""


def build_resolve_prompt(mention: str, context: str, candidates: Sequence[str]) -> str:
    listing = "\n".join(f"- {c}" for c in candidates)
    return RESOLVE_PROMPT.format(mention=mention, context=context, candidates=listing)