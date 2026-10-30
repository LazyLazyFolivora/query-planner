"""LLM 客户端：对外只暴露 complete(prompt) -> str。

解析层只依赖这一个方法，因此换供应商不影响业务代码。

    client = OpenAICompatClient(
        base_url="https://api.aifang.online",
        model="deepseek-v4-flash",
        wire_api="responses",
        reasoning_effort="high",
    )
    text = client.complete("...")

密钥不写死在源码里，从环境变量读取（本项目约定 OPENAI_API_KEY）。
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable

from .dotenv_loader import load_env_file


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(self, prompt: str) -> str: ...

    def is_available(self) -> bool:
        """是否具备真实调用能力。用于区分"没配 LLM"和"LLM 调失败了"。"""
        ...


def extract_text(body: Any) -> str:
    """从响应体里取模型输出文本，兼容两种返回格式。

    实测 api.aifang.online 的 /responses 返回的是 chat.completions 结构
    （顶层 choices），而不是官方 Responses API 的 output 数组，所以两种都要认。
    """
    if not isinstance(body, dict):
        return ""

    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            return str(message.get("content") or "").strip()

    # 官方 Responses API：output 是一个 item 数组，文本藏在 content[].text
    output = body.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("text"):
                    chunks.append(str(part["text"]))
        if chunks:
            return "".join(chunks).strip()

    return str(body.get("output_text") or "").strip()


class OpenAICompatClient:
    """OpenAI 兼容接口客户端（仅用标准库 urllib，不强制依赖 SDK）。

    wire_api="responses"       -> POST {base_url}/responses
    wire_api="chat_completions" -> POST {base_url}/chat/completions

    reasoning_effort 为 None 时不发送推理参数，两种协议都能用。
    """

    name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        wire_api: str = "chat_completions",
        api_key_env: str = "OPENAI_API_KEY",
        api_key: str | None = None,
        timeout: float = 120.0,
        temperature: float = 0.0,
        reasoning_effort: str | None = None,
        disable_response_storage: bool = True,
        store_raw: bool | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.wire_api = wire_api
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env, "")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        # 对应 Codex 配置里的 disable_response_storage：只对 responses 协议有意义。
        # store_raw 是给环境变量用的字符串开关，优先级高于布尔默认值。
        self.disable_response_storage = (
            disable_response_storage if store_raw is None else store_raw
        )

    def is_available(self) -> bool:
        """有 base_url、有 model、有 key 才算可用。

        三者缺一就不是"LLM 挂了"，而是"压根没配"——这两种情况在消融实验里
        结论完全不同，不能混在同一个 try/except 里。
        """
        return bool(self.base_url and self.model and self.api_key)

    def _endpoint(self) -> str:
        suffix = "/responses" if self.wire_api == "responses" else "/chat/completions"
        return f"{self.base_url}{suffix}"

    def _payload(self, prompt: str) -> dict:
        if self.wire_api == "responses":
            payload: dict = {"model": self.model, "input": prompt}
            if self.reasoning_effort:
                payload["reasoning"] = {"effort": self.reasoning_effort}
            if self.disable_response_storage:
                # 不落库：避免用户问题/中间证据被服务端留存
                payload["store"] = False
        else:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
            }
            if self.reasoning_effort:
                payload["reasoning_effort"] = self.reasoning_effort
        return payload

    def complete(self, prompt: str) -> str:
        if not self.is_available():
            raise RuntimeError(
                f"LLM 不可用：缺少 base_url/model/key"
                f"（key 请设置环境变量 {self.api_key_env}）"
            )
        req = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(self._payload(prompt)).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:  # noqa: S310
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM 网络错误: {exc.reason}") from exc
        return extract_text(body)


class NullLLM:
    """默认实现：明确不可用。触发时返回 None（等价于"没补齐"）。"""

    name = "null"

    def complete(self, prompt: str) -> str:  # noqa: ARG002
        return ""

    def is_available(self) -> bool:
        return False


def client_from_env(
    *,
    base_url_env: str = "QP_LLM_BASE_URL",
    model_env: str = "QP_LLM_MODEL",
    api_key_env: str = "OPENAI_API_KEY",
    wire_api_env: str = "QP_LLM_WIRE_API",
    reasoning_env: str = "QP_LLM_REASONING_EFFORT",
) -> OpenAICompatClient:
    """按环境变量组装客户端；缺 base_url 或 key 时返回不可用实例。

    配置来源按优先级：真实环境变量 > 项目根目录 .env > 内置默认值。
    所以临时做消融实验时，直接 QP_LLM_MODEL=xxx python xxx.py 就能覆盖 .env，
    不必去改文件。
    """
    # .env 只是补充，不是覆盖：已在环境里的变量保持原值
    load_env_file()

    return OpenAICompatClient(
        base_url=_clean_base_url(os.environ.get(base_url_env, "")),
        model=os.environ.get(model_env, "").strip(),
        wire_api=os.environ.get(wire_api_env, "chat_completions").strip() or "chat_completions",
        reasoning_effort=os.environ.get(reasoning_env) or None,
        api_key_env=api_key_env,
        store_raw=os.environ.get("QP_LLM_DISABLE_RESPONSE_STORAGE", "").strip().lower()
        not in {"0", "false", "no"},
    )


def _clean_base_url(raw: str) -> str:
    """规整 base_url：去掉复制过程中粘进来的包装符号。

    从文档/聊天里复制地址时经常带进 Markdown 链接或成对括号，
    例如 [https://api.x.com](https://api.x.com)、<https://api.x.com>、
    （https://api.x.com）。这些直接拼进 URL 会 404 或直接报错，
    而症状出现在网络层，很难联想到是配置文件格式问题，所以在源头洗掉。
    """
    value = (raw or "").strip().strip("'\"")
    if not value:
        return ""
    for closer in ("]", ")", "）"):
        index = value.find(closer)
        if index > 0:
            value = value[:index]
    return value.strip().rstrip("/")
