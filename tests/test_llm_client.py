from __future__ import annotations

import json
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.llm_client import OpenAICompatClient, client_from_env, extract_text


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class ExtractTextTest(unittest.TestCase):
    def test_chat_completions_shape(self):
        """实际网关返回的形状：顶层 choices[].message.content"""
        body = {"choices": [{"message": {"content": "  hello  "}}]}
        self.assertEqual(extract_text(body), "hello")

    def test_responses_shape(self):
        """官方 Responses API 形状：output[].content[].text"""
        body = {
            "output": [
                {"content": [{"type": "output_text", "text": "he"}]},
                {"content": [{"type": "output_text", "text": "llo"}]},
            ]
        }
        self.assertEqual(extract_text(body), "hello")

    def test_output_text_shape(self):
        self.assertEqual(extract_text({"output_text": "hi"}), "hi")

    def test_garbage_returns_empty(self):
        self.assertEqual(extract_text(None), "")
        self.assertEqual(extract_text({"unexpected": 1}), "")


class PayloadTest(unittest.TestCase):
    """请求体构造：store 只出现在 responses，reasoning 两边按协议映射。"""

    # 不封装为 setUp：测试应该对真实环境不敏感。
    # 渲染器只读 constructor 参数，不会意外取到环境变量。
    def _payload(self, **kw):
        client = OpenAICompatClient("https://x", "m", api_key="k", **kw)
        return client._payload("q")

    def test_responses_payload(self):
        payload = self._payload(
            wire_api="responses", reasoning_effort="high", disable_response_storage=True
        )
        self.assertEqual(payload["model"], "m")
        self.assertEqual(payload["input"], "q")
        self.assertEqual(payload["reasoning"], {"effort": "high"})
        self.assertIs(payload["store"], False)

    def test_responses_payload_without_storage_flag(self):
        payload = self._payload(
            wire_api="responses", disable_response_storage=False
        )
        self.assertNotIn("store", payload)
        self.assertNotIn("reasoning", payload)

    def test_chat_completions_payload(self):
        payload = self._payload(
            wire_api="chat_completions", reasoning_effort="high", temperature=0.0
        )
        self.assertEqual(payload["messages"], [{"role": "user", "content": "q"}])
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["temperature"], 0.0)
        # chat/completions 没有 store 概念，也不能出现 input
        self.assertNotIn("store", payload)
        self.assertNotIn("input", payload)

    def test_endpoints(self):
        resp = OpenAICompatClient("https://api.example.com/", "m", api_key="k",
                                  wire_api="responses")
        chat = OpenAICompatClient("https://api.example.com", "m", api_key="k",
                                  wire_api="chat_completions")
        self.assertEqual(resp._endpoint(), "https://api.example.com/responses")
        self.assertEqual(chat._endpoint(), "https://api.example.com/chat/completions")


class AvailabilityTest(unittest.TestCase):
    def test_all_configured(self):
        client = OpenAICompatClient("https://x", "m", api_key="k")
        self.assertTrue(client.is_available())

    def test_missing_key(self):
        client = OpenAICompatClient("https://x", "m", api_key="")
        self.assertFalse(client.is_available())

    def test_missing_base_url(self):
        client = OpenAICompatClient("", "m", api_key="k")
        self.assertFalse(client.is_available())

    def test_missing_model(self):
        client = OpenAICompatClient("https://x", "", api_key="k")
        self.assertFalse(client.is_available())

    def test_complete_raises_when_unavailable(self):
        client = OpenAICompatClient("https://x", "m", api_key="")
        with self.assertRaises(RuntimeError):
            client.complete("hi")

    def test_client_from_env(self):
        env = {
            "QP_LLM_BASE_URL": "https://env.example.com",
            "QP_LLM_MODEL": "env-model",
            "QP_LLM_WIRE_API": "responses",
            "QP_LLM_REASONING_EFFORT": "high",
            "OPENAI_API_KEY": "env-key",
        }
        with unittest.mock.patch.dict("os.environ", env, clear=False):
            client = client_from_env()
        self.assertEqual(client.base_url, "https://env.example.com")
        self.assertEqual(client.model, "env-model")
        self.assertEqual(client.wire_api, "responses")
        self.assertEqual(client.reasoning_effort, "high")
        self.assertTrue(client.is_available())


class CompleteTest(unittest.TestCase):
    """complete() 只做一件事：发请求 + 取文本。"""

    def test_sends_post_with_auth_header(self):
        client = OpenAICompatClient(
            "https://api.example.com", "m", api_key="secret-key",
            wire_api="responses", reasoning_effort="high",
        )
        captured = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["method"] = req.method
            captured["body"] = json.loads(req.data.decode("utf-8"))
            captured["auth"] = req.get_header("Authorization")
            captured["timeout"] = timeout
            return _FakeResponse({"choices": [{"message": {"content": "答案"}}]})

        with unittest.mock.patch("urllib.request.urlopen", fake_urlopen):
            text = client.complete("问题")

        self.assertEqual(text, "答案")
        self.assertEqual(captured["url"], "https://api.example.com/responses")
        self.assertEqual(captured["method"], "POST")
        self.assertEqual(captured["auth"], "Bearer secret-key")
        self.assertEqual(captured["body"]["input"], "问题")
        self.assertIs(captured["body"]["store"], False)


class AdapterWiringTest(unittest.TestCase):
    """Adapter 必须按 is_available 决定用 LLM 还是空实现。"""

    def _adapter(self, client):
        from src.parsing.adapter import PipelineParserAdapter

        return PipelineParserAdapter(llm=client)

    def test_available_client_becomes_llm_rewriter(self):
        class Ready:
            name = "ready"

            def complete(self, prompt):
                return ""

            def is_available(self):
                return True

        adapter = self._adapter(Ready())
        from src.parsing.llm_intent import LLMIntentRewriter

        self.assertIsInstance(adapter.pipeline.semantic.rewriter, LLMIntentRewriter)
        self.assertIsNotNone(adapter.llm)

    def test_unavailable_client_becomes_null_rewriter(self):
        class NotReady:
            name = "not_ready"

            def complete(self, prompt):
                return ""

            def is_available(self):
                return False

        adapter = self._adapter(NotReady())
        from src.parsing.llm_intent import NullIntentRewriter

        self.assertIsInstance(adapter.pipeline.semantic.rewriter, NullIntentRewriter)
        self.assertIsNone(adapter.llm)

    def test_none_client_becomes_null_rewriter(self):
        from src.parsing.llm_intent import NullIntentRewriter

        adapter = self._adapter(None)
        self.assertIsInstance(adapter.pipeline.semantic.rewriter, NullIntentRewriter)


if __name__ == "__main__":
    unittest.main()
