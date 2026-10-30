from __future__ import annotations

import os
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.dotenv_loader import load_env_file
from src.common.llm_client import client_from_env


class ParseTest(unittest.TestCase):
    """降级解析器的行为：注释、空行、引号、export 前缀。"""

    def _load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(text, encoding="utf-8")
            with unittest.mock.patch.dict(os.environ, {}, clear=False):
                for key in ("QP_TEST_A", "QP_TEST_B", "QP_TEST_C"):
                    os.environ.pop(key, None)
                return load_env_file(path)

    def test_basic_pairs(self):
        applied = self._load("QP_TEST_A=1\nQP_TEST_B=two\n")
        self.assertEqual(applied["QP_TEST_A"], "1")
        self.assertEqual(applied["QP_TEST_B"], "two")

    def test_comments_and_blank_lines_ignored(self):
        applied = self._load("# comment\n\nQP_TEST_A=1  # trailing\n")
        self.assertEqual(applied["QP_TEST_A"], "1")

    def test_hash_without_leading_space_is_kept(self):
        """只有前面是空白时 # 才是注释，否则可能是值的一部分。"""
        applied = self._load("QP_TEST_A=sk-ab#cd\n")
        self.assertEqual(applied["QP_TEST_A"], "sk-ab#cd")

    def test_quotes_are_stripped(self):
        applied = self._load('QP_TEST_A="hello"\nQP_TEST_B=\'world\'\n')
        self.assertEqual(applied["QP_TEST_A"], "hello")
        self.assertEqual(applied["QP_TEST_B"], "world")

    def test_export_prefix(self):
        applied = self._load("export QP_TEST_A=1\n")
        self.assertEqual(applied["QP_TEST_A"], "1")

    def test_missing_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_env_file(Path(tmp) / "nope.env"), {})


class PrecedenceTest(unittest.TestCase):
    """真实环境变量优先于 .env——消融实验要靠这个临时改模型。"""

    def test_existing_env_wins_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("QP_TEST_MODEL=from-file\n", encoding="utf-8")
            with unittest.mock.patch.dict(os.environ, {"QP_TEST_MODEL": "from-env"}):
                applied = load_env_file(path)
                self.assertEqual(applied, {})
                self.assertEqual(os.environ["QP_TEST_MODEL"], "from-env")

    def test_override_true_replaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("QP_TEST_MODEL=from-file\n", encoding="utf-8")
            with unittest.mock.patch.dict(os.environ, {"QP_TEST_MODEL": "from-env"}):
                load_env_file(path, override=True)
                self.assertEqual(os.environ["QP_TEST_MODEL"], "from-file")

    def test_missing_values_are_not_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("QP_TEST_A=\n", encoding="utf-8")
            applied = load_env_file(path)
        self.assertEqual(applied, {})
        self.assertNotIn("QP_TEST_A", os.environ)


class ClientFromEnvFileTest(unittest.TestCase):
    """client_from_env 必须能只靠 .env 就装配出可用客户端。"""

    def _client(self, text, extra_env=None):
        """在**干净环境**里装配客户端。

        必须屏蔽两样东西，否则测试结果取决于开发机上恰好有什么配置：
          1. 进程里已有的 QP_*/OPENAI_* 变量；
          2. 项目根目录真实的 .env —— client_from_env() 会自己去读它，
             所以本机的 QP_LLM_WIRE_API=responses 之类会渗进来，
             让"默认值"用例悄悄跑到非默认分支上。
        """
        from src.common import llm_client as llm_module

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(text, encoding="utf-8")
            env = dict(extra_env or {})
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                with unittest.mock.patch.object(
                    llm_module, "load_env_file",
                    lambda *a, **k: load_env_file(path),
                ):
                    return client_from_env()

    def test_full_config_from_file(self):
        client = self._client(
            "OPENAI_API_KEY=sk-test\n"
            "QP_LLM_BASE_URL=https://api.example.com\n"
            "QP_LLM_MODEL=deepseek-v4-flash\n"
            "QP_LLM_WIRE_API=responses\n"
            "QP_LLM_REASONING_EFFORT=high\n"
        )
        self.assertEqual(client.model, "deepseek-v4-flash")
        self.assertEqual(client.wire_api, "responses")
        self.assertEqual(client.reasoning_effort, "high")
        self.assertTrue(client.is_available())

    def test_markdown_link_style_base_url_is_cleaned(self):
        """用户常把 base_url 粘成 Markdown 链接，直接用会 404。"""
        client = self._client(
            "OPENAI_API_KEY=sk-test\n"
            "QP_LLM_BASE_URL=https://api.aifang.online](https://api.aifang.online)\n"
            "QP_LLM_MODEL=m\n"
            "QP_LLM_WIRE_API=responses\n"
        )
        self.assertEqual(client.base_url, "https://api.aifang.online")
        self.assertEqual(client._endpoint(), "https://api.aifang.online/responses")

    def test_disable_storage_switch(self):
        client = self._client(
            "OPENAI_API_KEY=sk-test\n"
            "QP_LLM_BASE_URL=https://x\n"
            "QP_LLM_MODEL=m\n"
            "QP_LLM_WIRE_API=responses\n"
            "QP_LLM_DISABLE_RESPONSE_STORAGE=false\n"
        )
        self.assertFalse(client.disable_response_storage)
        self.assertNotIn("store", client._payload("q"))

    def test_storage_disabled_by_default(self):
        """store 只属于 responses 协议，默认协议下不该出现在请求体里。"""
        chat = self._client(
            "OPENAI_API_KEY=sk-test\nQP_LLM_BASE_URL=https://x\nQP_LLM_MODEL=m\n"
        )
        self.assertEqual(chat.wire_api, "chat_completions")
        self.assertTrue(chat.disable_response_storage)
        self.assertNotIn("store", chat._payload("q"))

        resp = self._client(
            "OPENAI_API_KEY=sk-test\n"
            "QP_LLM_BASE_URL=https://x\n"
            "QP_LLM_MODEL=m\n"
            "QP_LLM_WIRE_API=responses\n"
        )
        self.assertIs(resp._payload("q")["store"], False)

    def test_missing_config_yields_unavailable_client(self):
        """什么都不配时返回的是"不可用"客户端，而不是抛异常。"""
        client = self._client("")
        self.assertFalse(client.is_available())


if __name__ == "__main__":
    unittest.main()
