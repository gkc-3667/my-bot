import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from brain.agent import ResearchAgent
from brain.knowledge import KnowledgeBase
from brain.llm import OpenAICompatibleProvider


class FakeProvider:
    model = "test-model"

    def __init__(self):
        self.calls = []

    def complete(self, messages, max_tokens=256):
        self.calls.append((messages, max_tokens))
        return "Mình đã hiểu câu hỏi và có thể giải thích từng bước."


class LLMProviderTests(unittest.TestCase):
    def test_from_env_returns_none_without_model(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(OpenAICompatibleProvider.from_env())

    def test_from_env_reads_configuration_without_requiring_key(self):
        values = {
            "GKC_AI_MODEL": "local-test",
            "GKC_AI_BASE_URL": "http://127.0.0.1:11434/v1",
            "GKC_AI_API_KEY": "",
            "GKC_AI_TIMEOUT": "12",
        }
        with patch.dict(os.environ, values, clear=True):
            provider = OpenAICompatibleProvider.from_env()
        self.assertEqual(provider.model, "local-test")
        self.assertEqual(provider.endpoint, "http://127.0.0.1:11434/v1/chat/completions")
        self.assertEqual(provider.timeout, 12)

    def test_complete_sends_messages_and_parses_answer(self):
        response = BytesIO(json.dumps({
            "choices": [{"message": {"content": "Xin chào từ mô hình."}}]
        }).encode("utf-8"))
        provider = OpenAICompatibleProvider(
            "https://example.invalid/v1", "test-model", "secret-test", timeout=5
        )
        with patch("brain.llm.urlopen", return_value=response) as mocked:
            answer = provider.complete(
                [{"role": "user", "content": "Xin chào"}], max_tokens=80
            )
        self.assertEqual(answer, "Xin chào từ mô hình.")
        request = mocked.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["max_tokens"], 80)
        self.assertEqual(request.get_header("Authorization"), "Bearer secret-test")

    def test_rejects_invalid_endpoint(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleProvider("file:///tmp/model", "test-model")

    def test_agent_uses_provider_and_keeps_multi_turn_context(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = FakeProvider()
            agent = ResearchAgent(
                model=None,
                llm_provider=provider,
                memory=KnowledgeBase(Path(folder) / "memory.sqlite3"),
                search_provider=type("NoSearch", (), {"search": lambda *a, **k: []})(),
            )
            first = agent.reply("Chào bạn")
            second = agent.reply("Bạn vừa nói gì?")
        self.assertTrue(first.model_used)
        self.assertTrue(second.model_used)
        self.assertEqual(len(provider.calls), 2)
        second_messages = provider.calls[1][0]
        self.assertTrue(any(
            item["role"] == "assistant"
            and "giải thích từng bước" in item["content"]
            for item in second_messages
        ))
        self.assertTrue(any(
            item["role"] == "user" and item["content"] == "Chào bạn"
            for item in second_messages
        ))


if __name__ == "__main__":
    unittest.main()
