import tempfile
import unittest
from pathlib import Path

from brain.agent import ResearchAgent
from brain.knowledge import KnowledgeBase


class EmptySearchProvider:
    def search(self, query, max_results=5):
        return []


class AgentFallbackTests(unittest.TestCase):
    def test_symbolic_reasoning_works_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = ResearchAgent(
                model=None,
                device="cpu",
                memory=KnowledgeBase(Path(folder) / "memory.sqlite3"),
                search_provider=EmptySearchProvider(),
            )
            result = agent.reply("Tính 15 trừ 3 nhân 4.")
            self.assertEqual(result.answer, "Kết quả là 3.")
            self.assertEqual(result.reasoning_method, "Tính toán số học có kiểm tra cú pháp")
            self.assertFalse(result.model_used)

    def test_unknown_question_is_not_fabricated_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = ResearchAgent(
                model=None,
                device="cpu",
                memory=KnowledgeBase(Path(folder) / "memory.sqlite3"),
                search_provider=EmptySearchProvider(),
            )
            result = agent.reply("Giải thích lịch sử triết học phương Tây.")
            self.assertIn("chưa có checkpoint", result.answer)
            self.assertFalse(result.model_used)

    def test_taught_note_is_returned_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = KnowledgeBase(Path(folder) / "memory.sqlite3")
            memory.add("Python dùng thụt lề để tạo khối lệnh.", topic="Python")
            agent = ResearchAgent(
                model=None,
                device="cpu",
                memory=memory,
                search_provider=EmptySearchProvider(),
            )
            result = agent.reply("Python dùng gì để tạo khối lệnh?")
            self.assertIn("thụt lề", result.answer)
            self.assertEqual(len(result.memory_hits), 1)


if __name__ == "__main__":
    unittest.main()
