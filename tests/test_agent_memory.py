import tempfile
import unittest
from pathlib import Path

from brain.agent import classify_task, needs_web_search
from brain.knowledge import KnowledgeBase


class AgentRoutingTests(unittest.TestCase):
    def test_classifies_code_requests(self):
        self.assertEqual(classify_task("Viết code Python đọc JSON"), "code")

    def test_classifies_reasoning_requests(self):
        self.assertEqual(classify_task("Vì sao kết luận này đúng?"), "reasoning")

    def test_detects_fresh_information_requests(self):
        self.assertTrue(needs_web_search("Tìm thông tin mới nhất về Python"))
        self.assertFalse(needs_web_search("Giải thích định lý Pythagore"))


class KnowledgeBaseTests(unittest.TestCase):
    def test_add_and_retrieve_note(self):
        with tempfile.TemporaryDirectory() as folder:
            database = Path(folder) / "memory.sqlite3"
            memory = KnowledgeBase(database)
            memory.add(
                "Python dùng thụt lề để xác định khối lệnh.",
                topic="Python",
                source="Tài liệu khóa học",
            )
            results = memory.search("Python khối lệnh")
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].topic, "Python")
            self.assertIn("thụt lề", results[0].content)
            self.assertEqual(memory.count(), 1)

    def test_deduplicates_training_examples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "learned.jsonl"
            memory = KnowledgeBase(Path(folder) / "memory.sqlite3")
            self.assertTrue(memory.add_training_example("2+2?", "4", path))
            self.assertFalse(memory.add_training_example("2+2?", "4", path))
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
