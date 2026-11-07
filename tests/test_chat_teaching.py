import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

from brain.chat import handle_learning_command
from brain.knowledge import KnowledgeBase


class ChatTeachingCommandTests(unittest.TestCase):
    def test_teach_command_persists_a_note(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = KnowledgeBase(Path(folder) / "memory.sqlite3")
            agent = SimpleNamespace(memory=memory)
            output = StringIO()
            with redirect_stdout(output):
                handled = handle_learning_command(
                    agent,
                    "/teach Python | Python dùng thụt lề. | Tài liệu Python",
                )
            self.assertTrue(handled)
            self.assertIn("Đã lưu ghi chú", output.getvalue())
            results = memory.search("Python thụt lề")
            self.assertEqual(len(results), 1)
            self.assertIn("thụt lề", results[0].content)
            self.assertEqual(results[0].source, "Tài liệu Python")

    def test_status_reports_checkpoint_and_rule_reasoner(self):
        with tempfile.TemporaryDirectory() as folder:
            memory = KnowledgeBase(Path(folder) / "memory.sqlite3")
            agent = SimpleNamespace(
                memory=memory,
                model=None,
                search_provider=None,
                search_setup_error="ddgs not installed",
            )
            output = StringIO()
            with redirect_stdout(output):
                handled = handle_learning_command(agent, "/status")
            self.assertTrue(handled)
            self.assertIn("chưa có checkpoint", output.getvalue())
            self.assertIn("Suy luận quy tắc: đang bật", output.getvalue())
            self.assertIn("Ghi chú đã dạy: 0", output.getvalue())

    def test_invalid_teach_command_shows_usage(self):
        agent = SimpleNamespace(memory=None)
        output = StringIO()
        with redirect_stdout(output):
            handled = handle_learning_command(agent, "/teach thiếu dấu phân cách")
        self.assertTrue(handled)
        self.assertIn("Cách dùng", output.getvalue())


if __name__ == "__main__":
    unittest.main()
