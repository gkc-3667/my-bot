import tempfile
import unittest
from pathlib import Path

from training.inspect_corpus import inspect_corpus


class CorpusInspectionTests(unittest.TestCase):
    def test_reports_valid_empty_invalid_and_duplicate_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "a.txt").write_text("Xin chao", encoding="utf-8")
            (root / "b.txt").write_text("Xin chao", encoding="utf-8")
            (root / "empty.txt").write_bytes(b"  \n")
            (root / "bad.txt").write_bytes(b"\xff\xfe")
            report = inspect_corpus(root, seq_len=2, val_fraction=0.2)

        self.assertEqual(report["files_found"], 4)
        self.assertEqual(len(report["valid_files"]), 2)
        self.assertEqual(len(report["empty_files"]), 1)
        self.assertEqual(len(report["invalid_utf8_files"]), 1)
        self.assertEqual(len(report["duplicate_groups"]), 1)
        self.assertEqual(report["bytes"], len("Xin chao".encode("utf-8")) * 2)

    def test_reports_not_ready_when_corpus_is_too_small(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "tiny.txt").write_text("hi", encoding="utf-8")
            report = inspect_corpus(folder, seq_len=16, val_fraction=0.1)
        self.assertFalse(report["ready_for_smoke_training"])

    def test_rejects_invalid_split_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "data.txt").write_text("hello world", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "seq_len"):
                inspect_corpus(folder, seq_len=0)
            with self.assertRaisesRegex(ValueError, "val_fraction"):
                inspect_corpus(folder, val_fraction=0.5)

    def test_requires_text_files(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                inspect_corpus(folder)


if __name__ == "__main__":
    unittest.main()
