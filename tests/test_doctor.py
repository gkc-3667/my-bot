import tempfile
import unittest
from pathlib import Path

from brain.doctor import run_diagnostics


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostics_report_missing_checkpoint_without_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checks = run_diagnostics(
                checkpoint=root / "missing.pt",
                corpus_dir=root / "empty-corpus",
                memory_path=root / "memory.sqlite3",
            )
            by_name = {item["name"]: item for item in checks}
            self.assertIn("Symbolic reasoning", by_name)
            self.assertTrue(by_name["Symbolic reasoning"]["ok"])
            self.assertFalse(by_name["Language-model checkpoint"]["ok"])
            self.assertTrue(by_name["SQLite knowledge memory"]["ok"])


if __name__ == "__main__":
    unittest.main()
