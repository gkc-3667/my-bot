import json
import unittest
from pathlib import Path

from brain.reasoning import solve_question


class HeldOutReasoningDatasetTests(unittest.TestCase):
    def test_symbolic_engine_solves_smoke_dataset(self):
        path = Path(__file__).resolve().parents[1] / "data" / "evaluation" / "reasoning_smoke.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(rows), 8)
        failures = []
        for row in rows:
            result = solve_question(row["prompt"])
            if result is None:
                failures.append((row["prompt"], "no supported derivation"))
                continue
            output = result.answer.casefold()
            if not any(expected.casefold() in output for expected in row["expected"]):
                failures.append((row["prompt"], result.answer))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
