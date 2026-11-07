import json
import tempfile
import unittest
from pathlib import Path

from brain.tokenizer import ByteTokenizer
from training.sft import load_examples


class SFTDataTests(unittest.TestCase):
    def test_loads_examples_from_multiple_jsonl_files(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / "one.jsonl"
            second = Path(folder) / "two.jsonl"
            first.write_text(
                json.dumps({"prompt": "Câu hỏi một", "answer": "Trả lời một"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            second.write_text(
                json.dumps({"prompt": "Câu hỏi hai", "answer": "Trả lời hai"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            examples = load_examples([first, second], ByteTokenizer(), 256)
            self.assertEqual(len(examples), 2)
            self.assertTrue((examples[0][1] == -100).any().item())
            self.assertTrue((examples[1][1] == -100).any().item())


if __name__ == "__main__":
    unittest.main()
