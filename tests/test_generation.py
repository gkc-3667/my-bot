import unittest
from types import SimpleNamespace

import torch

from brain.generate import generate
from brain.tokenizer import EOS_ID


class AlwaysEndModel:
    def __init__(self):
        self.position_embedding = SimpleNamespace(num_embeddings=16)

    def eval(self):
        return self

    def __call__(self, tokens):
        logits = torch.zeros(tokens.size(0), tokens.size(1), 259)
        logits[:, :, EOS_ID] = 100.0
        return logits, None


class GenerationTests(unittest.TestCase):
    def test_generation_stops_on_eos(self):
        result = generate(
            AlwaysEndModel(),
            prompt_ids=[10, 11],
            max_new_tokens=10,
            temperature=1.0,
            top_k=1,
            device="cpu",
        )
        self.assertEqual(result, [10, 11, EOS_ID])

    def test_empty_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            generate(
                AlwaysEndModel(),
                prompt_ids=[],
                max_new_tokens=1,
                temperature=1.0,
                top_k=1,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main()
