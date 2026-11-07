import random
import unittest
from types import SimpleNamespace

import torch

from training.train import evaluate, split_tokens


class _MeanTokenModel:
    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def __call__(self, inputs, targets):
        return None, inputs.float().mean()


class TrainingSplitTests(unittest.TestCase):
    def test_validation_partition_is_separate_and_chronological(self):
        train, validation = split_tokens(
            list(range(1000)), seq_len=10, val_fraction=0.2
        )
        self.assertEqual(train[-1].item(), 799)
        self.assertEqual(validation[0].item(), 800)
        self.assertEqual(len(train), 800)
        self.assertEqual(len(validation), 200)

    def test_rejects_too_small_validation_partition_instead_of_leaking_train_data(self):
        with self.assertRaisesRegex(
            ValueError, "Validation will not reuse training data"
        ):
            split_tokens(list(range(100)), seq_len=20, val_fraction=0.05)

    def test_rejects_invalid_validation_fraction(self):
        for fraction in (0, -0.1, 0.5, 1):
            with self.subTest(fraction=fraction):
                with self.assertRaises(ValueError):
                    split_tokens(
                        list(range(1000)), seq_len=10, val_fraction=fraction
                    )

    def test_evaluation_is_repeatable_and_does_not_consume_training_rng(self):
        tokens = torch.arange(100, dtype=torch.long)
        config = SimpleNamespace(max_seq_len=8)
        model = _MeanTokenModel()

        random.seed(1234)
        expected_next_random = random.random()
        random.seed(1234)
        first = evaluate(model, tokens, config, "cpu", batches=3, batch_size=2)
        actual_next_random = random.random()
        second = evaluate(model, tokens, config, "cpu", batches=3, batch_size=2)

        self.assertEqual(first, second)
        self.assertEqual(actual_next_random, expected_next_random)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
