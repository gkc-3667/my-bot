import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from training.train import (
    _restore_rng_state,
    restore_checkpoint_state,
    save_checkpoint,
)


class _Config:
    def __init__(self):
        self.vocab_size = 4
        self.max_seq_len = 8
        self.d_model = 2


class TrainingResumeTests(unittest.TestCase):
    def _make_model_optimizer(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        loss = model(torch.ones(2, 2)).sum()
        loss.backward()
        optimizer.step()
        return model, optimizer

    def test_checkpoint_restores_weights_optimizer_and_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "resume.pt"
            model, optimizer = self._make_model_optimizer()
            config = _Config()
            expected_weights = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }
            random.seed(123)
            torch.manual_seed(123)
            save_checkpoint(
                path, model, optimizer, config, 17, 1.25, 0.2,
                "abc123", 123, 2,
            )
            expected_python_random = random.random()
            expected_torch_random = torch.rand(4)

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(10)
            new_optimizer = torch.optim.AdamW(model.parameters(), lr=0.5)
            random.seed(999)
            torch.manual_seed(999)
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            result = restore_checkpoint_state(
                checkpoint, model, new_optimizer, config,
                "abc123", 0.2, 2,
            )

            self.assertEqual(result, (17, True, True, True))
            for key, value in expected_weights.items():
                self.assertTrue(torch.equal(model.state_dict()[key], value))
            self.assertTrue(new_optimizer.state)
            self.assertEqual(random.random(), expected_python_random)
            self.assertTrue(torch.equal(torch.rand(4), expected_torch_random))
            self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_reports_incomplete_rng_restore_when_cuda_state_is_missing(self):
        state = {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": []}
        with patch("training.train.torch.cuda.is_available", return_value=True):
            with patch("training.train.torch.cuda.device_count", return_value=1):
                with patch("training.train.torch.cuda.set_rng_state_all") as set_cuda:
                    self.assertFalse(_restore_rng_state(state))
                    set_cuda.assert_not_called()

    def test_rejects_different_corpus_unless_explicitly_allowed(self):
        model, optimizer = self._make_model_optimizer()
        config = _Config()
        checkpoint = {
            "model": model.state_dict(),
            "config": vars(config).copy(),
            "step": 4,
            "data_sha256": "old-data",
            "val_fraction": 0.2,
            "batch_size": 2,
        }
        with self.assertRaisesRegex(ValueError, "Training corpus differs"):
            restore_checkpoint_state(
                checkpoint, model, optimizer, config, "new-data", 0.2, 2
            )
        result = restore_checkpoint_state(
            checkpoint, model, optimizer, config, "new-data", 0.2, 2,
            allow_data_change=True,
        )
        self.assertEqual(result, (4, False, False, False))

    def test_rejects_batch_size_mismatch(self):
        model, optimizer = self._make_model_optimizer()
        config = _Config()
        checkpoint = {
            "model": model.state_dict(),
            "config": vars(config).copy(),
            "step": 4,
            "data_sha256": "same-data",
            "val_fraction": 0.2,
            "batch_size": 4,
        }
        with self.assertRaisesRegex(ValueError, "Batch size differs"):
            restore_checkpoint_state(
                checkpoint, model, optimizer, config, "same-data", 0.2, 2
            )

    def test_rejects_different_validation_split(self):
        model, optimizer = self._make_model_optimizer()
        config = _Config()
        checkpoint = {
            "model": model.state_dict(),
            "config": vars(config).copy(),
            "step": 4,
            "data_sha256": "same-data",
            "val_fraction": 0.2,
            "batch_size": 2,
        }
        with self.assertRaisesRegex(ValueError, "Validation split differs"):
            restore_checkpoint_state(
                checkpoint, model, optimizer, config, "same-data", 0.1, 2
            )

    def test_legacy_checkpoint_can_restore_weights_but_not_exact_training_state(self):
        model, optimizer = self._make_model_optimizer()
        config = _Config()
        checkpoint = {
            "model": model.state_dict(),
            "config": vars(config).copy(),
            "step": 3,
        }
        result = restore_checkpoint_state(
            checkpoint, model, optimizer, config, "data", 0.05, 2
        )
        self.assertEqual(result, (3, False, False, False))


if __name__ == "__main__":
    unittest.main()
