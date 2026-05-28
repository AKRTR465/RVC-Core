import unittest
from pathlib import Path

import torch
from torch import amp

from src.train import utils as train_utils
from tests.equivalence_helpers import make_temp_dir


class TrainCheckpointTest(unittest.TestCase):
    def test_get_logger_routes_checkpoint_messages_to_train_log(self):
        with make_temp_dir() as tmp:
            model_dir = Path(tmp) / "model"
            checkpoint_path = model_dir / "G_1.pth"

            logger = train_utils.get_logger(str(model_dir))
            model = torch.nn.Linear(4, 2)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
            scaler = amp.GradScaler("cpu", enabled=True, init_scale=32.0)

            train_utils.save_checkpoint(
                model,
                optimizer,
                1.0e-4,
                3,
                str(checkpoint_path),
                scaler=scaler,
            )

            restored_model = torch.nn.Linear(4, 2)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1.0e-4)
            restored_scaler = amp.GradScaler("cpu", enabled=True, init_scale=8.0)
            train_utils.load_checkpoint(
                str(checkpoint_path),
                restored_model,
                restored_optimizer,
                scaler=restored_scaler,
            )

            for handler in logger.handlers:
                if hasattr(handler, "flush"):
                    handler.flush()

            log_text = (model_dir / "train.log").read_text(encoding="utf-8")

        self.assertIn("Saving model and optimizer state at epoch 3", log_text)
        self.assertIn("Loaded checkpoint", log_text)

    def test_latest_checkpoint_path_prefers_highest_numeric_suffix(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            for name in ("G_2.pth", "G_10.pth", "G_9.pth"):
                (tmp_path / name).write_bytes(b"checkpoint")

            latest = train_utils.latest_checkpoint_path(str(tmp_path))

        self.assertEqual(Path(latest).name, "G_10.pth")


if __name__ == "__main__":
    unittest.main()
