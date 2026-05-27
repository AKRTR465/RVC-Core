import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import amp

from configs.project_config import load_project_config
from src.train.deterministic_gpu import reflect_pad_last, spectrogram_torch
from src.train import utils as train_utils
from tests.equivalence_helpers import make_temp_dir


def write_project_config(path, work_dir, extra=""):
    root = work_dir.parent
    content = f"""base_config: mute.yaml
name: strict_unit_voice
work_dir: {work_dir.as_posix()}
data_root: {(root / "data").as_posix()}
ckpt_root: {(root / "ckpt").as_posix()}
pretrain_root: {(root / "pretrain").as_posix()}
model:
  spk_embed_dim: 1
"""
    if extra:
        content += extra.lstrip("\n")
        if not content.endswith("\n"):
            content += "\n"
    Path(path).write_text(content, encoding="utf-8")


class StrictRuntimeConfigTest(unittest.TestCase):
    def test_load_project_config_enforces_strict_runtime_defaults(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "strict_unit_voice"
            write_project_config(
                config_path,
                work_dir,
                extra="""
train:
  numeric_backend: deterministic_gpu
""",
            )

            with mock.patch(
                "configs.project_config._detect_runtime_environment",
                return_value={
                    "device": "cuda:0",
                    "device_request": "cuda:0",
                    "gpu_name": "Mock GPU",
                    "gpu_mem_gb": 8,
                    "supports_half": True,
                },
            ):
                config = load_project_config(config_path, reset=True)

        self.assertEqual(config["train"]["numeric_backend"], "deterministic_gpu")
        self.assertEqual(config["train"]["grad_scaler_init_scale"], 32.0)
        self.assertEqual(config["runtime"]["deterministic_algorithms"], "error")
        self.assertTrue(config["runtime"]["disable_tf32"])
        self.assertEqual(config["runtime"]["cublas_workspace_config"], ":4096:8")

    def test_load_project_config_accepts_legacy_mel_loss_device_alias(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "strict_unit_voice"
            write_project_config(
                config_path,
                work_dir,
                extra="""
train:
  mel_loss_device: deterministic_gpu
""",
            )

            with mock.patch(
                "configs.project_config._detect_runtime_environment",
                return_value={
                    "device": "cuda:0",
                    "device_request": "cuda:0",
                    "gpu_name": "Mock GPU",
                    "gpu_mem_gb": 8,
                    "supports_half": True,
                },
            ):
                config = load_project_config(config_path, reset=True)

        self.assertEqual(config["train"]["numeric_backend"], "deterministic_gpu")
        self.assertNotIn("mel_loss_device", config["train"])

    def test_load_project_config_rejects_equal_test_cpu_alias(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "strict_unit_voice"
            write_project_config(
                config_path,
                work_dir,
                extra="""
train:
  mel_loss_device: cpu
""",
            )

            with self.assertRaisesRegex(ValueError, "equal-test alignment runs"):
                load_project_config(config_path, reset=True)


class DeterministicGpuHelperTest(unittest.TestCase):
    def test_reflect_pad_last_matches_expected_indices(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])

        padded = reflect_pad_last(x, 1, 2)

        expected = torch.tensor([[2.0, 1.0, 2.0, 3.0, 2.0, 1.0]])
        self.assertTrue(torch.equal(padded, expected))

    def test_spectrogram_torch_returns_finite_cpu_tensor(self):
        y = torch.arange(12, dtype=torch.float32).unsqueeze(0)

        spec = spectrogram_torch(
            y,
            n_fft=4,
            sampling_rate=48000,
            hop_size=2,
            win_size=4,
            center=False,
        )

        self.assertEqual(tuple(spec.shape), (1, 3, 6))
        self.assertTrue(torch.isfinite(spec).all().item())

    def test_checkpoint_round_trip_restores_grad_scaler_state(self):
        with make_temp_dir() as tmp:
            checkpoint_path = Path(tmp) / "G_1.pth"
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
            _, _, _, epoch = train_utils.load_checkpoint(
                str(checkpoint_path),
                restored_model,
                restored_optimizer,
                scaler=restored_scaler,
            )

        self.assertEqual(epoch, 3)
        self.assertEqual(restored_scaler.state_dict()["scale"], scaler.state_dict()["scale"])


if __name__ == "__main__":
    unittest.main()
