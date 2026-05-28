import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
import yaml

from configs.project_config import load_project_config, save_project_config_snapshot
from src.models.commons import center_slice_segments
from src.models.models import SynthesizerTrnMs256NSFsid, SynthesizerTrnMs256NSFsid_nono
from src.train.runner import (
    build_ddsp_validation_audio_dict,
    build_ddsp_validation_image_dict,
    extract_validation_sample_names,
)
from src.train import utils as train_utils
from tests.equivalence_helpers import make_temp_dir


def write_project_config(path, work_dir, extra=""):
    root = work_dir.parent
    content = f"""base_config: mute.yaml
name: unit_voice
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


def build_tiny_f0_model():
    return SynthesizerTrnMs256NSFsid(
        spec_channels=5,
        segment_size=4,
        inter_channels=8,
        hidden_channels=8,
        filter_channels=16,
        n_heads=1,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 1, 1]],
        upsample_rates=[2, 2],
        upsample_initial_channel=16,
        upsample_kernel_sizes=[4, 4],
        spk_embed_dim=2,
        gin_channels=4,
        sr=32000,
        is_half=False,
    )


def build_tiny_nof0_model():
    return SynthesizerTrnMs256NSFsid_nono(
        spec_channels=5,
        segment_size=4,
        inter_channels=8,
        hidden_channels=8,
        filter_channels=16,
        n_heads=1,
        n_layers=1,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 1, 1]],
        upsample_rates=[2, 2],
        upsample_initial_channel=16,
        upsample_kernel_sizes=[4, 4],
        spk_embed_dim=2,
        gin_channels=4,
        is_half=False,
    )


class TrainValidationConfigTest(unittest.TestCase):
    def test_load_project_config_uses_preprocess_split_paths(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "unit_voice"
            write_project_config(
                config_path,
                work_dir,
                extra="""
preprocess:
  validation_split: 0.25
  validation_seed: 999
""",
            )

            config = load_project_config(config_path, reset=True)

            self.assertEqual(config["preprocess"]["validation_split"], 0.25)
            self.assertEqual(config["preprocess"]["validation_seed"], 999)
            self.assertTrue(config["paths"]["training_files"].endswith("train_filelist.txt"))
            self.assertTrue(config["paths"]["validation_files"].endswith("val_filelist.txt"))
            self.assertEqual(
                config["data"]["training_files"],
                config["paths"]["training_files"],
            )
            self.assertEqual(
                config["data"]["validation_files"],
                config["paths"]["validation_files"],
            )

    def test_load_project_config_rejects_removed_train_validation_keys_in_source(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "unit_voice"
            write_project_config(
                config_path,
                work_dir,
                extra="""
train:
  validation_split: 0.2
""",
            )

            with self.assertRaisesRegex(ValueError, "preprocess.validation_split"):
                load_project_config(config_path, reset=True)

    def test_load_project_config_rejects_removed_train_validation_keys_in_snapshot(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "unit_voice"
            write_project_config(config_path, work_dir)
            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "config.yaml").write_text(
                "train:\n  validation_preview_index: 0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Validation now logs the full validation set"):
                load_project_config(config_path, reset=False)

    def test_load_project_config_resolves_dot_relative_base_config(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_dir = tmp_path / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            base_path = config_dir / "base.yaml"
            config_path = config_dir / "project.yaml"
            work_dir = tmp_path / "ckpt" / "unit_voice"
            root = work_dir.parent

            base_path.write_text(
                """base_config: mute.yaml
preprocess:
  validation_split: 0.25
""",
                encoding="utf-8",
            )
            config_path.write_text(
                f"""base_config: ./base
name: unit_voice
work_dir: {work_dir.as_posix()}
data_root: {(root / "data").as_posix()}
ckpt_root: {(root / "ckpt").as_posix()}
pretrain_root: {(root / "pretrain").as_posix()}
model:
  spk_embed_dim: 1
""",
                encoding="utf-8",
            )

            config = load_project_config(config_path, reset=True)

        self.assertEqual(config["preprocess"]["validation_split"], 0.25)

    def test_save_project_config_snapshot_uses_replayable_payload(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "unit_voice"
            write_project_config(config_path, work_dir)

            config = load_project_config(config_path, reset=True)
            snapshot_path = work_dir / "saved_config.yaml"

            save_project_config_snapshot(config, snapshot_path)

            payload = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["base_config"], [])
        self.assertEqual(payload["infer"]["model_path"], "auto")
        self.assertEqual(
            payload["infer"]["model_path"],
            config["replayable_config"]["infer"]["model_path"],
        )
        self.assertNotEqual(payload["infer"]["model_path"], config["infer"]["model_path"])
        self.assertNotIn("paths", payload)
        self.assertNotIn("replayable_config", payload)
        self.assertNotIn("snapshot_lookup_path", payload)
        self.assertNotIn("device_request", payload["runtime"])
        self.assertNotIn("is_half_request", payload["runtime"])
        self.assertNotIn("n_cpu_request", payload["runtime"])
        self.assertNotIn("profile", payload["runtime"])
        self.assertNotIn("fp16_run_request", payload["train"])

    def test_center_slice_segments_rejects_short_validation_segments(self):
        x = torch.zeros(1, 2, 3)
        with self.assertRaisesRegex(ValueError, "segment_size=4"):
            center_slice_segments(x, torch.tensor([3]), 4)

    def test_ddsp_validation_audio_tags_use_sample_name(self):
        gt_audio = torch.tensor([1.0, 2.0])
        pred_audio = torch.tensor([3.0, 4.0])

        audio_dict = build_ddsp_validation_audio_dict("clip_0", gt_audio, pred_audio)

        self.assertTrue(torch.equal(audio_dict["clip_0/gt.wav"], gt_audio))
        self.assertTrue(torch.equal(audio_dict["clip_0/pred.wav"], pred_audio))

    def test_ddsp_validation_image_tags_use_sample_name(self):
        gt_mel = torch.ones(4, 3)
        pred_mel = torch.zeros(4, 3)

        image_dict = build_ddsp_validation_image_dict("clip_0", gt_mel, pred_mel)

        self.assertEqual(list(image_dict.keys()), ["clip_0"])
        self.assertEqual(image_dict["clip_0"].ndim, 3)
        self.assertEqual(image_dict["clip_0"].shape[2], 3)

    def test_ddsp_validation_image_uses_gt_pred_and_signed_diff(self):
        gt_mel = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        pred_mel = torch.tensor([[0.5, 3.0], [5.0, 1.5]])
        fake_image = np.zeros((4, 4, 3), dtype=np.uint8)

        with mock.patch(
            "src.train.runner.utils.plot_validation_mels_to_numpy",
            return_value=fake_image,
        ) as plot_mock:
            image_dict = build_ddsp_validation_image_dict("clip_0", gt_mel, pred_mel)

        self.assertIs(image_dict["clip_0"], fake_image)
        called_gt, called_pred, called_diff = plot_mock.call_args.args
        np.testing.assert_array_equal(called_gt, gt_mel.numpy())
        np.testing.assert_array_equal(called_pred, pred_mel.numpy())
        np.testing.assert_array_equal(called_diff, (pred_mel - gt_mel).numpy())

    def test_plot_validation_mels_to_numpy_returns_rgb_image(self):
        gt_mel = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.5]], dtype=np.float32)
        pred_mel = np.array([[0.5, 0.5, -0.5], [1.0, 1.5, 0.25]], dtype=np.float32)
        diff_mel = pred_mel - gt_mel

        image = train_utils.plot_validation_mels_to_numpy(gt_mel, pred_mel, diff_mel)

        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)

    def test_plot_spectrogram_to_numpy_returns_rgb_image(self):
        spectrogram = np.array([[1.0, 0.0, -1.0], [2.0, 1.0, 0.5]], dtype=np.float32)

        image = train_utils.plot_spectrogram_to_numpy(spectrogram)

        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)

    def test_plot_alignment_to_numpy_returns_rgb_image(self):
        alignment = np.array([[0.0, 0.5, 1.0], [1.0, 0.5, 0.0]], dtype=np.float32)

        image = train_utils.plot_alignment_to_numpy(alignment, info="demo")

        self.assertEqual(image.ndim, 3)
        self.assertEqual(image.shape[2], 3)

    def test_extract_validation_sample_names_reads_appended_metadata(self):
        self.assertEqual(
            extract_validation_sample_names(tuple(range(9)) + (("a",),), use_f0=True),
            ("a",),
        )
        self.assertEqual(
            extract_validation_sample_names(tuple(range(7)) + (("b",),), use_f0=False),
            ("b",),
        )

    def test_reconstruct_full_returns_full_length_audio_for_f0_model(self):
        model = build_tiny_f0_model()
        phone = torch.randn(1, 8, 256)
        phone_lengths = torch.tensor([8], dtype=torch.long)
        pitch = torch.randint(1, 255, (1, 8), dtype=torch.long)
        pitchf = torch.rand(1, 8) + 1.0
        spec = torch.randn(1, 5, 8)
        spec_lengths = torch.tensor([8], dtype=torch.long)
        sid = torch.tensor([0], dtype=torch.long)

        with torch.no_grad():
            audio = model.reconstruct_full(
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
            )

        self.assertEqual(tuple(audio.shape[:2]), (1, 1))
        self.assertEqual(audio.size(-1), 32)

    def test_reconstruct_full_returns_full_length_audio_for_nof0_model(self):
        model = build_tiny_nof0_model()
        phone = torch.randn(1, 8, 256)
        phone_lengths = torch.tensor([8], dtype=torch.long)
        spec = torch.randn(1, 5, 8)
        spec_lengths = torch.tensor([8], dtype=torch.long)
        sid = torch.tensor([0], dtype=torch.long)

        with torch.no_grad():
            audio = model.reconstruct_full(
                phone, phone_lengths, spec, spec_lengths, sid
            )

        self.assertEqual(tuple(audio.shape[:2]), (1, 1))
        self.assertEqual(audio.size(-1), 32)


if __name__ == "__main__":
    unittest.main()
