import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

from src.features import f0 as feature_f0
from src.features import hubert as feature_hubert
from src.features.hubert_fairseq_compat import HubertConfig, HubertModel
from tests.equivalence_helpers import make_temp_dir


def write_tiny_hubert_checkpoint(path, normalize=True, model_name="hubert"):
    config = HubertConfig(
        conv_layers=((4, 2, 2),),
        encoder_layers=1,
        encoder_embed_dim=4,
        encoder_ffn_embed_dim=8,
        encoder_attention_heads=2,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        encoder_layerdrop=0.0,
        dropout_input=0.0,
        dropout_features=0.0,
        final_dim=3,
        label_embs_num=5,
        conv_pos=4,
        conv_pos_groups=1,
    )
    model = HubertModel(config)
    torch.save(
        {
            "cfg": {
                "model": {
                    "_name": model_name,
                    "conv_feature_layers": repr(list(config.conv_layers)),
                    "extractor_mode": config.extractor_mode,
                    "conv_bias": config.conv_bias,
                    "encoder_layers": config.encoder_layers,
                    "encoder_embed_dim": config.encoder_embed_dim,
                    "encoder_ffn_embed_dim": config.encoder_ffn_embed_dim,
                    "encoder_attention_heads": config.encoder_attention_heads,
                    "activation_fn": config.activation_fn,
                    "layer_norm_first": config.layer_norm_first,
                    "dropout": config.dropout,
                    "attention_dropout": config.attention_dropout,
                    "activation_dropout": config.activation_dropout,
                    "encoder_layerdrop": config.encoder_layerdrop,
                    "dropout_input": config.dropout_input,
                    "dropout_features": config.dropout_features,
                    "conv_pos": config.conv_pos,
                    "conv_pos_groups": config.conv_pos_groups,
                    "required_seq_len_multiple": config.required_seq_len_multiple,
                },
                "task": {"normalize": normalize},
            },
            "model": model.state_dict(),
        },
        path,
    )


class HubertHelperTest(unittest.TestCase):
    def test_load_hubert_model_uses_internal_checkpoint_loader(self):
        with make_temp_dir() as tmp:
            checkpoint_path = Path(tmp) / "hubert_tiny.pt"
            write_tiny_hubert_checkpoint(checkpoint_path, normalize=True)

            logs = []
            model, saved_cfg = feature_hubert.load_hubert_model(
                checkpoint_path,
                "cpu",
                is_half=True,
                log_fn=logs.append,
            )

        self.assertIsNotNone(model)
        self.assertFalse(model.training)
        self.assertTrue(model.task_normalize)
        self.assertTrue(saved_cfg.task.normalize)
        self.assertEqual(next(model.parameters()).dtype, torch.float32)
        self.assertIn(f"load model(s) from {checkpoint_path}", logs)
        self.assertIn("move model to cpu", logs)

    def test_load_hubert_model_missing_file_returns_none(self):
        with make_temp_dir() as tmp:
            checkpoint_path = Path(tmp) / "missing.pt"
            logs = []
            model, saved_cfg = feature_hubert.load_hubert_model(
                checkpoint_path,
                "cpu",
                is_half=False,
                log_fn=logs.append,
            )

        self.assertIsNone(model)
        self.assertIsNone(saved_cfg)
        self.assertIn("does not exist", logs[0])

    def test_load_hubert_model_rejects_unsupported_checkpoint(self):
        with make_temp_dir() as tmp:
            checkpoint_path = Path(tmp) / "unsupported.pt"
            write_tiny_hubert_checkpoint(checkpoint_path, model_name="wav2vec2")

            with self.assertRaisesRegex(ValueError, "Unsupported HuBERT checkpoint model"):
                feature_hubert.load_hubert_model(checkpoint_path, "cpu", is_half=False)

    def test_prepare_hubert_waveform_downmixes_and_batches(self):
        waveform = np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32)

        prepared = feature_hubert.prepare_hubert_waveform(waveform, normalize=False)

        expected = torch.tensor([[2.0, 6.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(prepared, expected))

    def test_extract_hubert_features_uses_version_profile(self):
        class FakeModel:
            task_normalize = True

            def __init__(self):
                self.last_inputs = None

            def extract_features(self, **inputs):
                self.last_inputs = inputs
                return (inputs["source"] + 2.0,)

            def final_proj(self, tensor):
                return tensor * 3.0

        model = FakeModel()
        output = feature_hubert.extract_hubert_features(
            model,
            np.array([1.0, 2.0], dtype=np.float32),
            "v1",
            "cpu",
            False,
        )

        self.assertEqual(model.last_inputs["output_layer"], 9)
        self.assertEqual(model.last_inputs["source"].device.type, "cpu")
        expected = torch.tensor([[3.0, 9.0]], dtype=torch.float32)
        self.assertTrue(torch.allclose(output, expected, atol=2.0e-4))

    def test_extract_hubert_features_v2_skips_final_proj(self):
        class FakeModel:
            task_normalize = False

            def __init__(self):
                self.last_inputs = None

            def extract_features(self, **inputs):
                self.last_inputs = inputs
                return (inputs["source"] + 5.0,)

            def final_proj(self, tensor):  # pragma: no cover - should not run
                raise AssertionError("final_proj should not be used for v2")

        model = FakeModel()
        output = feature_hubert.extract_hubert_features(
            model,
            np.array([1.0, 2.0], dtype=np.float32),
            "v2",
            "cpu",
            False,
        )

        self.assertEqual(model.last_inputs["output_layer"], 12)
        expected = torch.tensor([[6.0, 7.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(output, expected))

    def test_read_wave_16k_returns_raw_waveform_before_feature_prep(self):
        soundfile_module = mock.Mock()
        soundfile_module.read.return_value = (
            np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
            16000,
        )

        waveform = feature_hubert.read_wave_16k("demo.wav", soundfile_module)

        np.testing.assert_array_equal(
            waveform,
            np.array([[1.0, 3.0], [5.0, 7.0]], dtype=np.float32),
        )

    def test_read_wave_16k_extract_chain_normalizes_once(self):
        class FakeModel:
            task_normalize = True

            def extract_features(self, **inputs):
                self.last_inputs = inputs
                return (inputs["source"],)

            def final_proj(self, tensor):
                return tensor

        soundfile_module = mock.Mock()
        soundfile_module.read.return_value = (np.array([1.0, 2.0], dtype=np.float32), 16000)

        waveform = feature_hubert.read_wave_16k("demo.wav", soundfile_module)
        output = feature_hubert.extract_hubert_features(
            FakeModel(),
            waveform,
            "v1",
            "cpu",
            False,
            normalize=True,
        )

        self.assertEqual(tuple(output.shape), (1, 2))


class F0HelperTest(unittest.TestCase):
    def test_supported_f0_methods_include_crepe(self):
        self.assertIn("crepe", feature_f0.F0_METHODS)

    def test_compute_f0_by_method_uses_existing_rmvpe_model(self):
        model = mock.Mock()
        model.infer_from_audio.return_value = np.array([123.0], dtype=np.float32)

        with mock.patch("src.features.f0.load_rmvpe_model") as load_model:
            f0, returned_model = feature_f0.compute_f0_by_method(
                np.ones(10, dtype=np.float32),
                16000,
                1,
                160,
                "rmvpe",
                device="cpu",
                is_half=False,
                rmvpe_model=model,
            )

        load_model.assert_not_called()
        model.infer_from_audio.assert_called_once()
        self.assertIs(returned_model, model)
        np.testing.assert_array_equal(f0, np.array([123.0], dtype=np.float32))

    def test_compute_f0_by_method_loads_rmvpe_model_when_missing(self):
        model = mock.Mock()
        model.infer_from_audio.return_value = np.array([456.0], dtype=np.float32)

        with mock.patch("src.features.f0.load_rmvpe_model", return_value=model) as load_model:
            f0, returned_model = feature_f0.compute_f0_by_method(
                np.ones(10, dtype=np.float32),
                16000,
                1,
                160,
                "rmvpe",
                device="cpu",
                is_half=False,
                rmvpe_path="custom_rmvpe.pt",
                log_fn=print,
            )

        load_model.assert_called_once_with("custom_rmvpe.pt", "cpu", False, print)
        self.assertIs(returned_model, model)
        np.testing.assert_array_equal(f0, np.array([456.0], dtype=np.float32))

    def test_compute_f0_by_method_dispatches_world_methods(self):
        with mock.patch(
            "src.features.f0.compute_world_f0",
            return_value=np.array([1.0, 2.0], dtype=np.float32),
        ) as compute_world:
            f0, returned_model = feature_f0.compute_f0_by_method(
                np.ones(10, dtype=np.float32),
                16000,
                1,
                160,
                "harvest",
            )

        compute_world.assert_called_once_with(
            mock.ANY,
            16000,
            160,
            "harvest",
            f0_min=feature_f0.F0_MIN,
            f0_max=feature_f0.F0_MAX,
        )
        self.assertIsNone(returned_model)
        np.testing.assert_array_equal(f0, np.array([1.0, 2.0], dtype=np.float32))

    def test_compute_f0_by_method_dispatches_crepe(self):
        with mock.patch(
            "src.features.f0.compute_crepe_f0",
            return_value=np.array([3.0, 4.0], dtype=np.float32),
        ) as compute_crepe:
            f0, returned_model = feature_f0.compute_f0_by_method(
                np.ones(10, dtype=np.float32),
                16000,
                1,
                160,
                "crepe",
                device="cuda:0",
            )

        compute_crepe.assert_called_once_with(
            mock.ANY,
            16000,
            160,
            "cuda:0",
            f0_min=feature_f0.F0_MIN,
            f0_max=feature_f0.F0_MAX,
        )
        self.assertIsNone(returned_model)
        np.testing.assert_array_equal(f0, np.array([3.0, 4.0], dtype=np.float32))

    def test_f0_to_coarse_returns_int_bins(self):
        coarse = feature_f0.f0_to_coarse(np.array([110.0, 220.0], dtype=np.float32))

        self.assertEqual(coarse.dtype, np.int32)
        self.assertTrue(((coarse >= 1) & (coarse <= 255)).all())

    def test_compute_f0_by_method_rejects_unknown_method(self):
        with self.assertRaisesRegex(ValueError, "Unsupported f0 method"):
            feature_f0.compute_f0_by_method(
                np.ones(10, dtype=np.float32),
                16000,
                1,
                160,
                "unknown",
            )


if __name__ == "__main__":
    unittest.main()
