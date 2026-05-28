import io
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from src.preprocess import audio as audio_stage
from src.preprocess import f0 as f0_stage
from src.preprocess import features as feature_stage
from tests.equivalence_helpers import patched_argv


class PreprocessCliArgsTest(unittest.TestCase):
    def assert_parse_error(self, fn, argv, pattern):
        stderr = io.StringIO()
        with (
            patched_argv(argv),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exc,
        ):
            fn()
        self.assertEqual(exc.exception.code, 2)
        self.assertRegex(stderr.getvalue(), pattern)

    def test_audio_parse_args_accepts_config_mode(self):
        project = {
            "paths": {
                "dataset_dir": "data/dataset",
                "preprocess_dir": "data/preprocess",
            },
            "data": {"sampling_rate": 48000},
            "runtime": {"n_cpu": 6},
            "preprocess": {"noparallel": True},
        }
        with (
            patched_argv(["audio", "--config", "project.yaml"]),
            mock.patch("src.preprocess.audio.load_project_config", return_value=project),
        ):
            args = audio_stage.parse_args()

        self.assertEqual(
            args,
            ("data/dataset", 48000, 6, "data/preprocess", True),
        )

    def test_audio_parse_args_accepts_named_manual_mode(self):
        with patched_argv(
            [
                "audio",
                "--inp_root",
                "data/raw",
                "--preprocess_dir",
                "data/preprocess",
                "--sample-rate",
                "48000",
                "--n_p",
                "3",
                "--noparallel",
            ]
        ):
            args = audio_stage.parse_args()

        self.assertEqual(
            args,
            ("data/raw", 48000, 3, "data/preprocess", True),
        )

    def test_audio_parse_args_rejects_removed_positional_mode(self):
        self.assert_parse_error(
            audio_stage.parse_args,
            ["audio", "data/raw", "48000", "3", "data/preprocess", "True"],
            r"unrecognized arguments:",
        )

    def test_f0_parse_args_accepts_config_mode(self):
        project = {
            "paths": {"preprocess_dir": "data/preprocess"},
            "preprocess": {"f0method": "harvest"},
            "runtime": {
                "device": "cpu",
                "is_half": True,
                "n_cpu": 4,
            },
        }
        with (
            patched_argv(["f0", "--config", "project.yaml"]),
            mock.patch("src.preprocess.f0.load_project_config", return_value=project),
        ):
            args = f0_stage.parse_args()

        self.assertEqual(args.exp_dir, "data/preprocess")
        self.assertEqual(args.f0method, "harvest")
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.i_gpu, "")
        self.assertTrue(args.is_half)

    def test_f0_parse_args_rejects_removed_positional_mode(self):
        self.assert_parse_error(
            f0_stage.parse_args,
            ["f0", "data/preprocess", "4", "rmvpe"],
            r"unrecognized arguments:",
        )

    def test_features_parse_args_accepts_config_mode(self):
        project = {
            "paths": {
                "preprocess_dir": "data/preprocess",
                "pretrain_root": "pretrain",
            },
            "selectors": {"version": "v2"},
            "runtime": {
                "device": "cpu",
                "is_half": False,
            },
        }
        with (
            patched_argv(["features", "--config", "project.yaml"]),
            mock.patch("src.preprocess.features.load_project_config", return_value=project),
        ):
            args = feature_stage.parse_args()

        self.assertEqual(args.exp_dir, "data/preprocess")
        self.assertEqual(args.version, "v2")
        self.assertEqual(args.device, "cpu")
        self.assertFalse(args.is_half)
        self.assertEqual(args.model_path, str(Path("pretrain") / "hubert" / "hubert_base.pt"))

    def test_features_parse_args_rejects_removed_positional_mode(self):
        self.assert_parse_error(
            feature_stage.parse_args,
            ["features", "cpu", "1", "0", "data/preprocess", "v1", "True"],
            r"unrecognized arguments:",
        )


if __name__ == "__main__":
    unittest.main()
