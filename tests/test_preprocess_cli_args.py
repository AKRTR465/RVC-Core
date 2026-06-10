import io
import unittest
from contextlib import redirect_stderr
from unittest import mock

from src.preprocess import pipeline as preprocess_pipeline
from tests.equivalence_helpers import patched_argv


class PreprocessCliArgsTest(unittest.TestCase):
    def assert_parse_error(self, argv, pattern):
        stderr = io.StringIO()
        with (
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exc,
        ):
            preprocess_pipeline.main(argv)
        self.assertEqual(exc.exception.code, 2)
        self.assertRegex(stderr.getvalue(), pattern)

    def test_main_accepts_config_mode_and_rmvpe_override(self):
        project = {
            "paths": {
                "dataset_dir": "data/dataset",
                "preprocess_dir": "data/preprocess",
                "pretrain_root": "pretrain",
                "hubert_path": "pretrain/hubert/hubert_base.pt",
            },
            "data": {"sampling_rate": 48000},
            "runtime": {"n_cpu": 6, "device": "cpu", "is_half": False},
            "model": {"spk_embed_dim": 1},
            "selectors": {"version": "v2", "if_f0": 1},
            "preprocess": {"noparallel": True, "f0method": "rmvpe"},
        }
        with mock.patch(
            "src.preprocess.pipeline.load_project_config",
            return_value=project,
        ), mock.patch("src.preprocess.pipeline.run_pipeline") as run_pipeline:
            preprocess_pipeline.main(
                [
                    "--config",
                    "project.yaml",
                    "--f0method",
                    "rmvpe",
                    "--workers",
                    "3",
                    "--device",
                    "cpu",
                ]
            )

        run_pipeline.assert_called_once_with(
            project,
            preprocess_pipeline.DEFAULT_STAGES,
            "rmvpe",
            3,
            device_override="cpu",
            is_half_override=None,
        )

    def test_main_rejects_removed_crepe_f0_method(self):
        self.assert_parse_error(
            ["--config", "project.yaml", "--f0method", "crepe"],
            r"invalid choice: 'crepe'",
        )

    def test_main_allows_is_half_override(self):
        project = {
            "paths": {
                "dataset_dir": "data/dataset",
                "preprocess_dir": "data/preprocess",
                "pretrain_root": "pretrain",
                "hubert_path": "pretrain/hubert/hubert_base.pt",
            },
            "data": {"sampling_rate": 48000},
            "runtime": {"n_cpu": 6, "device": "cuda:0", "is_half": False},
            "model": {"spk_embed_dim": 1},
            "selectors": {"version": "v2", "if_f0": 1},
            "preprocess": {"noparallel": True, "f0method": "rmvpe"},
        }
        with mock.patch(
            "src.preprocess.pipeline.load_project_config",
            return_value=project,
        ), mock.patch("src.preprocess.pipeline.run_pipeline") as run_pipeline:
            preprocess_pipeline.main(["--config", "project.yaml", "--is-half"])

        run_pipeline.assert_called_once_with(
            project,
            preprocess_pipeline.DEFAULT_STAGES,
            "rmvpe",
            None,
            device_override=None,
            is_half_override=True,
        )

    def test_main_rejects_invalid_stage_list(self):
        self.assert_parse_error(
            ["--config", "project.yaml", "--stages", "audio,unknown"],
            r"Unsupported stage\(s\): unknown",
        )

    def test_main_rejects_invalid_workers(self):
        self.assert_parse_error(
            ["--config", "project.yaml", "--workers", "0"],
            r"--workers must be >= 1",
        )


if __name__ == "__main__":
    unittest.main()
