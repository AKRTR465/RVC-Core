import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.index import cli as index_cli
from src.index.common import SOURCE_MATRIX_NAME, load_source_matrix, save_source_matrix
from src.index.retrieval import load_retrieval_index
from tests.equivalence_helpers import make_temp_dir, patched_argv


def write_project_config(path, work_dir, *, version, sample_rate="48k", n_cpu=6):
    root = work_dir.parents[1]
    content = f"""base_config: mute.yaml
name: unit_index
work_dir: {work_dir.as_posix()}
data_root: {(root / "data").as_posix()}
ckpt_root: {(root / "ckpt").as_posix()}
pretrain_root: {(root / "pretrain").as_posix()}
selectors:
  version: {version}
  sample_rate: {sample_rate}
  if_f0: 1
runtime:
  device: cpu
  is_half: false
  n_cpu: {n_cpu}
model:
  spk_embed_dim: 1
"""
    Path(path).write_text(content, encoding="utf-8")


def fake_runtime_profile():
    return {
        "device": "cpu",
        "device_request": "cpu",
        "gpu_name": None,
        "gpu_mem_gb": None,
        "supports_half": False,
    }


class IndexCliTest(unittest.TestCase):
    def test_parse_args_config_mode_v1_uses_project_paths(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "unit_index"
            write_project_config(config_path, work_dir, version="v1")

            with patched_argv(["index", "--config", str(config_path)]), mock.patch(
                "configs.project_config._detect_runtime_environment",
                return_value=fake_runtime_profile(),
            ):
                request = index_cli.parse_args()

        self.assertEqual(request.feature_dim, 256)
        self.assertIsNone(request.n_cpu)
        self.assertEqual(
            request.inp_root,
            tmp_path / "data" / "unit_index" / "preprocess_data" / "3_feature256",
        )
        self.assertEqual(request.index_dir, work_dir / "index")
        self.assertEqual(request.output, work_dir / "index" / "unit_index.index")

    def test_parse_args_config_mode_v2_uses_runtime_n_cpu(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "unit_index"
            write_project_config(config_path, work_dir, version="v2", n_cpu=3)

            with patched_argv(["index", "--config", str(config_path)]), mock.patch(
                "configs.project_config._detect_runtime_environment",
                return_value=fake_runtime_profile(),
            ):
                request = index_cli.parse_args()

        self.assertEqual(request.feature_dim, 768)
        self.assertEqual(request.n_cpu, 3)
        self.assertEqual(
            request.inp_root,
            tmp_path / "data" / "unit_index" / "preprocess_data" / "3_feature768",
        )

    def test_parse_args_manual_mode_uses_feature_dim_and_parent_index_dir(self):
        with patched_argv(
            [
                "index",
                "-i",
                "features",
                "-o",
                "artifacts/demo.index",
                "--feature-dim",
                "768",
                "-n",
                "4",
            ]
        ):
            request = index_cli.parse_args()

        self.assertEqual(request.inp_root, Path("features"))
        self.assertEqual(request.output, Path("artifacts/demo.index"))
        self.assertEqual(request.index_dir, Path("artifacts/demo.index").resolve().parent)
        self.assertEqual(request.feature_dim, 768)
        self.assertEqual(request.n_cpu, 4)

    def test_main_builds_with_parsed_request(self):
        request = object()
        with mock.patch("src.index.cli.parse_args", return_value=request), mock.patch(
            "src.index.cli.build_index"
        ) as build_index:
            index_cli.main()

        build_index.assert_called_once_with(request)

    def test_source_matrix_contract_is_shared(self):
        with make_temp_dir() as tmp:
            index_dir = Path(tmp)
            matrix = np.ones((2, 3), dtype=np.float32)
            save_source_matrix(index_dir, matrix)

            loaded = load_source_matrix(index_dir, feature_dim=3)
            self.assertTrue((index_dir / SOURCE_MATRIX_NAME).is_file())

        np.testing.assert_array_equal(loaded, matrix)

    def test_load_retrieval_index_uses_shared_source_matrix_loader(self):
        fake_index = mock.Mock(ntotal=2, d=3)
        with make_temp_dir() as tmp:
            index_path = Path(tmp) / "demo.index"
            index_path.write_bytes(b"idx")
            save_source_matrix(index_path.parent, np.ones((2, 3), dtype=np.float32))
            with mock.patch("faiss.read_index", return_value=fake_index):
                index, source = load_retrieval_index(index_path)

        self.assertIs(index, fake_index)
        self.assertEqual(tuple(source.shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
