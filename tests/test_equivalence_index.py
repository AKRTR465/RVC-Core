import importlib
import runpy
import unittest
from pathlib import Path

import numpy as np

from tests.equivalence_helpers import REPO_ROOT, make_temp_dir, patched_argv


class IndexEquivalenceTest(unittest.TestCase):
    def test_v1_index_outputs_match_infer(self):
        faiss = importlib.import_module("faiss")

        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            old_features = tmp_path / "old_features_v1"
            new_features = tmp_path / "new_features_v1"
            old_features.mkdir()
            new_features.mkdir()
            rng = np.random.default_rng(123)
            for idx in range(3):
                matrix = rng.random((20, 256), dtype=np.float32)
                np.save(old_features / f"{idx}.npy", matrix)
                np.save(new_features / f"{idx}.npy", matrix)

            old_index = tmp_path / "old_v1" / "test.index"
            new_index = tmp_path / "new_v1" / "test.index"
            with patched_argv(
                ["train-index.py", "-i", str(old_features), "-o", str(old_index)]
            ):
                runpy.run_path(
                    str(REPO_ROOT / "infer/index/train-index.py"),
                    run_name="__main__",
                )
            with patched_argv(
                ["build_v1.py", "-i", str(new_features), "-o", str(new_index)]
            ):
                runpy.run_module("src.index.build_v1", run_name="__main__")

            np.testing.assert_array_equal(
                np.load(old_index.parent / "big_src_feature.npy"),
                np.load(new_index.parent / "big_src_feature.npy"),
            )
            old_faiss = faiss.read_index(str(old_index))
            new_faiss = faiss.read_index(str(new_index))
            self.assertEqual(
                (old_faiss.d, old_faiss.ntotal),
                (new_faiss.d, new_faiss.ntotal),
            )

    def test_v2_index_outputs_match_infer(self):
        faiss = importlib.import_module("faiss")

        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            old_features = tmp_path / "old_features_v2"
            new_features = tmp_path / "new_features_v2"
            old_features.mkdir()
            new_features.mkdir()
            rng = np.random.default_rng(456)
            for idx in range(2):
                matrix = rng.random((40, 768), dtype=np.float32)
                np.save(old_features / f"{idx}.npy", matrix)
                np.save(new_features / f"{idx}.npy", matrix)

            old_index = tmp_path / "old_v2" / "test.index"
            new_index = tmp_path / "new_v2" / "test.index"
            np.random.seed(789)
            with patched_argv(
                [
                    "train-index-v2.py",
                    "-i",
                    str(old_features),
                    "-o",
                    str(old_index),
                    "-n",
                    "1",
                ]
            ):
                runpy.run_path(
                    str(REPO_ROOT / "infer/index/train-index-v2.py"),
                    run_name="__main__",
                )
            np.random.seed(789)
            with patched_argv(
                ["build_v2.py", "-i", str(new_features), "-o", str(new_index), "-n", "1"]
            ):
                runpy.run_module("src.index.build_v2", run_name="__main__")

            np.testing.assert_array_equal(
                np.load(old_index.parent / "big_src_feature.npy"),
                np.load(new_index.parent / "big_src_feature.npy"),
            )
            old_faiss = faiss.read_index(str(old_index))
            new_faiss = faiss.read_index(str(new_index))
            self.assertEqual(
                (old_faiss.d, old_faiss.ntotal),
                (new_faiss.d, new_faiss.ntotal),
            )


if __name__ == "__main__":
    unittest.main()
