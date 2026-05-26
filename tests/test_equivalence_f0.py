import importlib
import unittest
from pathlib import Path

import numpy as np

from tests.equivalence_helpers import REPO_ROOT, make_temp_dir


@unittest.skipUnless((REPO_ROOT / "infer").exists(), "legacy infer tree removed")
class F0EquivalenceTest(unittest.TestCase):
    def test_coarse_f0_matches_infer(self):
        old_f0 = importlib.import_module("infer.modules.train.extract_f0_print")
        new_f0 = importlib.import_module("src.preprocess.f0")

        f0 = np.array([0.0, 50.0, 110.0, 220.0, 440.0, 880.0], dtype=np.float32)
        np.testing.assert_array_equal(
            old_f0.FeatureInput().coarse_f0(f0),
            new_f0.FeatureInput().coarse_f0(f0),
        )

    def test_path_planning_matches_infer(self):
        old_f0 = importlib.import_module("infer.modules.train.extract_f0_print")
        new_f0 = importlib.import_module("src.preprocess.f0")

        with make_temp_dir() as tmp:
            exp_dir = Path(tmp)
            wav_dir = exp_dir / "1_16k_wavs"
            wav_dir.mkdir(parents=True)
            (wav_dir / "a.wav").write_bytes(b"")
            (wav_dir / "spec_skip.wav").write_bytes(b"")
            self.assertEqual(
                old_f0.build_paths(str(exp_dir)),
                new_f0.build_paths(str(exp_dir)),
            )


if __name__ == "__main__":
    unittest.main()
