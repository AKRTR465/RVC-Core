import importlib
import unittest
from pathlib import Path

import numpy as np

from tests.equivalence_helpers import (
    collect_binary_tree,
    fake_librosa,
    make_temp_dir,
)


class PreprocessEquivalenceTest(unittest.TestCase):
    def test_audio_preprocess_outputs_match_infer(self):
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False)
        audio = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

        def fake_load_audio(path, target_sr):
            self.assertEqual(target_sr, sr)
            return audio.copy()

        with fake_librosa():
            import sys

            sys.modules.pop("infer.modules.train.preprocess", None)
            sys.modules.pop("src.preprocess.audio", None)
            old_preprocess = importlib.import_module("infer.modules.train.preprocess")
            new_preprocess = importlib.import_module("src.preprocess.audio")

        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            dataset = tmp_path / "dataset"
            old_out = tmp_path / "old"
            new_out = tmp_path / "new"
            dataset.mkdir()
            (dataset / "sample.wav").write_bytes(b"placeholder")

            old_load_audio = old_preprocess.load_audio
            new_load_audio = new_preprocess.load_audio
            old_preprocess.load_audio = fake_load_audio
            new_preprocess.load_audio = fake_load_audio
            try:
                old_preprocess.preprocess_trainset(
                    str(dataset), sr, 1, str(old_out), True, 1.0
                )
                new_preprocess.preprocess_trainset(
                    str(dataset), sr, 1, str(new_out), True, 1.0
                )
            finally:
                old_preprocess.load_audio = old_load_audio
                new_preprocess.load_audio = new_load_audio

            self.assertEqual(
                collect_binary_tree(old_out, ignored_names={"preprocess.log"}),
                collect_binary_tree(new_out, ignored_names={"preprocess.log"}),
            )


if __name__ == "__main__":
    unittest.main()
