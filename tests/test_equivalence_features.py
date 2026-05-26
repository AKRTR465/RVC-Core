import os
import runpy
import unittest
from pathlib import Path

import numpy as np

from tests.equivalence_helpers import (
    REPO_ROOT,
    fake_fairseq,
    make_temp_dir,
    patched_argv,
    write_sine_wav,
)


@unittest.skipUnless((REPO_ROOT / "infer").exists(), "legacy infer tree removed")
class FeatureEquivalenceTest(unittest.TestCase):
    def test_feature_worker_outputs_match_infer_with_fake_hubert(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            old_exp = tmp_path / "old"
            new_exp = tmp_path / "new"
            for exp_dir in (old_exp, new_exp):
                wav_dir = exp_dir / "1_16k_wavs"
                wav_dir.mkdir(parents=True)
                write_sine_wav(wav_dir / "sample.wav", seconds=0.2)

            pretrain_root = tmp_path / "pretrain"
            model_path = pretrain_root / "hubert" / "hubert_base.pt"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"fake")

            old_env = os.environ.get("pretrain_root")
            os.environ["pretrain_root"] = str(pretrain_root)
            try:
                with fake_fairseq():
                    with patched_argv(
                        [
                            "extract_feature_print.py",
                            "cpu",
                            "1",
                            "0",
                            str(old_exp),
                            "v1",
                            "false",
                        ]
                    ):
                        runpy.run_path(
                            str(REPO_ROOT / "infer/modules/train/extract_feature_print.py"),
                            run_name="__main__",
                        )

                    with patched_argv(
                        [
                            "features.py",
                            "cpu",
                            "1",
                            "0",
                            str(new_exp),
                            "v1",
                            "false",
                        ]
                    ):
                        runpy.run_module("src.preprocess.features", run_name="__main__")
            finally:
                if old_env is None:
                    os.environ.pop("pretrain_root", None)
                else:
                    os.environ["pretrain_root"] = old_env

            np.testing.assert_array_equal(
                np.load(old_exp / "3_feature256" / "sample.npy"),
                np.load(new_exp / "3_feature256" / "sample.npy"),
            )


if __name__ == "__main__":
    unittest.main()
