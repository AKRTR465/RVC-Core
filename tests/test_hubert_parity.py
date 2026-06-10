import os
import unittest
from pathlib import Path

import numpy as np
import torch

from src.features.hubert import extract_hubert_features
from src.features.hubert import load_hubert_model as load_internal_hubert_model
from tests.equivalence_helpers import REPO_ROOT


@unittest.skipUnless(
    os.environ.get("RVC_HUBERT_PARITY") == "1",
    "Set RVC_HUBERT_PARITY=1 to run optional fairseq parity tests.",
)
class HubertFairseqParityTest(unittest.TestCase):
    def setUp(self):
        path = os.environ.get("RVC_HUBERT_PATH")
        self.model_path = Path(path) if path else REPO_ROOT / "pretrain" / "hubert" / "hubert_base.pt"
        if not self.model_path.is_file():
            self.skipTest(f"Missing HuBERT checkpoint: {self.model_path}")

        try:
            import fairseq  # noqa: F401
            from fairseq import checkpoint_utils
        except ImportError as exc:
            self.skipTest(f"fairseq is not installed for parity testing: {exc}")

        models, saved_cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
            [str(self.model_path)],
            suffix="",
        )
        self.fairseq_model = models[0].float().eval()
        self.fairseq_normalize = bool(getattr(saved_cfg.task, "normalize", False))
        self.internal_model, self.internal_cfg = load_internal_hubert_model(
            self.model_path,
            "cpu",
            is_half=False,
        )

    def test_internal_hubert_matches_fairseq_outputs(self):
        waveform = (0.2 * np.sin(np.linspace(0, 64 * np.pi, 16000))).astype(np.float32)

        for version in ("v1", "v2"):
            with self.subTest(version=version):
                expected = extract_hubert_features(
                    self.fairseq_model,
                    waveform,
                    version,
                    "cpu",
                    is_half=False,
                    normalize=self.fairseq_normalize,
                )
                actual = extract_hubert_features(
                    self.internal_model,
                    waveform,
                    version,
                    "cpu",
                    is_half=False,
                    normalize=self.internal_cfg.task.normalize,
                )

                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertTrue(
                    torch.allclose(actual, expected, rtol=1.0e-5, atol=1.0e-5),
                    msg=f"max abs diff={torch.max(torch.abs(actual - expected)).item()}",
                )
