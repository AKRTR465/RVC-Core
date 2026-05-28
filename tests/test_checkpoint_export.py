import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from src.train.checkpoint_export import savee
from tests.equivalence_helpers import make_temp_dir


def build_hps(export_dir):
    return SimpleNamespace(
        export_dir=export_dir,
        data=SimpleNamespace(
            filter_length=1024,
            sampling_rate=40000,
        ),
        model=SimpleNamespace(
            inter_channels=192,
            hidden_channels=192,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[10, 10, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            spk_embed_dim=2,
            gin_channels=256,
        ),
    )


class CheckpointExportTest(unittest.TestCase):
    def test_savee_filters_enc_q_and_writes_to_hps_export_dir(self):
        with make_temp_dir() as tmp:
            export_dir = Path(tmp) / "export"
            hps = build_hps(str(export_dir))

            result = savee(
                {
                    "enc_q.proj.weight": torch.ones(1, dtype=torch.float32),
                    "emb_g.weight": torch.ones((2, 2), dtype=torch.float32),
                },
                48000,
                1,
                "demo_model",
                3,
                "v2",
                hps,
            )

            self.assertEqual(result, "Success.")
            output_path = export_dir / "demo_model.pth"
            self.assertTrue(output_path.is_file())
            payload = torch.load(output_path, map_location="cpu")
            self.assertNotIn("enc_q.proj.weight", payload["weight"])
            self.assertEqual(payload["weight"]["emb_g.weight"].dtype, torch.float16)
            self.assertEqual(payload["info"], "3epoch")
            self.assertEqual(payload["config"][-1], 40000)

    def test_savee_falls_back_to_ckpt_root_with_guessed_project_dir(self):
        with make_temp_dir() as tmp:
            ckpt_root = Path(tmp) / "ckpt_root"
            hps = build_hps(None)

            with mock.patch.dict(os.environ, {"ckpt_root": str(ckpt_root)}):
                result = savee(
                    {
                        "emb_g.weight": torch.ones((2, 2), dtype=torch.float32),
                    },
                    48000,
                    1,
                    "demo_model_e3_s4",
                    3,
                    "v2",
                    hps,
                )

            self.assertEqual(result, "Success.")
            output_path = ckpt_root / "demo_model" / "export" / "demo_model_e3_s4.pth"
            self.assertTrue(output_path.is_file())
            payload = torch.load(output_path, map_location="cpu")
            self.assertEqual(payload["version"], "v2")


if __name__ == "__main__":
    unittest.main()
