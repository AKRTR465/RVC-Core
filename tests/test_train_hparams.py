import unittest
from pathlib import Path
from unittest import mock

from src.train import utils as train_utils
from tests.equivalence_helpers import make_temp_dir, patched_argv


def write_project_config(path, work_dir, extra=""):
    root = work_dir.parent
    content = f"""base_config: mute.yaml
name: unit_hparams
work_dir: {work_dir.as_posix()}
data_root: {(root / "data").as_posix()}
ckpt_root: {(root / "ckpt").as_posix()}
pretrain_root: {(root / "pretrain").as_posix()}
model:
  spk_embed_dim: 1
"""
    if extra:
        content += extra.lstrip("\n")
        if not content.endswith("\n"):
            content += "\n"
    Path(path).write_text(content, encoding="utf-8")


def fake_runtime_profile():
    return {
        "device": "cpu",
        "device_request": "cpu",
        "gpu_name": None,
        "gpu_mem_gb": None,
        "supports_half": False,
    }


class TrainHparamsTest(unittest.TestCase):
    def test_get_hparams_applies_cli_overrides_and_saves_snapshot(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "unit_hparams"
            write_project_config(config_path, work_dir)

            with patched_argv(
                [
                    "train",
                    "--config",
                    str(config_path),
                    "-se",
                    "5",
                    "-te",
                    "300",
                    "-bs",
                    "2",
                    "-pg",
                    "pretrain/G.pth",
                    "-pd",
                    "pretrain/D.pth",
                    "-l",
                    "1",
                    "-c",
                    "0",
                    "-sw",
                    "1",
                ]
            ), mock.patch(
                "configs.project_config._detect_runtime_environment",
                return_value=fake_runtime_profile(),
            ):
                hps = train_utils.get_hparams()

        self.assertEqual(hps.save_every_epoch, 5)
        self.assertEqual(hps.total_epoch, 300)
        self.assertEqual(hps.train.batch_size, 2)
        self.assertEqual(hps.train.pretrainG, "pretrain/G.pth")
        self.assertEqual(hps.train.pretrainD, "pretrain/D.pth")
        self.assertEqual(hps.if_latest, 1)
        self.assertEqual(hps.if_cache_data_in_gpu, 0)
        self.assertEqual(hps.save_every_weights, "1")
        self.assertTrue(hps.export_dir.endswith("export"))
        self.assertTrue((work_dir / "config.yaml").exists())
        self.assertTrue(hps.data.training_files.endswith("train_filelist.txt"))
        self.assertTrue(hps.data.validation_files.endswith("val_filelist.txt"))


if __name__ == "__main__":
    unittest.main()
