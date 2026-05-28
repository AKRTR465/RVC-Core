import unittest
from pathlib import Path

from configs.project_config import HparamsParseError, load_project_config, parse_hparams_overrides
from tests.equivalence_helpers import make_temp_dir


class ProjectLoadingTest(unittest.TestCase):
    def test_parse_hparams_overrides_parses_scalars_and_quoted_commas(self):
        overrides = parse_hparams_overrides(
            'infer.formant="a,b",train.batch_size=2,runtime.disable_tf32=true,train.grad_scaler_init_scale=16.0'
        )

        self.assertEqual(overrides["infer"]["formant"], "a,b")
        self.assertEqual(overrides["train"]["batch_size"], 2)
        self.assertTrue(overrides["runtime"]["disable_tf32"])
        self.assertEqual(overrides["train"]["grad_scaler_init_scale"], 16.0)

    def test_parse_hparams_overrides_rejects_conflicting_keys(self):
        with self.assertRaisesRegex(HparamsParseError, "Conflicting --hparams key"):
            parse_hparams_overrides("train=1,train.batch_size=2")

    def test_load_project_config_rejects_base_config_cycle(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_dir = tmp_path / "configs"
            config_dir.mkdir(parents=True, exist_ok=True)
            work_dir = tmp_path / "ckpt" / "unit_voice"
            root = work_dir.parent

            (config_dir / "base_a.yaml").write_text(
                "base_config: ./base_b.yaml\n",
                encoding="utf-8",
            )
            (config_dir / "base_b.yaml").write_text(
                "base_config: ./base_a.yaml\n",
                encoding="utf-8",
            )
            (config_dir / "project.yaml").write_text(
                f"""base_config: ./base_a.yaml
name: unit_voice
work_dir: {work_dir.as_posix()}
data_root: {(root / "data").as_posix()}
ckpt_root: {(root / "ckpt").as_posix()}
pretrain_root: {(root / "pretrain").as_posix()}
model:
  spk_embed_dim: 1
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Detected base_config cycle"):
                load_project_config(config_dir / "project.yaml", reset=True)

    def test_load_project_config_keeps_nested_runtime_and_selector_sections(self):
        with make_temp_dir() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "project.yaml"
            work_dir = tmp_path / "ckpt" / "unit_voice"
            root = work_dir.parent

            config_path.write_text(
                f"""base_config: mute.yaml
name: unit_voice
work_dir: {work_dir.as_posix()}
data_root: {(root / "data").as_posix()}
ckpt_root: {(root / "ckpt").as_posix()}
pretrain_root: {(root / "pretrain").as_posix()}
selectors:
  version: v2
  sample_rate: 48k
  if_f0: 1
runtime:
  device: cpu
  is_half: false
  n_cpu: 2
model:
  spk_embed_dim: 1
""",
                encoding="utf-8",
            )

            config = load_project_config(config_path, reset=True)

        self.assertEqual(config["selectors"]["version"], "v2")
        self.assertEqual(config["runtime"]["device"], "cpu")
        self.assertTrue(config["paths"]["preprocess_dir"].endswith("preprocess_data"))
        self.assertTrue(config["paths"]["training_files"].endswith("train_filelist.txt"))
        self.assertNotIn("version", config)
        self.assertNotIn("sample_rate", config)
        self.assertNotIn("if_f0", config)
        self.assertNotIn("device", config)
        self.assertNotIn("is_half", config)
        self.assertNotIn("n_cpu", config)
        self.assertNotIn("feature_dim", config)
        self.assertNotIn("feature_dir", config)
        self.assertNotIn("preprocess_dir", config)
        self.assertNotIn("hubert_path", config)
        self.assertNotIn("training_files", config)
        self.assertNotIn("validation_files", config)


if __name__ == "__main__":
    unittest.main()
