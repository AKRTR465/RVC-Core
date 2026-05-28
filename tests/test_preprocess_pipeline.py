import unittest
from pathlib import Path
from unittest import mock

from src.preprocess.layout import PreprocessLayout
from src.preprocess.pipeline import (
    DatasetItem,
    build_arg_parser,
    discover_dataset_items,
    generate_filelist,
    parse_stage_list,
    run_audio_stage,
    run_f0_stage,
    run_feature_stage,
    run_pipeline,
    split_filelist_rows,
    write_preprocess_manifest,
)
from src.rvc_profiles import get_feature_profile
from tests.equivalence_helpers import make_temp_dir


def touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def project_for(
    tmp_root,
    preprocess_dir,
    if_f0=1,
    version="v1",
    validation_split=0.5,
    validation_seed=1234,
):
    return {
        "paths": {
            "dataset_dir": str(Path(tmp_root) / "dataset"),
            "preprocess_dir": str(preprocess_dir),
            "pretrain_root": str(Path(tmp_root) / "pretrain"),
            "hubert_path": str(Path(tmp_root) / "pretrain" / "hubert" / "hubert_base.pt"),
        },
        "selectors": {
            "version": version,
            "if_f0": if_f0,
        },
        "runtime": {
            "device": "cpu",
            "is_half": False,
            "n_cpu": 2,
        },
        "data": {
            "sampling_rate": 48000,
        },
        "model": {
            "spk_embed_dim": 2,
        },
        "preprocess": {
            "validation_split": validation_split,
            "validation_seed": validation_seed,
            "noparallel": True,
            "f0method": "rmvpe",
        },
    }


class PreprocessPipelineTest(unittest.TestCase):
    def test_parse_stage_list_defaults_and_deduplicates(self):
        self.assertEqual(parse_stage_list(""), ("audio", "f0", "features", "filelist"))
        self.assertEqual(parse_stage_list("audio,f0,audio"), ("audio", "f0"))

    def test_flat_dataset_uses_single_speaker_zero(self):
        with make_temp_dir() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            touch(dataset_dir / "a.wav")
            touch(dataset_dir / "b.mp3")
            touch(dataset_dir / "ignore.txt")

            items = discover_dataset_items(dataset_dir, spk_embed_dim=1)

            self.assertEqual([item.speaker_id for item in items], [0, 0])
            self.assertEqual([item.index for item in items], [0, 1])

    def test_numbered_speaker_dirs_convert_to_zero_based_sids(self):
        with make_temp_dir() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            touch(dataset_dir / "1" / "a.wav")
            touch(dataset_dir / "2" / "b.wav")

            items = discover_dataset_items(dataset_dir, spk_embed_dim=2)

            self.assertEqual([item.speaker_id for item in items], [0, 1])
            self.assertEqual([item.index for item in items], [0, 1])

    def test_split_filelist_rows_is_stable_and_grouped_by_sid(self):
        rows = [
            "a.wav|a.npy|0",
            "b.wav|b.npy|0",
            "c.wav|c.npy|0",
            "d.wav|d.npy|1",
            "e.wav|e.npy|1",
        ]

        train_a, val_a = split_filelist_rows(rows, validation_split=0.5, validation_seed=1234)
        train_b, val_b = split_filelist_rows(rows, validation_split=0.5, validation_seed=1234)

        self.assertEqual(train_a, train_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(sorted(train_a + val_a), sorted(rows))
        self.assertEqual(sorted(line.split("|")[-1] for line in val_a), ["0", "1"])

    def test_filelist_columns_follow_if_f0_and_write_split_files(self):
        with make_temp_dir() as tmp:
            tmp_root = Path(tmp)
            preprocess_dir = tmp_root / "preprocess"
            layout = PreprocessLayout(preprocess_dir, get_feature_profile("v1"))
            source_a = touch(tmp_root / "dataset" / "a.wav")
            source_b = touch(tmp_root / "dataset" / "b.wav")
            gt_wav = touch(layout.gt_wavs_dir / "0_0.wav")
            wav16k = touch(layout.wav16k_dir / "0_0.wav")
            touch(layout.feature_dir / "0_0.npy")
            touch(layout.f0_dir / "0_0.wav.npy")
            touch(layout.f0nsf_dir / "0_0.wav.npy")
            touch(layout.gt_wavs_dir / "1_0.wav")
            touch(layout.wav16k_dir / "1_0.wav")
            touch(layout.feature_dir / "1_0.npy")
            touch(layout.f0_dir / "1_0.wav.npy")
            touch(layout.f0nsf_dir / "1_0.wav.npy")

            write_preprocess_manifest(
                [
                    DatasetItem(source_a.resolve(), speaker_id=0, index=0),
                    DatasetItem(source_b.resolve(), speaker_id=0, index=1),
                ],
                preprocess_dir,
            )

            filelist_path, row_count, skipped = generate_filelist(
                project_for(tmp_root, preprocess_dir, if_f0=1)
            )
            self.assertEqual((row_count, skipped), (2, 0))
            self.assertEqual(len(filelist_path.read_text(encoding="utf-8").splitlines()[0].split("|")), 5)
            self.assertTrue(layout.train_filelist_path.is_file())
            self.assertTrue(layout.val_filelist_path.is_file())

            filelist_path, row_count, skipped = generate_filelist(
                project_for(tmp_root, preprocess_dir, if_f0=0)
            )
            self.assertEqual((row_count, skipped), (2, 0))
            self.assertEqual(len(filelist_path.read_text(encoding="utf-8").splitlines()[0].split("|")), 3)
            self.assertTrue(gt_wav.is_file())
            self.assertTrue(wav16k.is_file())

    def test_filelist_requires_manifest(self):
        with make_temp_dir() as tmp:
            tmp_root = Path(tmp)
            preprocess_dir = tmp_root / "preprocess"
            with self.assertRaisesRegex(FileNotFoundError, "Run the audio stage"):
                generate_filelist(project_for(tmp_root, preprocess_dir, if_f0=0))

    def test_run_audio_stage_dispatches_workers_and_writes_manifest(self):
        with make_temp_dir() as tmp:
            tmp_root = Path(tmp)
            project = project_for(tmp_root, tmp_root / "preprocess")
            touch(tmp_root / "dataset" / "0.wav")
            touch(tmp_root / "dataset" / "1.wav")

            with mock.patch("src.preprocess.pipeline.run_worker_shards") as run_shards, mock.patch(
                "src.preprocess.pipeline._write_project_manifest",
                return_value=tmp_root / "preprocess" / "preprocess_manifest.jsonl",
            ) as write_manifest:
                manifest_path = run_audio_stage(project, workers_override=3)

            run_shards.assert_called_once()
            write_manifest.assert_called_once()
            self.assertTrue(str(manifest_path).endswith("preprocess_manifest.jsonl"))

    def test_run_f0_stage_supports_crepe_and_device_override(self):
        with make_temp_dir() as tmp:
            tmp_root = Path(tmp)
            project = project_for(tmp_root, tmp_root / "preprocess")

            with mock.patch(
                "src.preprocess.f0.resolve_runtime",
                return_value=("cuda:0", False),
            ) as resolve_runtime, mock.patch(
                "src.preprocess.f0.build_paths",
                return_value=[["a.wav", "a", "b"]],
            ), mock.patch("src.preprocess.pipeline.run_worker_shards") as run_shards:
                run_f0_stage(
                    project,
                    "crepe",
                    workers_override=2,
                    device_override="cuda:0",
                )

            resolve_runtime.assert_called_once()
            run_shards.assert_called_once()

    def test_run_feature_stage_uses_layout_and_device_resolution(self):
        with make_temp_dir() as tmp:
            tmp_root = Path(tmp)
            project = project_for(tmp_root, tmp_root / "preprocess", version="v2")

            with mock.patch(
                "src.preprocess.features.resolve_device",
                return_value="cpu",
            ) as resolve_device, mock.patch(
                "src.preprocess.features.extract_features"
            ) as extract_features:
                run_feature_stage(project)

            resolve_device.assert_called_once_with("cpu")
            extract_features.assert_called_once()
            self.assertEqual(
                extract_features.call_args.kwargs["layout"].feature_profile.version,
                "v2",
            )

    def test_run_pipeline_dispatches_requested_stages(self):
        project = {"dummy": True}
        with mock.patch("src.preprocess.pipeline.run_audio_stage") as run_audio, mock.patch(
            "src.preprocess.pipeline.run_f0_stage"
        ) as run_f0, mock.patch(
            "src.preprocess.pipeline.run_feature_stage"
        ) as run_feature, mock.patch(
            "src.preprocess.pipeline.generate_filelist"
        ) as generate_filelist:
            run_pipeline(project, ("audio", "features"), "rmvpe", 2)

        run_audio.assert_called_once_with(project, 2)
        run_feature.assert_called_once_with(project, device_override=None, is_half_override=None)
        run_f0.assert_not_called()
        generate_filelist.assert_not_called()

    def test_build_arg_parser_accepts_device_override(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--config", "project.yaml", "--device", "cuda:0"])
        self.assertEqual(args.device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
