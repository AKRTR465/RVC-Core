import unittest
from pathlib import Path

from src.preprocess.pipeline import (
    DatasetItem,
    discover_dataset_items,
    generate_filelist,
    split_filelist_rows,
    write_preprocess_manifest,
)
from tests.equivalence_helpers import make_temp_dir


def touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def project_for(
    preprocess_dir,
    if_f0=1,
    version="v1",
    validation_split=0.5,
    validation_seed=1234,
):
    return {
        "preprocess_dir": str(preprocess_dir),
        "version": version,
        "if_f0": if_f0,
        "preprocess": {
            "validation_split": validation_split,
            "validation_seed": validation_seed,
        },
    }


class PreprocessPipelineTest(unittest.TestCase):
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

    def test_non_numeric_speaker_dir_fails(self):
        with make_temp_dir() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            touch(dataset_dir / "alice" / "a.wav")

            with self.assertRaisesRegex(ValueError, "positive integers"):
                discover_dataset_items(dataset_dir, spk_embed_dim=2)

    def test_speaker_dir_id_cannot_exceed_embedding_dim(self):
        with make_temp_dir() as tmp:
            dataset_dir = Path(tmp) / "dataset"
            touch(dataset_dir / "3" / "a.wav")

            with self.assertRaisesRegex(ValueError, "exceeds"):
                discover_dataset_items(dataset_dir, spk_embed_dim=2)

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

    def test_split_filelist_rows_fails_when_validation_samples_cannot_be_created(self):
        rows = [
            "a.wav|a.npy|0",
            "b.wav|b.npy|1",
        ]

        with self.assertRaisesRegex(RuntimeError, "Validation split produced no validation samples"):
            split_filelist_rows(rows, validation_split=0.5, validation_seed=1234)

    def test_filelist_columns_follow_if_f0_and_write_split_files(self):
        with make_temp_dir() as tmp:
            preprocess_dir = Path(tmp) / "preprocess"
            source_a = touch(Path(tmp) / "dataset" / "a.wav")
            source_b = touch(Path(tmp) / "dataset" / "b.wav")
            gt_wav = touch(preprocess_dir / "0_gt_wavs" / "0_0.wav")
            wav16k = touch(preprocess_dir / "1_16k_wavs" / "0_0.wav")
            touch(preprocess_dir / "3_feature256" / "0_0.npy")
            touch(preprocess_dir / "2a_f0" / "0_0.wav.npy")
            touch(preprocess_dir / "2b-f0nsf" / "0_0.wav.npy")
            touch(preprocess_dir / "0_gt_wavs" / "1_0.wav")
            touch(preprocess_dir / "1_16k_wavs" / "1_0.wav")
            touch(preprocess_dir / "3_feature256" / "1_0.npy")
            touch(preprocess_dir / "2a_f0" / "1_0.wav.npy")
            touch(preprocess_dir / "2b-f0nsf" / "1_0.wav.npy")

            write_preprocess_manifest(
                [
                    DatasetItem(source_a.resolve(), speaker_id=0, index=0),
                    DatasetItem(source_b.resolve(), speaker_id=0, index=1),
                ],
                preprocess_dir,
            )

            filelist_path, row_count, skipped = generate_filelist(
                project_for(preprocess_dir, if_f0=1)
            )
            self.assertEqual((row_count, skipped), (2, 0))
            self.assertEqual(len(filelist_path.read_text(encoding="utf-8").splitlines()[0].split("|")), 5)
            self.assertTrue((preprocess_dir / "train_filelist.txt").is_file())
            self.assertTrue((preprocess_dir / "val_filelist.txt").is_file())

            filelist_path, row_count, skipped = generate_filelist(
                project_for(preprocess_dir, if_f0=0)
            )
            self.assertEqual((row_count, skipped), (2, 0))
            self.assertEqual(len(filelist_path.read_text(encoding="utf-8").splitlines()[0].split("|")), 3)
            self.assertTrue(gt_wav.is_file())
            self.assertTrue(wav16k.is_file())

    def test_filelist_skips_missing_outputs_and_fails_when_all_invalid(self):
        with make_temp_dir() as tmp:
            preprocess_dir = Path(tmp) / "preprocess"
            source_a = touch(Path(tmp) / "dataset" / "a.wav")
            source_b = touch(Path(tmp) / "dataset" / "b.wav")
            source_c = touch(Path(tmp) / "dataset" / "c.wav")
            touch(preprocess_dir / "0_gt_wavs" / "0_0.wav")
            touch(preprocess_dir / "1_16k_wavs" / "0_0.wav")
            touch(preprocess_dir / "3_feature256" / "0_0.npy")
            touch(preprocess_dir / "0_gt_wavs" / "1_0.wav")
            touch(preprocess_dir / "1_16k_wavs" / "1_0.wav")
            touch(preprocess_dir / "3_feature256" / "1_0.npy")
            touch(preprocess_dir / "0_gt_wavs" / "2_0.wav")
            touch(preprocess_dir / "1_16k_wavs" / "2_0.wav")

            write_preprocess_manifest(
                [
                    DatasetItem(source_a.resolve(), speaker_id=0, index=0),
                    DatasetItem(source_b.resolve(), speaker_id=0, index=1),
                    DatasetItem(source_c.resolve(), speaker_id=0, index=2),
                ],
                preprocess_dir,
            )

            filelist_path, row_count, skipped = generate_filelist(
                project_for(preprocess_dir, if_f0=0)
            )
            self.assertEqual((row_count, skipped), (2, 1))
            self.assertIn("1_0.npy", filelist_path.read_text(encoding="utf-8"))

            invalid_preprocess_dir = Path(tmp) / "invalid_preprocess"
            invalid_source = touch(Path(tmp) / "dataset" / "invalid.wav")
            touch(invalid_preprocess_dir / "0_gt_wavs" / "0_0.wav")
            touch(invalid_preprocess_dir / "1_16k_wavs" / "0_0.wav")
            write_preprocess_manifest(
                [DatasetItem(invalid_source.resolve(), speaker_id=0, index=0)],
                invalid_preprocess_dir,
            )
            with self.assertRaisesRegex(RuntimeError, "No valid preprocess samples"):
                generate_filelist(project_for(invalid_preprocess_dir, if_f0=0))

    def test_filelist_can_scan_legacy_single_speaker_outputs_without_manifest(self):
        with make_temp_dir() as tmp:
            preprocess_dir = Path(tmp) / "preprocess"
            touch(preprocess_dir / "0_gt_wavs" / "sample0.wav")
            touch(preprocess_dir / "1_16k_wavs" / "sample0.wav")
            touch(preprocess_dir / "3_feature256" / "sample0.npy")
            touch(preprocess_dir / "0_gt_wavs" / "sample1.wav")
            touch(preprocess_dir / "1_16k_wavs" / "sample1.wav")
            touch(preprocess_dir / "3_feature256" / "sample1.npy")

            filelist_path, row_count, skipped = generate_filelist(
                project_for(preprocess_dir, if_f0=0)
            )

            self.assertEqual((row_count, skipped), (2, 0))
            self.assertEqual(len(filelist_path.read_text(encoding="utf-8").splitlines()[0].split("|")), 3)
            self.assertTrue((preprocess_dir / "train_filelist.txt").is_file())
            self.assertTrue((preprocess_dir / "val_filelist.txt").is_file())


if __name__ == "__main__":
    unittest.main()
