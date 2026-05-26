import unittest
from pathlib import Path

from src.preprocess.pipeline import (
    DatasetItem,
    discover_dataset_items,
    generate_filelist,
    write_preprocess_manifest,
)
from tests.equivalence_helpers import make_temp_dir


def touch(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def project_for(preprocess_dir, if_f0=1, version="v1"):
    return {
        "preprocess_dir": str(preprocess_dir),
        "version": version,
        "if_f0": if_f0,
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

    def test_filelist_columns_follow_if_f0(self):
        with make_temp_dir() as tmp:
            preprocess_dir = Path(tmp) / "preprocess"
            source = touch(Path(tmp) / "dataset" / "a.wav")
            gt_wav = touch(preprocess_dir / "0_gt_wavs" / "0_0.wav")
            wav16k = touch(preprocess_dir / "1_16k_wavs" / "0_0.wav")
            touch(preprocess_dir / "3_feature256" / "0_0.npy")
            touch(preprocess_dir / "2a_f0" / "0_0.wav.npy")
            touch(preprocess_dir / "2b-f0nsf" / "0_0.wav.npy")

            write_preprocess_manifest(
                [DatasetItem(source.resolve(), speaker_id=0, index=0)],
                preprocess_dir,
            )

            filelist_path, row_count, skipped = generate_filelist(
                project_for(preprocess_dir, if_f0=1)
            )
            self.assertEqual((row_count, skipped), (1, 0))
            self.assertEqual(len(filelist_path.read_text(encoding="utf-8").split("|")), 5)

            filelist_path, row_count, skipped = generate_filelist(
                project_for(preprocess_dir, if_f0=0)
            )
            self.assertEqual((row_count, skipped), (1, 0))
            self.assertEqual(len(filelist_path.read_text(encoding="utf-8").split("|")), 3)
            self.assertTrue(gt_wav.is_file())
            self.assertTrue(wav16k.is_file())

    def test_filelist_skips_missing_outputs_and_fails_when_all_invalid(self):
        with make_temp_dir() as tmp:
            preprocess_dir = Path(tmp) / "preprocess"
            source = touch(Path(tmp) / "dataset" / "a.wav")
            touch(preprocess_dir / "0_gt_wavs" / "0_0.wav")
            touch(preprocess_dir / "0_gt_wavs" / "1_0.wav")
            touch(preprocess_dir / "1_16k_wavs" / "0_0.wav")
            touch(preprocess_dir / "1_16k_wavs" / "1_0.wav")
            touch(preprocess_dir / "3_feature256" / "1_0.npy")

            write_preprocess_manifest(
                [
                    DatasetItem(source.resolve(), speaker_id=0, index=0),
                    DatasetItem(source.resolve(), speaker_id=0, index=1),
                ],
                preprocess_dir,
            )

            filelist_path, row_count, skipped = generate_filelist(
                project_for(preprocess_dir, if_f0=0)
            )
            self.assertEqual((row_count, skipped), (1, 1))
            self.assertIn("1_0.npy", filelist_path.read_text(encoding="utf-8"))

            invalid_preprocess_dir = Path(tmp) / "invalid_preprocess"
            touch(invalid_preprocess_dir / "0_gt_wavs" / "0_0.wav")
            touch(invalid_preprocess_dir / "1_16k_wavs" / "0_0.wav")
            write_preprocess_manifest(
                [DatasetItem(source.resolve(), speaker_id=0, index=0)],
                invalid_preprocess_dir,
            )
            with self.assertRaisesRegex(RuntimeError, "No valid preprocess samples"):
                generate_filelist(project_for(invalid_preprocess_dir, if_f0=0))

    def test_filelist_can_scan_legacy_single_speaker_outputs_without_manifest(self):
        with make_temp_dir() as tmp:
            preprocess_dir = Path(tmp) / "preprocess"
            touch(preprocess_dir / "0_gt_wavs" / "sample.wav")
            touch(preprocess_dir / "1_16k_wavs" / "sample.wav")
            touch(preprocess_dir / "3_feature256" / "sample.npy")

            filelist_path, row_count, skipped = generate_filelist(
                project_for(preprocess_dir, if_f0=0)
            )

            self.assertEqual((row_count, skipped), (1, 0))
            self.assertEqual(len(filelist_path.read_text(encoding="utf-8").split("|")), 3)


if __name__ == "__main__":
    unittest.main()
