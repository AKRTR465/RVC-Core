from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.rvc_profiles import FeatureProfile


GT_WAV_DIR_NAME = "0_gt_wavs"
WAV16K_DIR_NAME = "1_16k_wavs"
F0_DIR_NAME = "2a_f0"
F0NSF_DIR_NAME = "2b-f0nsf"
MANIFEST_NAME = "preprocess_manifest.jsonl"
FULL_FILELIST_NAME = "filelist.txt"
TRAIN_FILELIST_NAME = "train_filelist.txt"
VAL_FILELIST_NAME = "val_filelist.txt"
PREPROCESS_LOG_NAME = "preprocess.log"
FEATURE_LOG_NAME = "extract_f0_feature.log"


@dataclass(frozen=True)
class PreprocessLayout:
    root: Path
    feature_profile: FeatureProfile

    @property
    def gt_wavs_dir(self) -> Path:
        return self.root / GT_WAV_DIR_NAME

    @property
    def wav16k_dir(self) -> Path:
        return self.root / WAV16K_DIR_NAME

    @property
    def f0_dir(self) -> Path:
        return self.root / F0_DIR_NAME

    @property
    def f0nsf_dir(self) -> Path:
        return self.root / F0NSF_DIR_NAME

    @property
    def feature_dir(self) -> Path:
        return self.root / self.feature_profile.feature_dir_name

    @property
    def preprocess_log_path(self) -> Path:
        return self.root / PREPROCESS_LOG_NAME

    @property
    def feature_log_path(self) -> Path:
        return self.root / FEATURE_LOG_NAME

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def filelist_path(self) -> Path:
        return self.root / FULL_FILELIST_NAME

    @property
    def train_filelist_path(self) -> Path:
        return self.root / TRAIN_FILELIST_NAME

    @property
    def val_filelist_path(self) -> Path:
        return self.root / VAL_FILELIST_NAME

    def ensure_stage_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.gt_wavs_dir.mkdir(parents=True, exist_ok=True)
        self.wav16k_dir.mkdir(parents=True, exist_ok=True)
        self.f0_dir.mkdir(parents=True, exist_ok=True)
        self.f0nsf_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
