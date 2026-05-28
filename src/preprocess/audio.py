from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from src.preprocess.common import log_message
from src.preprocess.layout import GT_WAV_DIR_NAME, PREPROCESS_LOG_NAME, WAV16K_DIR_NAME


class AudioPreprocessor:
    def __init__(self, sr: int, preprocess_dir: str | Path):
        self.sr = int(sr)
        self.preprocess_dir = Path(preprocess_dir)
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.max = 0.9
        self.alpha = 0.75
        self.gt_wavs_dir = self.preprocess_dir / GT_WAV_DIR_NAME
        self.wavs16k_dir = self.preprocess_dir / WAV16K_DIR_NAME
        self.log_path = self.preprocess_dir / PREPROCESS_LOG_NAME
        self.gt_wavs_dir.mkdir(parents=True, exist_ok=True)
        self.wavs16k_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, message: str) -> None:
        log_message(self.log_path, message)

    def norm_write(self, tmp_audio: np.ndarray, idx0: int, idx1: int) -> bool:
        if tmp_audio.size == 0:
            self._log(f"{idx0}-{idx1}-empty-skip")
            return False
        tmp_max = np.abs(tmp_audio).max()
        if not np.isfinite(tmp_max) or tmp_max <= 1e-7:
            self._log(f"{idx0}-{idx1}-{tmp_max}-silent-skip")
            return False
        if tmp_max > 2.5:
            self._log(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return False
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            self.gt_wavs_dir / f"{idx0}_{idx1}.wav",
            self.sr,
            tmp_audio.astype(np.float32),
        )
        import librosa

        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write(
            self.wavs16k_dir / f"{idx0}_{idx1}.wav",
            16000,
            tmp_audio.astype(np.float32),
        )
        return True

    def load_and_filter_audio(self, path: str | Path) -> np.ndarray:
        from src.utils.audio import load_audio

        audio = load_audio(path, self.sr)
        return signal.lfilter(self.bh, self.ah, audio)

    def write_audio(self, audio: np.ndarray, label: str, idx0: int) -> bool:
        if not self.norm_write(audio, idx0, 0):
            self._log(f"{label}\t-> no valid audio")
            return False
        self._log(f"{label}\t-> Success")
        return True

    def process_item(self, path: str | Path, idx0: int) -> bool:
        try:
            audio = self.load_and_filter_audio(path)
            return self.write_audio(audio, str(path), idx0)
        except (OSError, RuntimeError, ValueError) as exc:
            self._log(f"{path}\t-> {type(exc).__name__}: {exc}")
            return False


def run_audio_items(
    item_payload: list[tuple[str, int]],
    sampling_rate: int,
    preprocess_dir: str | Path,
) -> None:
    worker = AudioPreprocessor(sampling_rate, preprocess_dir)
    failures = 0
    for source_path, item_index in item_payload:
        if not worker.process_item(source_path, item_index):
            failures += 1
    if failures:
        raise RuntimeError(f"{failures} audio preprocessing item(s) failed")
