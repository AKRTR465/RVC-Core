import argparse
import os
from pathlib import Path

from scipy import signal

import numpy as np
from scipy.io import wavfile

from configs.project_config import load_project_config, parse_hparams_overrides
from src.preprocess.common import log_message, run_worker_shards

LOG_HANDLE = None


def init_log(preprocess_dir):
    global LOG_HANDLE
    os.makedirs(preprocess_dir, exist_ok=True)
    log_path = os.path.join(preprocess_dir, "preprocess.log")
    LOG_HANDLE = open(log_path, "a+", encoding="utf-8")


def println(message, log_path=None):
    log_message(log_path, message, handle=LOG_HANDLE)


class AudioPreprocessor:
    def __init__(self, sr, preprocess_dir, noparallel=False):
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.max = 0.9
        self.alpha = 0.75
        self.noparallel = noparallel
        self.gt_wavs_dir = os.path.join(preprocess_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(preprocess_dir, "1_16k_wavs")
        self.log_path = os.path.join(preprocess_dir, "preprocess.log")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        if tmp_audio.size == 0:
            println(f"{idx0}-{idx1}-empty-skip", self.log_path)
            return False
        tmp_max = np.abs(tmp_audio).max()
        if not np.isfinite(tmp_max) or tmp_max <= 1e-7:
            println(f"{idx0}-{idx1}-{tmp_max}-silent-skip", self.log_path)
            return False
        if tmp_max > 2.5:
            println(f"{idx0}-{idx1}-{tmp_max}-filtered", self.log_path)
            return False
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{idx0}_{idx1}.wav"),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        import librosa

        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{idx0}_{idx1}.wav"),
            16000,
            tmp_audio.astype(np.float32),
        )
        return True

    def load_and_filter_audio(self, path):
        from src.utils.audio import load_audio

        audio = load_audio(path, self.sr)
        return signal.lfilter(self.bh, self.ah, audio)

    def write_audio(self, audio, label, idx0):
        if not self.norm_write(audio, idx0, 0):
            println(f"{label}\t-> no valid audio", self.log_path)
            return False
        println(f"{label}\t-> Success", self.log_path)
        return True

    def pipeline(self, path, idx0):
        try:
            audio = self.load_and_filter_audio(path)
            return self.write_audio(audio, path, idx0)
        except (OSError, RuntimeError, ValueError) as exc:
            println(
                f"{path}\t-> {type(exc).__name__}: {exc}",
                self.log_path,
            )
            return False

    def pipeline_mp(self, infos):
        failures = 0
        for path, idx0 in infos:
            if not self.pipeline(path, idx0):
                failures += 1
        if failures:
            raise RuntimeError(f"{failures} preprocessing item(s) failed")

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            if n_p < 1:
                raise ValueError("n_p must be >= 1")
            infos = [
                (str(path), idx)
                for idx, path in enumerate(
                    sorted(path for path in Path(inp_root).iterdir() if path.is_file())
                )
            ]
            run_worker_shards(
                infos,
                n_p,
                self.pipeline_mp,
                lambda shard: (shard,),
                error_label="preprocess worker",
                parallel=not self.noparallel,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            println(f"Fail. {type(exc).__name__}: {exc}", self.log_path)
            raise


def preprocess_trainset(inp_root, sr, n_p, preprocess_dir, noparallel):
    global LOG_HANDLE
    init_log(preprocess_dir)
    try:
        pp = AudioPreprocessor(sr, preprocess_dir, noparallel=noparallel)
        println("start preprocess")
        pp.pipeline_mp_inp_dir(inp_root, n_p)
        println("end preprocess")
    finally:
        if LOG_HANDLE is not None:
            LOG_HANDLE.close()
            LOG_HANDLE = None


PreProcess = AudioPreprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("-i", "--inp_root", type=str, default="")
    parser.add_argument("-o", "--preprocess_dir", type=str, default="")
    parser.add_argument("-sr", "--sample-rate", "--sample_rate", type=int, default=None)
    parser.add_argument("-n", "--n_p", type=int, default=None)
    parser.add_argument("--noparallel", action="store_true")
    args = parser.parse_args()

    if args.config:
        if any(
            [
                args.inp_root,
                args.preprocess_dir,
                args.sample_rate is not None,
                args.n_p is not None,
                args.noparallel,
            ]
        ):
            parser.error("config mode only accepts --config, --hparams, and --reset")
        project = load_project_config(
            args.config,
            overrides=parse_hparams_overrides(args.hparams),
            reset=args.reset,
        )
        paths = project["paths"]
        runtime = project["runtime"]
        return (
            paths["dataset_dir"],
            int(project["data"]["sampling_rate"]),
            int(runtime["n_cpu"]),
            paths["preprocess_dir"],
            bool(project["preprocess"]["noparallel"]),
        )

    if any(
        [
            args.inp_root,
            args.preprocess_dir,
            args.sample_rate is not None,
            args.n_p is not None,
            args.noparallel,
        ]
    ):
        if args.inp_root == "" or args.preprocess_dir == "" or args.sample_rate is None:
            parser.error("manual mode requires --inp_root, --preprocess_dir, and --sample-rate")
        if args.n_p is None:
            parser.error("manual mode requires --n_p")
        if args.n_p < 1:
            parser.error("--n_p must be >= 1")
        return (
            args.inp_root,
            int(args.sample_rate),
            int(args.n_p),
            args.preprocess_dir,
            bool(args.noparallel),
        )

    parser.error(
        "provide --config or manual options --inp_root --preprocess_dir --sample-rate --n_p"
    )


if __name__ == "__main__":
    preprocess_trainset(*parse_args())

