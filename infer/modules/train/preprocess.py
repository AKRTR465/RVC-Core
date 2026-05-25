import argparse
import multiprocessing
import os
import sys
import traceback

from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)

import librosa
import numpy as np
from scipy.io import wavfile

from configs.project_config import load_project_config, parse_hparams_overrides
from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

LOG_HANDLE = None


def init_log(preprocess_dir):
    global LOG_HANDLE
    os.makedirs(preprocess_dir, exist_ok=True)
    LOG_HANDLE = open(
        os.path.join(preprocess_dir, "preprocess.log"), "a+", encoding="utf-8"
    )


def println(message):
    print(message)
    if LOG_HANDLE is not None:
        LOG_HANDLE.write(f"{message}\n")
        LOG_HANDLE.flush()


class PreProcess:
    def __init__(self, sr, preprocess_dir, noparallel=False, per=3.7):
        self.slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sr)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.preprocess_dir = preprocess_dir
        self.noparallel = noparallel
        self.gt_wavs_dir = os.path.join(preprocess_dir, "0_gt_wavs")
        self.wavs16k_dir = os.path.join(preprocess_dir, "1_16k_wavs")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            println(f"{idx0}-{idx1}-{tmp_max}-filtered")
            return
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
            1 - self.alpha
        ) * tmp_audio
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{idx0}_{idx1}.wav"),
            self.sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(tmp_audio, orig_sr=self.sr, target_sr=16000)
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{idx0}_{idx1}.wav"),
            16000,
            tmp_audio.astype(np.float32),
        )

    def pipeline(self, path, idx0):
        try:
            audio = load_audio(path, self.sr)
            audio = signal.lfilter(self.bh, self.ah, audio)

            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while True:
                    start = int(self.sr * (self.per - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sr:
                        tmp_audio = audio[start : start + int(self.per * self.sr)]
                        self.norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        idx1 += 1
                        break
                self.norm_write(tmp_audio, idx0, idx1)
            println(f"{path}\t-> Success")
        except Exception:
            println(f"{path}\t-> {traceback.format_exc()}")

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            infos = [
                (os.path.join(inp_root, name), idx)
                for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            ]
            if self.noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    process = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(process)
                    process.start()
                for process in ps:
                    process.join()
        except Exception:
            println(f"Fail. {traceback.format_exc()}")


def preprocess_trainset(inp_root, sr, n_p, preprocess_dir, noparallel, per):
    init_log(preprocess_dir)
    pp = PreProcess(sr, preprocess_dir, noparallel=noparallel, per=per)
    println("start preprocess")
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("end preprocess")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_inp_root", nargs="?")
    parser.add_argument("legacy_sr", nargs="?")
    parser.add_argument("legacy_n_p", nargs="?")
    parser.add_argument("legacy_preprocess_dir", nargs="?")
    parser.add_argument("legacy_noparallel", nargs="?")
    parser.add_argument("legacy_per", nargs="?")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("-i", "--inp_root", type=str, default="")
    parser.add_argument("-o", "--preprocess_dir", type=str, default="")
    parser.add_argument("-sr", "--sample-rate", "--sample_rate", type=int, default=None)
    parser.add_argument("-n", "--n_p", type=int, default=None)
    parser.add_argument("--noparallel", action="store_true")
    parser.add_argument("--per", type=float, default=None)
    args = parser.parse_args()

    if args.config:
        if any(
            [
                args.inp_root,
                args.preprocess_dir,
                args.sample_rate is not None,
                args.n_p is not None,
                args.noparallel,
                args.per is not None,
            ]
        ):
            parser.error("config mode only accepts --config, --hparams, and --reset")
        project = load_project_config(
            args.config,
            overrides=parse_hparams_overrides(args.hparams),
            reset=args.reset,
        )
        return (
            project["dataset_dir"],
            int(project["data"]["sampling_rate"]),
            int(project["n_cpu"]),
            project["preprocess_dir"],
            bool(project["preprocess"]["noparallel"]),
            float(project["preprocess"]["per"]),
        )

    if any(
        [
            args.inp_root,
            args.preprocess_dir,
            args.sample_rate is not None,
            args.n_p is not None,
            args.per is not None,
        ]
    ):
        if args.inp_root == "" or args.preprocess_dir == "" or args.sample_rate is None:
            parser.error("manual mode requires --inp_root, --preprocess_dir, and --sample-rate")
        if args.n_p is None or args.per is None:
            parser.error("manual mode requires --n_p and --per")
        return (
            args.inp_root,
            int(args.sample_rate),
            int(args.n_p),
            args.preprocess_dir,
            bool(args.noparallel),
            float(args.per),
        )

    if None in {
        args.legacy_inp_root,
        args.legacy_sr,
        args.legacy_n_p,
        args.legacy_preprocess_dir,
        args.legacy_noparallel,
        args.legacy_per,
    }:
        parser.error(
            "legacy mode requires: inp_root sr n_p preprocess_dir noparallel per"
        )

    return (
        args.legacy_inp_root,
        int(args.legacy_sr),
        int(args.legacy_n_p),
        args.legacy_preprocess_dir,
        args.legacy_noparallel == "True",
        float(args.legacy_per),
    )


if __name__ == "__main__":
    preprocess_trainset(*parse_args())
