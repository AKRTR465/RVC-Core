import argparse
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from configs.project_config import load_project_config, parse_hparams_overrides
from src.features.hubert import extract_hubert_features
from src.features.hubert import load_hubert_model as load_shared_hubert_model
from src.features.hubert import read_wave_16k
from src.preprocess.common import log_message


def resolve_device(device_request, i_gpu=""):
    if i_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    if device_request not in {"auto", "cpu", "cuda"} and not str(device_request).startswith("cuda:"):
        raise ValueError("--device must be one of: auto, cpu, cuda, cuda:<index>")
    if device_request == "cpu":
        return "cpu"
    if device_request == "cuda" or str(device_request).startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA for feature extraction, but CUDA is not available")
        return device_request
    return "cuda" if torch.cuda.is_available() else "cpu"


def output_dir_for_version(exp_dir, version):
    name = "3_feature256" if version == "v1" else "3_feature768"
    return Path(exp_dir) / name


def extract_features(
    exp_dir,
    version,
    n_part,
    i_part,
    device,
    is_half,
    model_path,
    log_path=None,
):
    exp_path = Path(exp_dir)
    wav_dir = exp_path / "1_16k_wavs"
    out_dir = output_dir_for_version(exp_path, version)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or exp_path / "extract_f0_feature.log"

    log_message(log_path, f"exp_dir: {exp_path}")

    todo = [
        path
        for path in sorted(wav_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".wav"
    ][i_part::n_part]
    every = max(1, len(todo) // 10)
    if not todo:
        log_message(log_path, "no-feature-todo")
        return

    model, saved_cfg = load_shared_hubert_model(
        model_path,
        device,
        is_half,
        log_fn=lambda message: log_message(log_path, message),
    )
    if model is None:
        return

    log_message(log_path, f"all-feature-{len(todo)}")
    failures = 0
    for idx, wav_path in enumerate(todo):
        file = wav_path.name
        try:
            out_path = out_dir / wav_path.with_suffix(".npy").name
            if out_path.exists():
                continue

            waveform = read_wave_16k(wav_path, sf, normalize=saved_cfg.task.normalize)
            feats = extract_hubert_features(
                model,
                waveform,
                version,
                device,
                is_half,
                normalize=saved_cfg.task.normalize,
                source_label=str(wav_path),
            )

            feature_npy = feats.squeeze(0).float().cpu().numpy()
            if np.isfinite(feature_npy).all():
                np.save(out_path, feature_npy, allow_pickle=False)
            else:
                log_message(log_path, f"{file}-contains non-finite values")
            if idx % every == 0:
                log_message(
                    log_path,
                    f"now-{idx},all-{len(todo)},{file},{feature_npy.shape}",
                )
        except (OSError, RuntimeError, ValueError, sf.SoundFileError) as exc:
            failures += 1
            log_message(log_path, f"{file}-feature-fail-{type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError(f"{failures} feature extraction item(s) failed")
    log_message(log_path, "all-feature-done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--exp-dir", type=str, default="")
    parser.add_argument("--version", choices=["v1", "v2"], default="")
    parser.add_argument("--n-part", type=int, default=1)
    parser.add_argument("--i-part", type=int, default=0)
    parser.add_argument("--i-gpu", type=str, default="")
    parser.add_argument("--is-half", action="store_true", default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pretrain-root", type=str, default=os.getenv("pretrain_root", "pretrain"))
    parser.add_argument("--model-path", type=str, default="")
    args = parser.parse_args()

    if args.config:
        if args.exp_dir or args.version or args.pretrain_root != os.getenv("pretrain_root", "pretrain"):
            parser.error(
                "config mode only accepts --config, --hparams, --reset, "
                "partition, device, and model options"
            )
        project = load_project_config(
            args.config,
            overrides=parse_hparams_overrides(args.hparams),
            reset=args.reset,
        )
        paths = project["paths"]
        runtime = project["runtime"]
        args.exp_dir = paths["preprocess_dir"]
        args.version = project["selectors"]["version"]
        args.pretrain_root = paths["pretrain_root"]
        if args.device == "auto":
            device = str(runtime["device"])
            args.device = device if device.startswith("cuda") else "cpu"
        if args.is_half is None:
            args.is_half = bool(runtime["is_half"])

    if args.exp_dir == "" or args.version == "":
        parser.error("provide --config or both --exp-dir and --version")
    if args.n_part < 1:
        parser.error("--n-part must be >= 1")
    if args.i_part < 0 or args.i_part >= args.n_part:
        parser.error("--i-part must satisfy 0 <= i_part < n_part")
    if args.model_path == "":
        args.model_path = str(Path(args.pretrain_root) / "hubert" / "hubert_base.pt")
    if args.is_half is None:
        args.is_half = False
    return args


def main():
    args = parse_args()
    device = resolve_device(args.device, args.i_gpu)
    log_path = Path(args.exp_dir) / "extract_f0_feature.log"
    log_message(log_path, " ".join(sys.argv))
    extract_features(
        exp_dir=args.exp_dir,
        version=args.version,
        n_part=args.n_part,
        i_part=args.i_part,
        device=device,
        is_half=bool(args.is_half),
        model_path=args.model_path,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
