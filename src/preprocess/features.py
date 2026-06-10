from __future__ import annotations

import os

import numpy as np
import torch

from src.features.hubert import extract_hubert_features
from src.features.hubert import load_hubert_model as load_shared_hubert_model
from src.features.hubert import read_wave_16k
from src.preprocess.common import log_message
from src.preprocess.layout import PreprocessLayout


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


def extract_features(
    layout: PreprocessLayout,
    n_part: int,
    i_part: int,
    device: str,
    is_half: bool,
    model_path: str,
):
    log_message(layout.feature_log_path, f"exp_dir: {layout.root}")
    layout.feature_dir.mkdir(parents=True, exist_ok=True)

    todo = [
        path
        for path in sorted(layout.wav16k_dir.iterdir())
        if path.is_file() and path.suffix.lower() == ".wav"
    ][i_part::n_part]
    every = max(1, len(todo) // 10)
    if not todo:
        log_message(layout.feature_log_path, "no-feature-todo")
        return

    model, saved_cfg = load_shared_hubert_model(
        model_path,
        device,
        is_half,
        log_fn=lambda message: log_message(layout.feature_log_path, message),
    )
    if model is None:
        return

    log_message(layout.feature_log_path, f"all-feature-{len(todo)}")
    failures = 0
    for idx, wav_path in enumerate(todo):
        file = wav_path.name
        try:
            out_path = layout.feature_dir / wav_path.with_suffix(".npy").name
            if out_path.exists():
                continue

            waveform = read_wave_16k(wav_path)
            feats = extract_hubert_features(
                model,
                waveform,
                layout.feature_profile.version,
                device,
                is_half,
                normalize=saved_cfg.task.normalize,
                source_label=str(wav_path),
            )

            feature_npy = feats.squeeze(0).float().cpu().numpy()
            if np.isfinite(feature_npy).all():
                np.save(out_path, feature_npy, allow_pickle=False)
            else:
                log_message(layout.feature_log_path, f"{file}-contains non-finite values")
            if idx % every == 0:
                log_message(
                    layout.feature_log_path,
                    f"now-{idx},all-{len(todo)},{file},{feature_npy.shape}",
                )
        except (OSError, RuntimeError, ValueError) as exc:
            failures += 1
            log_message(
                layout.feature_log_path,
                f"{file}-feature-fail-{type(exc).__name__}: {exc}",
            )
    if failures:
        raise RuntimeError(f"{failures} feature extraction item(s) failed")
    log_message(layout.feature_log_path, "all-feature-done")
