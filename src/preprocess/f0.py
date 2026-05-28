from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from src.features.f0 import (
    F0_BIN,
    F0_MAX,
    F0_METHODS,
    F0_MIN,
    compute_f0_by_method,
    f0_to_coarse,
)
from src.preprocess.common import log_message
from src.preprocess.layout import PreprocessLayout

logging.getLogger("numba").setLevel(logging.WARNING)


def detect_cuda_profile():
    try:
        import torch
    except ImportError:
        return {
            "device": "cpu",
            "gpu_name": None,
            "supports_half": False,
        }

    if not torch.cuda.is_available():
        return {
            "device": "cpu",
            "gpu_name": None,
            "supports_half": False,
        }

    device = "cuda:0"
    gpu_name = torch.cuda.get_device_name(0)
    upper_name = gpu_name.upper()
    supports_half = not (
        ("16" in gpu_name and "V100" not in upper_name)
        or "P40" in upper_name
        or "P10" in upper_name
        or "1060" in gpu_name
        or "1070" in gpu_name
        or "1080" in gpu_name
    )
    return {
        "device": device,
        "gpu_name": gpu_name,
        "supports_half": supports_half,
    }


class FeatureInput:
    def __init__(
        self,
        samplerate: int = 16000,
        hop_size: int = 160,
        device: str = "cpu",
        is_half: bool = False,
        pretrain_root: str = "pretrain",
    ):
        self.fs = samplerate
        self.hop = hop_size
        self.device = device
        self.is_half = is_half
        self.pretrain_root = pretrain_root
        self.f0_bin = F0_BIN
        self.f0_max = F0_MAX
        self.f0_min = F0_MIN

    def compute_f0(self, path: str | Path, f0_method: str):
        from src.utils.audio import load_audio

        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop
        f0, self.model_rmvpe = compute_f0_by_method(
            x,
            self.fs,
            p_len,
            self.hop,
            f0_method,
            device=self.device,
            is_half=self.is_half,
            pretrain_root=self.pretrain_root,
            rmvpe_model=getattr(self, "model_rmvpe", None),
            log_fn=print,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
        )
        return f0

    def coarse_f0(self, f0: np.ndarray):
        return f0_to_coarse(f0, self.f0_min, self.f0_max, self.f0_bin)

    def go(self, paths, f0_method, log_path):
        if len(paths) == 0:
            log_message(log_path, "no-f0-todo")
            return

        log_message(log_path, f"todo-f0-{len(paths)}")
        every = max(len(paths) // 5, 1)
        failures = 0
        for idx, (inp_path, opt_path1, opt_path2) in enumerate(paths):
            try:
                if idx % every == 0:
                    log_message(log_path, f"f0ing,now-{idx},all-{len(paths)},-{inp_path}")
                if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"):
                    continue
                featur_pit = self.compute_f0(inp_path, f0_method)
                np.save(opt_path2, featur_pit, allow_pickle=False)
                np.save(opt_path1, self.coarse_f0(featur_pit), allow_pickle=False)
            except (OSError, RuntimeError, ValueError, ImportError) as exc:
                failures += 1
                log_message(
                    log_path,
                    f"f0fail-{idx}-{inp_path}-{type(exc).__name__}: {exc}",
                )
        if failures:
            raise RuntimeError(f"{failures} f0 item(s) failed")


def build_paths(layout: PreprocessLayout):
    layout.f0_dir.mkdir(parents=True, exist_ok=True)
    layout.f0nsf_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for inp_path in sorted(layout.wav16k_dir.iterdir()):
        if not inp_path.is_file() or inp_path.suffix.lower() != ".wav":
            continue
        if "spec" in inp_path.stem:
            continue
        opt_path1 = layout.f0_dir / inp_path.name
        opt_path2 = layout.f0nsf_dir / inp_path.name
        paths.append([str(inp_path), str(opt_path1), str(opt_path2)])
    return paths


def resolve_runtime(
    f0_method: str,
    device_request: str,
    is_half_request: bool,
    log_path,
):
    if f0_method not in F0_METHODS:
        raise ValueError(f"Unsupported f0 method: {f0_method}")

    requested = str(device_request).strip().lower()
    if requested in {"", "auto"}:
        requested = "cuda" if f0_method in {"rmvpe", "crepe"} else "cpu"

    if f0_method in {"rmvpe", "crepe"}:
        try:
            import torch
        except ImportError:
            torch = None

        if requested.startswith("cuda") or requested == "cuda":
            if torch is None or not torch.cuda.is_available():
                log_message(log_path, f"{f0_method}-gpu-request-fell-back-to-cpu")
                return "cpu", False
            if f0_method == "crepe":
                if is_half_request:
                    log_message(log_path, "crepe-ignoring-is-half")
                return requested if requested != "cuda" else "cuda", False

            profile = detect_cuda_profile()
            resolved_device = requested if requested != "cuda" else profile["device"]
            if is_half_request and not profile["supports_half"]:
                log_message(
                    log_path,
                    f"rmvpe-half-request-ignored-unsupported-gpu={profile['gpu_name']}",
                )
            return resolved_device, bool(is_half_request and profile["supports_half"])

        if is_half_request:
            log_message(log_path, f"{f0_method}-cpu-mode-forces-fp32")
        return "cpu", False

    if requested.startswith("cuda"):
        log_message(log_path, f"ignoring GPU runtime for f0method={f0_method}")
    if is_half_request:
        log_message(log_path, f"ignoring --is-half for f0method={f0_method}")
    return "cpu", False


def run_worker(paths, f0_method, device, is_half, pretrain_root, log_path):
    worker = FeatureInput(
        device=device,
        is_half=is_half,
        pretrain_root=pretrain_root,
    )
    worker.go(paths, f0_method, log_path)
