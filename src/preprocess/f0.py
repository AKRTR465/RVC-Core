import argparse
import logging
import os
import sys
from multiprocessing import Process, cpu_count
from pathlib import Path

import numpy as np

from configs.project_config import load_project_config, parse_hparams_overrides
from src.features.f0 import (
    F0_BIN,
    F0_MAX,
    F0_MIN,
    compute_pm_f0,
    compute_world_f0,
    f0_to_coarse,
    load_rmvpe_model,
)

logging.getLogger("numba").setLevel(logging.WARNING)


def log_message(log_path, message):
    print(message)
    with open(log_path, "a+", encoding="utf-8") as handle:
        handle.write(f"{message}\n")
        handle.flush()


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
        samplerate=16000,
        hop_size=160,
        device="cpu",
        is_half=False,
        pretrain_root="pretrain",
    ):
        self.fs = samplerate
        self.hop = hop_size
        self.device = device
        self.is_half = is_half
        self.pretrain_root = pretrain_root

        self.f0_bin = F0_BIN
        self.f0_max = F0_MAX
        self.f0_min = F0_MIN

    def compute_f0(self, path, f0_method):
        from src.utils.audio import load_audio

        x = load_audio(path, self.fs)
        p_len = x.shape[0] // self.hop
        if f0_method == "pm":
            f0 = compute_pm_f0(x, self.fs, p_len, self.f0_min, self.f0_max, self.hop)
        elif f0_method in {"harvest", "dio"}:
            f0 = compute_world_f0(x, self.fs, self.hop, f0_method, self.f0_min, self.f0_max)
        elif f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") is False:
                self.model_rmvpe = load_rmvpe_model(
                    os.path.join(self.pretrain_root, "rmvpe", "rmvpe.pt"),
                    self.device,
                    self.is_half,
                    print,
                )
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        else:
            raise ValueError(f"Unsupported f0 method: {f0_method}")
        return f0

    def coarse_f0(self, f0):
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
                    log_message(
                        log_path, f"f0ing,now-{idx},all-{len(paths)},-{inp_path}"
                    )
                if os.path.exists(opt_path1 + ".npy") and os.path.exists(
                    opt_path2 + ".npy"
                ):
                    continue
                featur_pit = self.compute_f0(inp_path, f0_method)
                np.save(opt_path2, featur_pit, allow_pickle=False)
                coarse_pit = self.coarse_f0(featur_pit)
                np.save(opt_path1, coarse_pit, allow_pickle=False)
            except (OSError, RuntimeError, ValueError, ImportError) as exc:
                failures += 1
                log_message(
                    log_path,
                    f"f0fail-{idx}-{inp_path}-{type(exc).__name__}: {exc}",
                )
        if failures:
            raise RuntimeError(f"{failures} f0 item(s) failed")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_exp_dir", nargs="?")
    parser.add_argument("legacy_workers", nargs="?")
    parser.add_argument("legacy_f0method", nargs="?")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--exp-dir", type=str, default="")
    parser.add_argument(
        "--f0method",
        type=str,
        choices=["pm", "harvest", "dio", "rmvpe"],
        default="",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--n-part", type=int, default=None)
    parser.add_argument("--i-part", type=int, default=None)
    parser.add_argument("--i-gpu", type=str, default="")
    parser.add_argument("--is-half", action="store_true", default=None)
    args = parser.parse_args()

    legacy_values = [
        args.legacy_exp_dir,
        args.legacy_workers,
        args.legacy_f0method,
    ]
    if any(value is not None for value in legacy_values):
        if any(
            [
                args.config,
                args.exp_dir,
                args.f0method,
                args.workers is not None,
                args.is_half is not None,
            ]
        ):
            parser.error("legacy mode cannot be mixed with --exp-dir/--f0method/--workers")
        if None in legacy_values:
            parser.error("legacy mode requires: exp_dir workers f0method")
        args.exp_dir = args.legacy_exp_dir
        args.workers = int(args.legacy_workers)
        args.f0method = args.legacy_f0method

    if args.config:
        if any(value is not None for value in legacy_values) or args.exp_dir:
            parser.error("config mode cannot be mixed with legacy args or --exp-dir")
        project = load_project_config(
            args.config,
            overrides=parse_hparams_overrides(args.hparams),
            reset=args.reset,
        )
        args.exp_dir = project["preprocess_dir"]
        args.f0method = (
            args.f0method
            or project.get("preprocess", {}).get("f0method")
            or "rmvpe"
        )
        device = str(project["device"])
        if device.startswith("cuda:") and args.i_gpu == "":
            args.i_gpu = device.split(":", 1)[1]
        if args.is_half is None:
            args.is_half = bool(project["is_half"])
        if args.workers is None:
            args.workers = 1 if args.f0method == "rmvpe" else int(project["n_cpu"])

    if args.exp_dir == "" or args.f0method == "":
        parser.error("provide legacy args or --exp-dir and --f0method")
    if args.f0method not in {"pm", "harvest", "dio", "rmvpe"}:
        parser.error("--f0method must be one of: pm, harvest, dio, rmvpe")

    if args.workers is None:
        args.workers = 1 if args.f0method == "rmvpe" else cpu_count()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.is_half is None:
        args.is_half = False

    if (args.n_part is None) != (args.i_part is None):
        parser.error("--n-part and --i-part must be provided together")
    if args.n_part is not None:
        if args.n_part < 1:
            parser.error("--n-part must be >= 1")
        if args.i_part < 0 or args.i_part >= args.n_part:
            parser.error("--i-part must satisfy 0 <= i_part < n_part")

    return args


def build_paths(exp_dir):
    inp_root = Path(exp_dir) / "1_16k_wavs"
    opt_root1 = Path(exp_dir) / "2a_f0"
    opt_root2 = Path(exp_dir) / "2b-f0nsf"

    os.makedirs(opt_root1, exist_ok=True)
    os.makedirs(opt_root2, exist_ok=True)

    paths = []
    for inp_path in sorted(inp_root.iterdir()):
        if not inp_path.is_file() or inp_path.suffix.lower() != ".wav":
            continue
        if "spec" in inp_path.stem:
            continue
        opt_path1 = opt_root1 / inp_path.name
        opt_path2 = opt_root2 / inp_path.name
        paths.append([str(inp_path), str(opt_path1), str(opt_path2)])
    return paths


def resolve_runtime(args, log_path):
    pretrain_root = os.getenv("pretrain_root", "pretrain")
    if args.f0method == "rmvpe":
        if args.i_gpu != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.i_gpu)

        profile = detect_cuda_profile()
        device = profile["device"]
        supports_half = profile["supports_half"]

        if device.startswith("cuda"):
            if args.is_half and not supports_half:
                log_message(
                    log_path,
                    f"rmvpe-half-request-ignored-unsupported-gpu={profile['gpu_name']}",
                )
            is_half = bool(args.is_half and supports_half)
            return device, is_half, pretrain_root

        if args.is_half:
            log_message(log_path, "rmvpe-cpu-mode-forces-fp32")
        if args.i_gpu != "":
            log_message(log_path, "rmvpe-gpu-request-fell-back-to-cpu")
        return "cpu", False, pretrain_root

    if args.i_gpu != "":
        log_message(log_path, f"ignoring --i-gpu for f0method={args.f0method}")
    if args.is_half:
        log_message(log_path, f"ignoring --is-half for f0method={args.f0method}")
    return "cpu", False, pretrain_root


def run_worker(paths, f0_method, device, is_half, pretrain_root, log_path):
    worker = FeatureInput(
        device=device,
        is_half=is_half,
        pretrain_root=pretrain_root,
    )
    worker.go(paths, f0_method, log_path)


def main():
    args = parse_args()
    os.makedirs(args.exp_dir, exist_ok=True)
    log_path = os.path.join(args.exp_dir, "extract_f0_feature.log")
    log_message(log_path, " ".join(sys.argv))

    device, is_half, pretrain_root = resolve_runtime(args, log_path)
    paths = build_paths(args.exp_dir)
    sharded_mode = args.n_part is not None and args.i_part is not None

    if sharded_mode:
        log_message(
            log_path,
            f"mode=sharded,part={args.i_part}/{args.n_part},device={device},is_half={is_half}",
        )
        run_worker(
            paths[args.i_part :: args.n_part],
            args.f0method,
            device,
            is_half,
            pretrain_root,
            log_path,
        )
        return

    log_message(
        log_path,
        f"mode=workers,workers={args.workers},device={device},is_half={is_half}",
    )
    if args.workers == 1:
        run_worker(paths, args.f0method, device, is_half, pretrain_root, log_path)
        return

    ps = []
    for i in range(args.workers):
        process = Process(
            target=run_worker,
            args=(
                paths[i:: args.workers],
                args.f0method,
                device,
                is_half,
                pretrain_root,
                log_path,
            ),
        )
        ps.append(process)
        process.start()
    for process in ps:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"f0 worker {process.pid} exited with code {process.exitcode}"
            )


if __name__ == "__main__":
    main()

