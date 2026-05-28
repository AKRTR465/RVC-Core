from __future__ import annotations

import copy
import multiprocessing
from typing import Any, Callable

try:
    import torch
except ImportError:  # pragma: no cover - torch is required at runtime
    torch = None


DEFAULT_RUNTIME = {
    "device": "auto",
    "is_half": "auto",
    "n_cpu": "auto",
    "deterministic_algorithms": "off",
    "disable_tf32": False,
    "cublas_workspace_config": None,
    "slice": {
        "x_pad": "auto",
        "x_query": "auto",
        "x_center": "auto",
        "x_max": "auto",
    },
}

DEFAULT_INFER = {
    "model_path": "auto",
    "index_path": "auto",
    "f0method": "rmvpe",
    "pitch": 12.0,
    "index_rate": 0.0,
    "block_time": 0.15,
    "crossfade_length": 0.08,
    "extra_time": 2.0,
    "rms_mix_rate": 0.5,
    "formant": 0.0,
    "use_pv": False,
}

DEFAULT_TRAIN = {
    "fp16_run": "auto",
    "use_tqdm": "auto",
    "save_every_epoch": 10,
    "save_every_weights": False,
    "if_latest": 0,
    "if_cache_data_in_gpu": 0,
    "num_workers": "auto",
    "persistent_workers": "auto",
    "prefetch_factor": "auto",
    "pretrainG": "",
    "pretrainD": "",
    "numeric_backend": "native",
    "grad_scaler_init_scale": 32.0,
}


def normalize_auto_bool(value: Any, field_name: str) -> str | bool:
    if value in (None, "", "auto"):
        return "auto"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{field_name} must be auto|true|false, got: {value!r}")


def normalize_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{field_name} must be true|false, got: {value!r}")


def normalize_auto_int(value: Any, field_name: str) -> str | int:
    if value in (None, "", "auto"):
        return "auto"
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be auto or integer, got: {value!r}") from exc
    if number < 1:
        raise ValueError(f"{field_name} must be >= 1, got: {number}")
    return number


def normalize_slice_value(value: Any, field_name: str) -> str | int:
    if value in (None, "", "auto"):
        return "auto"
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be auto or integer, got: {value!r}") from exc


def normalize_device_request(value: Any) -> str:
    if value in (None, "", "auto"):
        return "auto"
    device = str(value).strip().lower()
    if device == "cuda":
        return "cuda:0"
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda:"):
        return device
    raise ValueError(f"runtime.device must be auto|cpu|cuda[:index], got: {value!r}")


def normalize_deterministic_algorithms(value: Any) -> str:
    if isinstance(value, bool):
        return "error" if value else "off"
    text = str(value).strip().lower()
    if text in {"", "none"}:
        return "off"
    if text not in {"off", "warn_only", "error"}:
        raise ValueError(
            "runtime.deterministic_algorithms must be off|warn_only|error, "
            f"got: {value!r}"
        )
    return text


def normalize_numeric_backend(value: Any) -> str:
    text = str(value if value is not None else "native").strip().lower()
    if text in {"", "none"}:
        return "native"
    if text not in {"native", "deterministic_gpu"}:
        raise ValueError(
            "train.numeric_backend must be native|deterministic_gpu, "
            f"got: {value!r}"
        )
    return text


def normalize_positive_float(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive float, got: {value!r}") from exc
    if number <= 0.0:
        raise ValueError(f"{field_name} must be > 0, got: {number}")
    return number


def detect_runtime_environment(device_request: str) -> dict[str, Any]:
    if device_request == "auto":
        if torch is not None and torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    else:
        device = device_request

    profile = {
        "device": device,
        "device_request": device_request,
        "gpu_name": None,
        "gpu_mem_gb": None,
        "supports_half": False,
    }

    if device == "cpu":
        return profile

    if torch is None or not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {device}, but CUDA is not available")

    try:
        gpu_index = int(device.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid CUDA device: {device}") from exc

    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested CUDA device {device}, but only {torch.cuda.device_count()} CUDA device(s) are available"
        )

    gpu_name = torch.cuda.get_device_name(gpu_index)
    gpu_mem_gb = int(
        torch.cuda.get_device_properties(gpu_index).total_memory / 1024 / 1024 / 1024 + 0.4
    )
    upper_name = gpu_name.upper()
    supports_half = not (
        ("16" in gpu_name and "V100" not in upper_name)
        or "P40" in upper_name
        or "P10" in upper_name
        or "1060" in gpu_name
        or "1070" in gpu_name
        or "1080" in gpu_name
    )

    profile.update(
        {
            "gpu_name": gpu_name,
            "gpu_mem_gb": gpu_mem_gb,
            "supports_half": supports_half,
        }
    )
    return profile


def resolve_runtime_profile(
    config: dict[str, Any],
    *,
    detect_runtime_environment_fn: Callable[[str], dict[str, Any]] | None = None,
    cpu_count_fn: Callable[[], int] | None = None,
) -> dict[str, Any]:
    if detect_runtime_environment_fn is None:
        detect_runtime_environment_fn = detect_runtime_environment
    if cpu_count_fn is None:
        cpu_count_fn = multiprocessing.cpu_count

    runtime = copy.deepcopy(config["runtime"])
    train = copy.deepcopy(config["train"])

    runtime["device_request"] = normalize_device_request(runtime.get("device"))
    runtime["is_half_request"] = normalize_auto_bool(runtime.get("is_half"), "runtime.is_half")
    runtime["n_cpu_request"] = normalize_auto_int(runtime.get("n_cpu"), "runtime.n_cpu")
    runtime["deterministic_algorithms"] = normalize_deterministic_algorithms(
        runtime.get("deterministic_algorithms", "off")
    )
    runtime["disable_tf32"] = normalize_bool(
        runtime.get("disable_tf32", False), "runtime.disable_tf32"
    )
    workspace_config = runtime.get("cublas_workspace_config")
    runtime["cublas_workspace_config"] = (
        None if workspace_config in (None, "") else str(workspace_config)
    )

    slice_block = copy.deepcopy(runtime.get("slice") or {})
    runtime["slice"] = {
        "x_pad": normalize_slice_value(slice_block.get("x_pad"), "runtime.slice.x_pad"),
        "x_query": normalize_slice_value(slice_block.get("x_query"), "runtime.slice.x_query"),
        "x_center": normalize_slice_value(slice_block.get("x_center"), "runtime.slice.x_center"),
        "x_max": normalize_slice_value(slice_block.get("x_max"), "runtime.slice.x_max"),
    }

    train["fp16_run_request"] = normalize_auto_bool(train.get("fp16_run"), "train.fp16_run")
    train["numeric_backend"] = normalize_numeric_backend(train.get("numeric_backend", "native"))
    train["grad_scaler_init_scale"] = normalize_positive_float(
        train.get("grad_scaler_init_scale", DEFAULT_TRAIN["grad_scaler_init_scale"]),
        "train.grad_scaler_init_scale",
    )

    environment = detect_runtime_environment_fn(runtime["device_request"])
    runtime["device"] = environment["device"]
    runtime["profile"] = environment

    if runtime["n_cpu_request"] == "auto":
        runtime["n_cpu"] = cpu_count_fn()
    else:
        runtime["n_cpu"] = runtime["n_cpu_request"]

    if runtime["is_half_request"] == "auto":
        runtime["is_half"] = bool(
            runtime["device"].startswith("cuda") and environment["supports_half"]
        )
    elif runtime["is_half_request"] is True:
        if runtime["device"] == "cpu":
            raise RuntimeError("runtime.is_half=true requires a CUDA device")
        if not environment["supports_half"]:
            raise RuntimeError(
                f"runtime.is_half=true is not supported on GPU {environment['gpu_name']}"
            )
        runtime["is_half"] = True
    else:
        runtime["is_half"] = False

    if train["fp16_run_request"] == "auto":
        train["fp16_run"] = bool(
            runtime["device"].startswith("cuda") and environment["supports_half"]
        )
    elif train["fp16_run_request"] is True:
        if runtime["device"] == "cpu":
            raise RuntimeError("train.fp16_run=true requires a CUDA device")
        if not environment["supports_half"]:
            raise RuntimeError(
                f"train.fp16_run=true is not supported on GPU {environment['gpu_name']}"
            )
        train["fp16_run"] = True
    else:
        train["fp16_run"] = False

    if environment["gpu_mem_gb"] is not None and environment["gpu_mem_gb"] <= 4:
        auto_slice = {"x_pad": 1, "x_query": 5, "x_center": 30, "x_max": 32}
    elif runtime["is_half"]:
        auto_slice = {"x_pad": 3, "x_query": 10, "x_center": 60, "x_max": 65}
    else:
        auto_slice = {"x_pad": 1, "x_query": 6, "x_center": 38, "x_max": 41}

    resolved_slice: dict[str, int] = {}
    for key, value in runtime["slice"].items():
        resolved_slice[key] = auto_slice[key] if value == "auto" else int(value)
    runtime["slice"] = resolved_slice

    if train["numeric_backend"] == "deterministic_gpu":
        if runtime["device"] == "cpu":
            raise RuntimeError("train.numeric_backend=deterministic_gpu requires a CUDA device")
        if runtime["deterministic_algorithms"] == "off":
            runtime["deterministic_algorithms"] = "error"
        if runtime["cublas_workspace_config"] is None:
            runtime["cublas_workspace_config"] = ":4096:8"
        runtime["disable_tf32"] = True

    config["runtime"] = runtime
    config["train"] = train
    replayable_config = config.get("replayable_config")
    if isinstance(replayable_config, dict):
        replayable_train = replayable_config.get("train")
        if isinstance(replayable_train, dict):
            replayable_train["numeric_backend"] = train["numeric_backend"]
            replayable_train["grad_scaler_init_scale"] = train["grad_scaler_init_scale"]
        replayable_runtime = replayable_config.get("runtime")
        if isinstance(replayable_runtime, dict):
            replayable_runtime["deterministic_algorithms"] = runtime["deterministic_algorithms"]
            replayable_runtime["disable_tf32"] = runtime["disable_tf32"]
            replayable_runtime["cublas_workspace_config"] = runtime["cublas_workspace_config"]
    return config


def resolve_infer_paths(infer: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    result = copy.deepcopy(infer)
    if result.get("model_path") in (None, "", "auto"):
        result["model_path"] = paths["final_model_path"]
    if result.get("index_path") in (None, "", "auto"):
        result["index_path"] = paths["final_index_path"]
    return result
