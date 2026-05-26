from __future__ import annotations

import copy
import multiprocessing
import os
from pathlib import Path
from typing import Any

import yaml

try:
    import torch
except ImportError:  # pragma: no cover - torch is required at runtime
    torch = None


REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "configs"
SNAPSHOT_NAME = "config.yaml"

DEFAULT_PREPROCESS = {
    "noparallel": False,
}

DEFAULT_RUNTIME = {
    "device": "auto",
    "is_half": "auto",
    "n_cpu": "auto",
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
}

VALID_SAMPLE_RATES = {
    "v1": {"32k", "40k", "48k"},
    "v2": {"32k", "48k"},
}

PATH_FIELD_NAMES = (
    "work_dir",
    "data_root",
    "ckpt_root",
    "pretrain_root",
    "dataset_dir",
    "preprocess_dir",
    "train_dir",
    "export_dir",
    "index_dir",
    "final_model_name",
    "final_index_name",
)

ROOT_SECTIONS = {
    "selectors",
    "preprocess",
    "runtime",
    "infer",
    "train",
    "data",
    "model",
    "variants",
}

RUNTIME_AUTO_FIELDS = {
    "device_request",
    "is_half_request",
    "n_cpu_request",
    "profile",
}

SNAPSHOT_EXCLUDE_TOP_LEVEL = {
    "paths",
    "version",
    "sample_rate",
    "if_f0",
    "device",
    "is_half",
    "n_cpu",
    "x_pad",
    "x_query",
    "x_center",
    "x_max",
    "feature_dir",
    "feature_dim",
    "model_dir",
    "experiment_dir",
    "ckpt_dir",
    "training_files",
    "hubert_path",
    "rmvpe_path",
    "final_model_path",
    "final_index_path",
    "preprocess_log_path",
    "feature_log_path",
    "noparallel",
    "resolved_variant",
    "source_config_path",
    "snapshot_lookup_path",
    "replayable_config",
}

ALLOWED_TOP_LEVEL_KEYS = {"base_config", "name", *PATH_FIELD_NAMES, *ROOT_SECTIONS}

REMOVED_TOP_LEVEL_FIELD_HINTS = {
    "paths": "Move path fields to the YAML top level.",
    "version": "Use selectors.version instead.",
    "sample_rate": "Use selectors.sample_rate instead.",
    "if_f0": "Use selectors.if_f0 instead.",
    "noparallel": "Use preprocess.noparallel instead.",
    "device": "Use runtime.device instead.",
    "is_half": "Use runtime.is_half instead.",
    "n_cpu": "Use runtime.n_cpu instead.",
    "fp16_run": "Use train.fp16_run instead.",
    "save_every_epoch": "Use train.save_every_epoch instead.",
    "save_every_weights": "Use train.save_every_weights instead.",
    "if_latest": "Use train.if_latest instead.",
    "if_cache_data_in_gpu": "Use train.if_cache_data_in_gpu instead.",
    "pretrainG": "Use train.pretrainG instead.",
    "pretrainD": "Use train.pretrainD instead.",
    "train_common": "Move these fields into train.",
    "data_common": "Move these fields into data.",
    "model_common": "Move these fields into model.",
    "ckpt_dir": "Use work_dir instead.",
    "experiment_dir": "Use work_dir instead.",
}


class HparamsParseError(ValueError):
    pass


def _top_level_key_error(key: str, source: str) -> ValueError:
    hint = REMOVED_TOP_LEVEL_FIELD_HINTS.get(key)
    if hint is not None:
        return ValueError(f"Legacy top-level key {key!r} is not supported in {source}. {hint}")
    return ValueError(
        f"Unsupported top-level key {key!r} in {source}. "
        "Only base_config, name, top-level path fields, and "
        "selectors/preprocess/runtime/infer/train/data/model/variants are supported."
    )


def _validate_config_mapping(raw: dict[str, Any], source: str) -> None:
    for key in raw:
        if key not in ALLOWED_TOP_LEVEL_KEYS:
            raise _top_level_key_error(key, source)
        if key in ROOT_SECTIONS and not isinstance(raw[key], dict):
            raise ValueError(f"Top-level key {key!r} in {source} must be a mapping")


def _read_yaml_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Project config must contain a mapping: {path}")
    _validate_config_mapping(data, str(path))
    return data



def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result



def _resolve_config_path(ref: str | Path, relative_to: Path | None = None) -> Path:
    path = Path(ref)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if relative_to is not None and str(ref).startswith("."):
            candidates.append((relative_to / path).resolve())
            if not path.suffix:
                candidates.append((relative_to / f"{path.name}.yaml").resolve())
                candidates.append((relative_to / f"{path.name}.yml").resolve())
        candidates.extend(
            [
                (REPO_ROOT / path).resolve(),
                (CONFIG_ROOT / path).resolve(),
            ]
        )
        if not path.suffix:
            candidates.extend(
                [
                    (REPO_ROOT / f"{path}.yaml").resolve(),
                    (REPO_ROOT / f"{path}.yml").resolve(),
                    (CONFIG_ROOT / f"{path.name}.yaml").resolve(),
                    (CONFIG_ROOT / f"{path.name}.yml").resolve(),
                    (CONFIG_ROOT / f"{path}.yaml").resolve(),
                    (CONFIG_ROOT / f"{path}.yml").resolve(),
                ]
            )
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.is_file():
            if candidate.suffix.lower() not in {".yaml", ".yml"}:
                raise ValueError(f"Project config must be a YAML file, got: {candidate}")
            return candidate
    raise FileNotFoundError(f"Project config not found: {ref}")



def _normalize_base_refs(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        refs: list[str] = []
        for item in value:
            if not isinstance(item, str) or item == "":
                raise ValueError("base_config entries must be non-empty strings")
            refs.append(item)
        return refs
    raise ValueError("base_config must be a string or list of strings")



def _load_config_chain(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Detected base_config cycle: {chain}")
    raw = _read_yaml_file(path)
    merged: dict[str, Any] = {}
    for base_ref in _normalize_base_refs(raw.get("base_config")):
        base_path = _resolve_config_path(base_ref, relative_to=path.parent)
        merged = _deep_merge(merged, _load_config_chain(base_path, (*stack, path)))
    current = copy.deepcopy(raw)
    current.pop("base_config", None)
    return _deep_merge(merged, current)



def _normalize_sample_rate(value: Any) -> str:
    if value in (None, ""):
        raise ValueError("selectors.sample_rate is required")
    if isinstance(value, (int, float)):
        value = int(value)
        if value in {32, 40, 48}:
            return f"{value}k"
        if value >= 1000 and value % 1000 == 0:
            return f"{value // 1000}k"
    value = str(value).strip().lower()
    if value.endswith("khz"):
        value = f"{value[:-3]}k"
    if value.endswith("k"):
        return value
    if value.isdigit():
        number = int(value)
        if number in {32, 40, 48}:
            return f"{number}k"
        if number >= 1000 and number % 1000 == 0:
            return f"{number // 1000}k"
    return value



def _normalize_version(value: Any) -> str:
    if value in (None, ""):
        raise ValueError("selectors.version is required")
    version = str(value).strip().lower()
    if version not in VALID_SAMPLE_RATES:
        raise ValueError(
            f"Unsupported selectors.version={value!r}. Expected one of: "
            f"{', '.join(sorted(VALID_SAMPLE_RATES))}"
        )
    return version



def _normalize_if_f0(value: Any) -> int:
    if value in (None, ""):
        raise ValueError("selectors.if_f0 is required")
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float)):
        number = int(value)
        if number in {0, 1}:
            return number
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return 1
    if text in {"0", "false", "no", "n"}:
        return 0
    raise ValueError(f"selectors.if_f0 must be 0 or 1, got: {value!r}")



def _normalize_auto_bool(value: Any, field_name: str) -> str | bool:
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



def _normalize_auto_int(value: Any, field_name: str) -> str | int:
    if value in (None, "", "auto"):
        return "auto"
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be auto or integer, got: {value!r}") from exc
    if number < 1:
        raise ValueError(f"{field_name} must be >= 1, got: {number}")
    return number



def _normalize_slice_value(value: Any, field_name: str) -> str | int:
    if value in (None, "", "auto"):
        return "auto"
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be auto or integer, got: {value!r}") from exc



def _normalize_device_request(value: Any) -> str:
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



def _expand_path(value: Any) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))



def _resolve_dir(value: Any, default: Path) -> Path:
    if value in (None, ""):
        return default.resolve()
    path = _expand_path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()

def _extract_path_config(raw: dict[str, Any]) -> dict[str, Any]:
    path_config: dict[str, Any] = {}
    for field in PATH_FIELD_NAMES:
        if field in raw:
            path_config[field] = copy.deepcopy(raw[field])
    return path_config



def _normalize_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if not overrides:
        return normalized

    for key, value in overrides.items():
        if value is None:
            continue
        if key in ROOT_SECTIONS and isinstance(value, dict):
            normalized[key] = _deep_merge(normalized.get(key, {}), value)
            continue
        if key == "base_config":
            normalized[key] = copy.deepcopy(value)
            continue
        if key in PATH_FIELD_NAMES:
            normalized[key] = copy.deepcopy(value)
            continue
        if "." in key:
            cursor = normalized
            parts = key.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
            continue
        normalized[key] = copy.deepcopy(value)
    _validate_config_mapping(normalized, "--hparams")
    return normalized



def _split_hparams_pairs(text: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quote: str | None = None
    escape = False
    for char in text:
        if escape:
            current.append(char)
            escape = False
            continue
        if char == "\\":
            current.append(char)
            escape = True
            continue
        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            continue
        if char in {'"', "'"}:
            current.append(char)
            quote = char
            continue
        if char == ",":
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    part = "".join(current).strip()
    if part:
        parts.append(part)
    return parts



def _parse_hparams_scalar(raw_value: str) -> Any:
    if raw_value == "":
        return ""
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise HparamsParseError(f"Invalid --hparams value: {raw_value!r}") from exc
    if isinstance(value, (dict, list)):
        raise HparamsParseError(
            f"--hparams only supports scalar values, got {type(value).__name__} for {raw_value!r}"
        )
    return value



def parse_hparams_overrides(text: str | None) -> dict[str, Any]:
    if text in (None, ""):
        return {}
    overrides: dict[str, Any] = {}
    for pair in _split_hparams_pairs(text):
        if "=" not in pair:
            raise HparamsParseError(f"Invalid --hparams entry: {pair!r}")
        key, raw_value = pair.split("=", 1)
        key = key.strip()
        if key == "":
            raise HparamsParseError(f"Invalid --hparams entry: {pair!r}")
        value = _parse_hparams_scalar(raw_value.strip())
        cursor = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            existing = cursor.get(part)
            if existing is None:
                cursor[part] = {}
                existing = cursor[part]
            if not isinstance(existing, dict):
                raise HparamsParseError(f"Conflicting --hparams key: {key!r}")
            cursor = existing
        leaf = parts[-1]
        if leaf in cursor and isinstance(cursor[leaf], dict):
            raise HparamsParseError(f"Conflicting --hparams key: {key!r}")
        cursor[leaf] = value
    return overrides



def _build_selectors(raw: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(raw.get("selectors") or {})



def _build_preprocess(raw: dict[str, Any]) -> dict[str, Any]:
    preprocess = _deep_merge(
        DEFAULT_PREPROCESS,
        copy.deepcopy(raw.get("preprocess") or {}),
    )
    allowed_keys = set(DEFAULT_PREPROCESS) | {"f0method"}
    return {key: value for key, value in preprocess.items() if key in allowed_keys}



def _build_runtime(raw: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(DEFAULT_RUNTIME, copy.deepcopy(raw.get("runtime") or {}))



def _build_infer(raw: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(DEFAULT_INFER, copy.deepcopy(raw.get("infer") or {}))



def _build_train_request(raw: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(DEFAULT_TRAIN, copy.deepcopy(raw.get("train") or {}))



def _build_data_request(raw: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(raw.get("data") or {})



def _build_model_request(raw: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(raw.get("model") or {})



def _detect_runtime_environment(device_request: str) -> dict[str, Any]:
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



def _resolve_runtime_profile(config: dict[str, Any]) -> dict[str, Any]:
    runtime = copy.deepcopy(config["runtime"])
    train = copy.deepcopy(config["train"])

    runtime["device_request"] = _normalize_device_request(runtime.get("device"))
    runtime["is_half_request"] = _normalize_auto_bool(runtime.get("is_half"), "runtime.is_half")
    runtime["n_cpu_request"] = _normalize_auto_int(runtime.get("n_cpu"), "runtime.n_cpu")

    slice_block = copy.deepcopy(runtime.get("slice") or {})
    runtime["slice"] = {
        "x_pad": _normalize_slice_value(slice_block.get("x_pad"), "runtime.slice.x_pad"),
        "x_query": _normalize_slice_value(slice_block.get("x_query"), "runtime.slice.x_query"),
        "x_center": _normalize_slice_value(slice_block.get("x_center"), "runtime.slice.x_center"),
        "x_max": _normalize_slice_value(slice_block.get("x_max"), "runtime.slice.x_max"),
    }

    train["fp16_run_request"] = _normalize_auto_bool(train.get("fp16_run"), "train.fp16_run")

    environment = _detect_runtime_environment(runtime["device_request"])
    runtime["device"] = environment["device"]
    runtime["profile"] = environment

    if runtime["n_cpu_request"] == "auto":
        runtime["n_cpu"] = multiprocessing.cpu_count()
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

    config["runtime"] = runtime
    config["train"] = train
    return config



def _resolve_paths(path_config: dict[str, Any], name: str) -> dict[str, str]:
    data_root = _resolve_dir(path_config.get("data_root"), REPO_ROOT / "data")
    ckpt_root = _resolve_dir(path_config.get("ckpt_root"), REPO_ROOT / "ckpt")
    pretrain_root = _resolve_dir(path_config.get("pretrain_root"), REPO_ROOT / "pretrain")
    work_dir = _resolve_dir(path_config.get("work_dir"), ckpt_root / name)

    dataset_dir = _resolve_dir(path_config.get("dataset_dir"), data_root / name / "dataset")
    preprocess_dir = _resolve_dir(
        path_config.get("preprocess_dir"), data_root / name / "preprocess_data"
    )
    train_dir = _resolve_dir(path_config.get("train_dir"), work_dir / "train")
    export_dir = _resolve_dir(path_config.get("export_dir"), work_dir / "export")
    index_dir = _resolve_dir(path_config.get("index_dir"), work_dir / "index")

    final_model_name = str(path_config.get("final_model_name") or f"{name}.pth")
    final_index_name = str(path_config.get("final_index_name") or f"{name}.index")

    return {
        "work_dir": str(work_dir),
        "ckpt_dir": str(work_dir),
        "data_root": str(data_root),
        "ckpt_root": str(ckpt_root),
        "pretrain_root": str(pretrain_root),
        "dataset_dir": str(dataset_dir),
        "preprocess_dir": str(preprocess_dir),
        "train_dir": str(train_dir),
        "export_dir": str(export_dir),
        "index_dir": str(index_dir),
        "training_files": str(preprocess_dir / "filelist.txt"),
        "preprocess_log_path": str(preprocess_dir / "preprocess.log"),
        "feature_log_path": str(preprocess_dir / "extract_f0_feature.log"),
        "final_model_name": final_model_name,
        "final_index_name": final_index_name,
        "final_model_path": str(export_dir / final_model_name),
        "final_index_path": str(index_dir / final_index_name),
        "hubert_path": str(pretrain_root / "hubert" / "hubert_base.pt"),
        "rmvpe_path": str(pretrain_root / "rmvpe" / "rmvpe.pt"),
    }



def _resolve_infer_paths(infer: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    result = copy.deepcopy(infer)
    if result.get("model_path") in (None, "", "auto"):
        result["model_path"] = paths["final_model_path"]
    if result.get("index_path") in (None, "", "auto"):
        result["index_path"] = paths["final_index_path"]
    return result



def _flatten_aliases(config: dict[str, Any]) -> dict[str, Any]:
    runtime = config["runtime"]
    paths = config["paths"]
    selectors = config["selectors"]

    config["version"] = selectors["version"]
    config["sample_rate"] = selectors["sample_rate"]
    config["if_f0"] = selectors["if_f0"]

    for key, value in paths.items():
        config[key] = value

    config["device"] = runtime["device"]
    config["is_half"] = runtime["is_half"]
    config["n_cpu"] = runtime["n_cpu"]
    config["x_pad"] = runtime["slice"]["x_pad"]
    config["x_query"] = runtime["slice"]["x_query"]
    config["x_center"] = runtime["slice"]["x_center"]
    config["x_max"] = runtime["slice"]["x_max"]

    feature_dir_name = "3_feature256" if selectors["version"] == "v1" else "3_feature768"
    config["feature_dir"] = str(Path(paths["preprocess_dir"]) / feature_dir_name)
    config["feature_dim"] = 256 if selectors["version"] == "v1" else 768
    config["model_dir"] = paths["train_dir"]
    config["experiment_dir"] = paths["work_dir"]
    config["noparallel"] = config["preprocess"]["noparallel"]
    return config



def _build_snapshot_config(
    name: str,
    selectors: dict[str, Any],
    preprocess: dict[str, Any],
    runtime: dict[str, Any],
    infer: dict[str, Any],
    train: dict[str, Any],
    data: dict[str, Any],
    model: dict[str, Any],
    variants: dict[str, Any],
    paths: dict[str, str],
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "base_config": [],
        "name": name,
        "work_dir": paths["work_dir"],
        "data_root": paths["data_root"],
        "ckpt_root": paths["ckpt_root"],
        "pretrain_root": paths["pretrain_root"],
        "dataset_dir": paths["dataset_dir"],
        "preprocess_dir": paths["preprocess_dir"],
        "train_dir": paths["train_dir"],
        "export_dir": paths["export_dir"],
        "index_dir": paths["index_dir"],
        "final_model_name": paths["final_model_name"],
        "final_index_name": paths["final_index_name"],
        "selectors": copy.deepcopy(selectors),
        "preprocess": copy.deepcopy(preprocess),
        "runtime": copy.deepcopy(runtime),
        "infer": copy.deepcopy(infer),
        "train": copy.deepcopy(train),
        "data": copy.deepcopy(data),
        "model": copy.deepcopy(model),
        "variants": copy.deepcopy(variants),
    }
    return snapshot



def _sanitize_snapshot_payload(payload: dict[str, Any]) -> dict[str, Any]:
    snapshot = copy.deepcopy(payload)
    snapshot["base_config"] = []

    runtime = snapshot.get("runtime")
    if isinstance(runtime, dict):
        for key in RUNTIME_AUTO_FIELDS:
            runtime.pop(key, None)

    train = snapshot.get("train")
    if isinstance(train, dict):
        train.pop("fp16_run_request", None)

    for key in list(snapshot.keys()):
        if key in SNAPSHOT_EXCLUDE_TOP_LEVEL:
            snapshot.pop(key, None)

    return snapshot



def load_project_config(
    ref: str | Path,
    overrides: dict[str, Any] | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    source_path = _resolve_config_path(ref)
    merged_raw = _load_config_chain(source_path)

    initial_name = merged_raw.get("name") or source_path.stem
    initial_paths = _resolve_paths(_extract_path_config(merged_raw), initial_name)
    snapshot_lookup_path = Path(initial_paths["work_dir"]) / SNAPSHOT_NAME
    if not reset and snapshot_lookup_path.exists() and snapshot_lookup_path.is_file():
        merged_raw = _deep_merge(merged_raw, _read_yaml_file(snapshot_lookup_path))

    if isinstance(overrides, str):
        overrides = parse_hparams_overrides(overrides)
    override_map = _normalize_overrides(overrides)
    merged_raw = _deep_merge(merged_raw, override_map)

    name = merged_raw.get("name") or source_path.stem
    selectors = _build_selectors(merged_raw)
    version = _normalize_version(selectors.get("version"))
    sample_rate = _normalize_sample_rate(selectors.get("sample_rate"))
    if_f0 = _normalize_if_f0(selectors.get("if_f0"))
    if sample_rate not in VALID_SAMPLE_RATES[version]:
        valid = ", ".join(sorted(VALID_SAMPLE_RATES[version]))
        raise ValueError(
            f"Invalid selector combination: version={version}, sample_rate={sample_rate}. Supported sample rates for {version}: {valid}"
        )

    variants = copy.deepcopy(merged_raw.get("variants") or {})
    if not isinstance(variants, dict):
        raise ValueError("variants must be a mapping")
    try:
        variant_patch = copy.deepcopy(variants[version][sample_rate])
    except KeyError as exc:
        raise ValueError(
            f"Missing variant definition for version={version}, sample_rate={sample_rate}"
        ) from exc

    preprocess = _build_preprocess(merged_raw)
    runtime = _build_runtime(merged_raw)
    infer_request = _build_infer(merged_raw)
    train = _deep_merge(_build_train_request(merged_raw), copy.deepcopy(variant_patch.get("train") or {}))
    data = _deep_merge(_build_data_request(merged_raw), copy.deepcopy(variant_patch.get("data") or {}))
    model = _deep_merge(_build_model_request(merged_raw), copy.deepcopy(variant_patch.get("model") or {}))

    paths = _resolve_paths(_extract_path_config(merged_raw), name)
    infer = _resolve_infer_paths(infer_request, paths)

    resolved_selectors = {
        "version": version,
        "sample_rate": sample_rate,
        "if_f0": if_f0,
    }
    replayable_config = _build_snapshot_config(
        name,
        resolved_selectors,
        preprocess,
        runtime,
        infer_request,
        train,
        data,
        model,
        variants,
        paths,
    )

    config: dict[str, Any] = {
        "name": name,
        "selectors": resolved_selectors,
        "paths": paths,
        "preprocess": preprocess,
        "runtime": runtime,
        "infer": infer,
        "train": train,
        "data": data,
        "model": model,
        "variants": variants,
        "source_config_path": str(source_path),
        "snapshot_lookup_path": str(snapshot_lookup_path),
        "resolved_variant": f"{version}/{sample_rate}",
        "replayable_config": replayable_config,
    }

    config["data"]["training_files"] = paths["training_files"]
    config = _resolve_runtime_profile(config)
    config = _flatten_aliases(config)
    return config



def save_project_config_snapshot(config: dict[str, Any], target_path: str | Path) -> Path:
    if "replayable_config" in config and isinstance(config["replayable_config"], dict):
        payload = config["replayable_config"]
    else:
        payload = config
    sanitized = _sanitize_snapshot_payload(payload)
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        yaml.safe_dump(
            sanitized,
            handle,
            allow_unicode=True,
            sort_keys=False,
        )
    return target
