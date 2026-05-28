from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from configs.project_paths import PATH_FIELD_NAMES, resolve_config_path

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

REMOVED_TRAIN_FIELD_HINTS = {
    "validation_split": "Use preprocess.validation_split instead.",
    "validation_seed": "Use preprocess.validation_seed instead.",
    "validation_every_epoch": "Validation now runs every train.save_every_epoch.",
    "validation_preview_index": "Validation now logs the full validation set.",
    "train_filelist": "Training filelists are generated during preprocessing.",
    "val_filelist": "Validation filelists are generated during preprocessing.",
    "mel_loss_device": "Use train.numeric_backend instead.",
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


def validate_config_mapping(raw: dict[str, Any], source: str) -> None:
    for key in raw:
        if key not in ALLOWED_TOP_LEVEL_KEYS:
            raise _top_level_key_error(key, source)
        if key in ROOT_SECTIONS and not isinstance(raw[key], dict):
            raise ValueError(f"Top-level key {key!r} in {source} must be a mapping")
    train = raw.get("train")
    if isinstance(train, dict):
        for key, hint in REMOVED_TRAIN_FIELD_HINTS.items():
            if key in train:
                raise ValueError(f"Unsupported train key {key!r} in {source}. {hint}")


def read_yaml_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Project config must contain a mapping: {path}")
    validate_config_mapping(data, str(path))
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


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


def load_config_chain(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Detected base_config cycle: {chain}")
    raw = read_yaml_file(path)
    merged: dict[str, Any] = {}
    for base_ref in _normalize_base_refs(raw.get("base_config")):
        base_path = resolve_config_path(base_ref, relative_to=path.parent)
        merged = deep_merge(merged, load_config_chain(base_path, (*stack, path)))
    current = copy.deepcopy(raw)
    current.pop("base_config", None)
    return deep_merge(merged, current)


def normalize_overrides(overrides: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if not overrides:
        return normalized

    for key, value in overrides.items():
        if value is None:
            continue
        if key in ROOT_SECTIONS and isinstance(value, dict):
            normalized[key] = deep_merge(normalized.get(key, {}), value)
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
    validate_config_mapping(normalized, "--hparams")
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
