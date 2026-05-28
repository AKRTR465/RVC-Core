from __future__ import annotations

import copy
from typing import Any

from configs.project_loading import deep_merge
from configs.project_runtime import DEFAULT_INFER, DEFAULT_RUNTIME, DEFAULT_TRAIN

DEFAULT_PREPROCESS = {
    "noparallel": False,
    "validation_split": 0.1,
    "validation_seed": 1234,
}

VALID_SAMPLE_RATES = {
    "v1": {"32k", "40k", "48k"},
    "v2": {"32k", "48k"},
}


def normalize_sample_rate(value: Any) -> str:
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


def normalize_version(value: Any) -> str:
    if value in (None, ""):
        raise ValueError("selectors.version is required")
    version = str(value).strip().lower()
    if version not in VALID_SAMPLE_RATES:
        raise ValueError(
            f"Unsupported selectors.version={value!r}. Expected one of: "
            f"{', '.join(sorted(VALID_SAMPLE_RATES))}"
        )
    return version


def normalize_if_f0(value: Any) -> int:
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


def build_selectors(raw: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(raw.get("selectors") or {})


def build_preprocess(raw: dict[str, Any]) -> dict[str, Any]:
    preprocess = deep_merge(
        DEFAULT_PREPROCESS,
        copy.deepcopy(raw.get("preprocess") or {}),
    )
    allowed_keys = set(DEFAULT_PREPROCESS) | {"f0method"}
    return {key: value for key, value in preprocess.items() if key in allowed_keys}


def build_runtime(raw: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(DEFAULT_RUNTIME, copy.deepcopy(raw.get("runtime") or {}))


def build_infer(raw: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(DEFAULT_INFER, copy.deepcopy(raw.get("infer") or {}))


def build_train_request(raw: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(DEFAULT_TRAIN, copy.deepcopy(raw.get("train") or {}))


def build_data_request(raw: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(raw.get("data") or {})


def build_model_request(raw: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(raw.get("model") or {})
