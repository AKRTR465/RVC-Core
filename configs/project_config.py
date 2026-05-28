from __future__ import annotations

import copy
import multiprocessing
from pathlib import Path
from typing import Any

from configs.project_loading import (
    HparamsParseError,
    deep_merge as _deep_merge,
    load_config_chain as _load_config_chain,
    normalize_overrides as _normalize_overrides,
    parse_hparams_overrides,
    read_yaml_file as _read_yaml_file,
)
from configs.project_paths import (
    extract_path_config,
    resolve_config_path,
    resolve_paths,
)
from configs.project_sections import (
    VALID_SAMPLE_RATES,
    build_data_request as _build_data_request,
    build_infer as _build_infer,
    build_model_request as _build_model_request,
    build_preprocess as _build_preprocess,
    build_runtime as _build_runtime,
    build_selectors as _build_selectors,
    build_train_request as _build_train_request,
    normalize_if_f0 as _normalize_if_f0,
    normalize_sample_rate as _normalize_sample_rate,
    normalize_version as _normalize_version,
)
from configs.project_runtime import (
    detect_runtime_environment as _project_detect_runtime_environment,
    resolve_infer_paths as _project_resolve_infer_paths,
    resolve_runtime_profile as _project_resolve_runtime_profile,
)
from configs.project_snapshot import (
    SNAPSHOT_NAME,
    build_snapshot_config,
    save_project_config_snapshot,
)

def _detect_runtime_environment(device_request: str) -> dict[str, Any]:
    return _project_detect_runtime_environment(device_request)


def _resolve_runtime_profile(config: dict[str, Any]) -> dict[str, Any]:
    return _project_resolve_runtime_profile(
        config,
        detect_runtime_environment_fn=_detect_runtime_environment,
        cpu_count_fn=multiprocessing.cpu_count,
    )


def _resolve_infer_paths(infer: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    return _project_resolve_infer_paths(infer, paths)



def load_project_config(
    ref: str | Path,
    overrides: dict[str, Any] | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    source_path = resolve_config_path(ref)
    merged_raw = _load_config_chain(source_path)

    initial_name = merged_raw.get("name") or source_path.stem
    initial_paths = resolve_paths(extract_path_config(merged_raw), initial_name)
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

    paths = resolve_paths(extract_path_config(merged_raw), name)
    infer = _resolve_infer_paths(infer_request, paths)

    resolved_selectors = {
        "version": version,
        "sample_rate": sample_rate,
        "if_f0": if_f0,
    }
    replayable_config = build_snapshot_config(
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
    config["data"]["validation_files"] = paths["validation_files"]
    config = _resolve_runtime_profile(config)
    return config
