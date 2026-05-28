from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

SNAPSHOT_NAME = "config.yaml"

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
    "validation_files",
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


def build_snapshot_config(
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
    return {
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


def sanitize_snapshot_payload(payload: dict[str, Any]) -> dict[str, Any]:
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


def save_project_config_snapshot(config: dict[str, Any], target_path: str | Path) -> Path:
    if "replayable_config" in config and isinstance(config["replayable_config"], dict):
        payload = config["replayable_config"]
    else:
        payload = config
    sanitized = sanitize_snapshot_payload(payload)
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
