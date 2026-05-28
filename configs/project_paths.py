from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO_ROOT / "configs"

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


def resolve_config_path(ref: str | Path, relative_to: Path | None = None) -> Path:
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


def _expand_path(value: Any) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value))))


def _resolve_dir(value: Any, default: Path) -> Path:
    if value in (None, ""):
        return default.resolve()
    path = _expand_path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def extract_path_config(raw: dict[str, Any]) -> dict[str, Any]:
    path_config: dict[str, Any] = {}
    for field in PATH_FIELD_NAMES:
        if field in raw:
            path_config[field] = copy.deepcopy(raw[field])
    return path_config


def resolve_paths(path_config: dict[str, Any], name: str) -> dict[str, str]:
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
        "training_files": str(preprocess_dir / "train_filelist.txt"),
        "validation_files": str(preprocess_dir / "val_filelist.txt"),
        "preprocess_log_path": str(preprocess_dir / "preprocess.log"),
        "feature_log_path": str(preprocess_dir / "extract_f0_feature.log"),
        "final_model_name": final_model_name,
        "final_index_name": final_index_name,
        "final_model_path": str(export_dir / final_model_name),
        "final_index_path": str(index_dir / final_index_name),
        "hubert_path": str(pretrain_root / "hubert" / "hubert_base.pt"),
        "rmvpe_path": str(pretrain_root / "rmvpe" / "rmvpe.pt"),
    }
