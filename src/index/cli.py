from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

from configs.project_config import load_project_config, parse_hparams_overrides
from src.index.builder import build_faiss_index

INDEX_BUILD_PROFILES = {
    256: {
        "uses_n_cpu": False,
        "build_kwargs": {
            "nprobe": 9,
        },
    },
    768: {
        "uses_n_cpu": True,
        "build_kwargs": {
            "shuffle": True,
            "reduce_large": True,
            "nprobe": 1,
            "add_batch_size": 8192,
        },
    },
}


@dataclass(frozen=True)
class IndexBuildRequest:
    inp_root: Path
    output: Path
    index_dir: Path
    feature_dim: int
    n_cpu: int | None = None


def _feature_layout_for_version(version: str) -> tuple[str, int]:
    if version == "v1":
        return "3_feature256", 256
    return "3_feature768", 768


def build_parser(*, include_feature_dim: bool = False, include_n_cpu: bool = False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("-i", "--inp_root", type=str, default="")
    parser.add_argument("-o", "--output", type=str, default="")
    if include_feature_dim:
        parser.add_argument("--feature-dim", type=int, choices=[256, 768], default=0)
    if include_n_cpu:
        parser.add_argument("-n", "--n_cpu", type=int, default=0)
    return parser


def resolve_project_request(
    args,
    *,
    include_n_cpu: bool = False,
):
    project = load_project_config(
        args.config,
        overrides=parse_hparams_overrides(args.hparams),
        reset=args.reset,
    )
    feature_dir_name, feature_dim = _feature_layout_for_version(
        str(project["selectors"]["version"])
    )
    paths = project["paths"]
    n_cpu = None
    if include_n_cpu and INDEX_BUILD_PROFILES[feature_dim]["uses_n_cpu"]:
        n_cpu = int(project["runtime"]["n_cpu"])
    return IndexBuildRequest(
        inp_root=Path(paths["preprocess_dir"]) / feature_dir_name,
        output=Path(paths["final_index_path"]),
        index_dir=Path(paths["index_dir"]),
        feature_dim=feature_dim,
        n_cpu=n_cpu,
    )


def resolve_manual_request(
    args,
    parser,
    *,
    include_n_cpu: bool = False,
):
    resolved_feature_dim = int(args.feature_dim)
    if args.inp_root == "" or args.output == "":
        parser.error("manual mode requires --inp_root, --output, and --feature-dim")

    n_cpu = None
    if include_n_cpu and INDEX_BUILD_PROFILES[resolved_feature_dim]["uses_n_cpu"]:
        n_cpu = int(args.n_cpu) if int(args.n_cpu) > 0 else cpu_count()
        if n_cpu < 1:
            parser.error("--n_cpu must be >= 1")

    return IndexBuildRequest(
        inp_root=Path(args.inp_root),
        output=Path(args.output),
        index_dir=Path(args.output).resolve().parent,
        feature_dim=resolved_feature_dim,
        n_cpu=n_cpu,
    )


def build_index(request: IndexBuildRequest):
    profile = INDEX_BUILD_PROFILES.get(int(request.feature_dim))
    if profile is None:
        raise ValueError(f"Unsupported feature_dim: {request.feature_dim}")

    request.index_dir.mkdir(parents=True, exist_ok=True)
    build_kwargs = {
        "feature_dim": int(request.feature_dim),
        "index_dir": request.index_dir,
        **profile["build_kwargs"],
    }
    if profile["uses_n_cpu"]:
        build_kwargs["n_cpu"] = request.n_cpu
    return build_faiss_index(request.inp_root, request.output, **build_kwargs)
