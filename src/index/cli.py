from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path

from configs.project_config import load_project_config, parse_hparams_overrides
from src.index.builder import build_faiss_index
from src.rvc_profiles import get_feature_profile, get_feature_profile_by_dim


@dataclass(frozen=True)
class IndexBuildRequest:
    inp_root: Path
    output: Path
    index_dir: Path
    feature_dim: int
    n_cpu: int | None = None


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("-i", "--inp_root", type=str, default="")
    parser.add_argument("-o", "--output", type=str, default="")
    parser.add_argument("--feature-dim", type=int, choices=[256, 768], default=0)
    parser.add_argument("-n", "--n_cpu", type=int, default=0)
    return parser


def resolve_project_request(args):
    project = load_project_config(
        args.config,
        overrides=parse_hparams_overrides(args.hparams),
        reset=args.reset,
    )
    profile = get_feature_profile(str(project["selectors"]["version"]))
    paths = project["paths"]
    n_cpu = None
    if profile.index_uses_n_cpu:
        n_cpu = int(project["runtime"]["n_cpu"])
    return IndexBuildRequest(
        inp_root=Path(paths["preprocess_dir"]) / profile.feature_dir_name,
        output=Path(paths["final_index_path"]),
        index_dir=Path(paths["index_dir"]),
        feature_dim=profile.feature_dim,
        n_cpu=n_cpu,
    )


def resolve_manual_request(args, parser):
    profile = get_feature_profile_by_dim(args.feature_dim)
    if args.inp_root == "" or args.output == "":
        parser.error("manual mode requires --inp_root, --output, and --feature-dim")

    n_cpu = None
    if profile.index_uses_n_cpu:
        n_cpu = int(args.n_cpu) if int(args.n_cpu) > 0 else cpu_count()
        if n_cpu < 1:
            parser.error("--n_cpu must be >= 1")

    return IndexBuildRequest(
        inp_root=Path(args.inp_root),
        output=Path(args.output),
        index_dir=Path(args.output).resolve().parent,
        feature_dim=profile.feature_dim,
        n_cpu=n_cpu,
    )


def build_index(request: IndexBuildRequest):
    profile = get_feature_profile_by_dim(request.feature_dim)
    request.index_dir.mkdir(parents=True, exist_ok=True)
    build_kwargs = {
        "feature_dim": int(request.feature_dim),
        "index_dir": request.index_dir,
        **dict(profile.index_build_kwargs),
    }
    if profile.index_uses_n_cpu:
        build_kwargs["n_cpu"] = request.n_cpu
    return build_faiss_index(request.inp_root, request.output, **build_kwargs)


def parse_args(argv: list[str] | None = None) -> IndexBuildRequest:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.config:
        if args.inp_root or args.output or args.feature_dim or args.n_cpu:
            parser.error("config mode only accepts --config, --hparams, and --reset")
        return resolve_project_request(args)

    if args.inp_root == "" or args.output == "" or args.feature_dim == 0:
        parser.error("manual mode requires --inp_root, --output, and --feature-dim")
    return resolve_manual_request(args, parser)


def main(argv: list[str] | None = None):
    return build_index(parse_args(argv))
