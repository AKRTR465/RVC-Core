import argparse
from pathlib import Path

from configs.project_config import load_project_config, parse_hparams_overrides
from src.index.build_v1 import build_index as build_v1_index
from src.index.build_v2 import build_index as build_v2_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("-i", "--inp_root", type=str, default="")
    parser.add_argument("-o", "--output", type=str, default="")
    parser.add_argument("--feature-dim", type=int, choices=[256, 768], default=0)
    parser.add_argument("-n", "--n_cpu", type=int, default=0)
    args = parser.parse_args()

    if args.config:
        if args.inp_root or args.output or args.feature_dim or args.n_cpu:
            parser.error("config mode only accepts --config, --hparams, and --reset")
        project = load_project_config(
            args.config,
            overrides=parse_hparams_overrides(args.hparams),
            reset=args.reset,
        )
        return (
            Path(project["feature_dir"]),
            Path(project["final_index_path"]),
            Path(project["index_dir"]),
            int(project["feature_dim"]),
            int(project["n_cpu"]),
        )

    if args.inp_root == "" or args.output == "" or args.feature_dim == 0:
        parser.error("manual mode requires --inp_root, --output, and --feature-dim")
    return (
        Path(args.inp_root),
        Path(args.output),
        Path(args.output).resolve().parent,
        int(args.feature_dim),
        int(args.n_cpu),
    )


def main():
    inp_root, output, index_dir, feature_dim, n_cpu = parse_args()
    if feature_dim == 256:
        build_v1_index(inp_root, output, index_dir)
    else:
        build_v2_index(inp_root, output, index_dir, n_cpu=n_cpu or None)


if __name__ == "__main__":
    main()
