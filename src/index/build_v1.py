import argparse
from pathlib import Path

from configs.project_config import load_project_config, parse_hparams_overrides
from src.index.builder import build_faiss_index


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("-i", "--inp_root", type=str, default="")
    parser.add_argument("-o", "--output", type=str, default="")
    args = parser.parse_args()

    if args.config:
        if args.inp_root or args.output:
            parser.error("config mode only accepts --config, --hparams, and --reset")
        project = load_project_config(
            args.config,
            overrides=parse_hparams_overrides(args.hparams),
            reset=args.reset,
        )
        if project["feature_dim"] != 256:
            raise ValueError(
                "src.index.build_v1 expects a v1/256-dim project. Use src.index.build_v2 "
                "for v2/768-dim features."
            )
        inp_root = project["feature_dir"]
        output = project["final_index_path"]
        index_dir = Path(project["index_dir"])
    else:
        if args.inp_root == "" or args.output == "":
            parser.error("provide --config or both --inp_root and --output")
        inp_root = args.inp_root
        output = args.output
        index_dir = Path(output).resolve().parent

    index_dir.mkdir(parents=True, exist_ok=True)
    return Path(inp_root), Path(output), index_dir


def build_index(inp_root, output, index_dir=None):
    return build_faiss_index(
        inp_root,
        output,
        feature_dim=256,
        index_dir=index_dir,
        nprobe=9,
    )


def main():
    inp_root, output, index_dir = parse_args()
    build_index(inp_root, output, index_dir)


if __name__ == "__main__":
    main()

