import argparse
from multiprocessing import cpu_count
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
    parser.add_argument("-n", "--n_cpu", type=int, default=0)
    args = parser.parse_args()

    if args.config:
        if args.inp_root or args.output or args.n_cpu:
            parser.error("config mode only accepts --config, --hparams, and --reset")
        project = load_project_config(
            args.config,
            overrides=parse_hparams_overrides(args.hparams),
            reset=args.reset,
        )
        if project["feature_dim"] != 768:
            raise ValueError("src.index.build_v2 expects a v2/768-dim project.")
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
    n_cpu = args.n_cpu or (int(project["n_cpu"]) if args.config else cpu_count())
    if n_cpu < 1:
        parser.error("--n_cpu must be >= 1")
    return Path(inp_root), Path(output), index_dir, n_cpu


def build_index(inp_root, output, index_dir=None, n_cpu=None):
    return build_faiss_index(
        inp_root,
        output,
        feature_dim=768,
        index_dir=index_dir,
        n_cpu=n_cpu,
        shuffle=True,
        reduce_large=True,
        nprobe=1,
        add_batch_size=8192,
    )


def main():
    inp_root, output, index_dir, n_cpu = parse_args()
    build_index(inp_root, output, index_dir, n_cpu)


if __name__ == "__main__":
    main()

