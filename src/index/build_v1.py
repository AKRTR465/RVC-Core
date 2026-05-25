import argparse
import logging
from pathlib import Path

import faiss

from configs.project_config import load_project_config, parse_hparams_overrides
from src.index.common import load_feature_matrix, save_source_matrix

logger = logging.getLogger(__name__)


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
    inp_root = Path(inp_root)
    output = Path(output)
    index_dir = Path(index_dir) if index_dir is not None else output.resolve().parent
    index_dir.mkdir(parents=True, exist_ok=True)

    big_npy = load_feature_matrix(inp_root)
    logger.debug(big_npy.shape)

    save_source_matrix(index_dir, big_npy)

    n_ivf = max(1, min(512, big_npy.shape[0]))
    index = faiss.index_factory(256, f"IVF{n_ivf},Flat")
    logger.info("Training...")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 9
    index.train(big_npy)
    trained_path = index_dir / f"trained_{output.name}"
    faiss.write_index(index, str(trained_path))
    logger.info("Adding...")
    index.add(big_npy)
    faiss.write_index(index, str(output))
    return output


def main():
    inp_root, output, index_dir = parse_args()
    build_index(inp_root, output, index_dir)


if __name__ == "__main__":
    main()

