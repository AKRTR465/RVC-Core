import argparse
import logging
import os
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.append(os.getcwd())

from configs.project_config import load_project_config, parse_hparams_overrides

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
                "train-index.py expects a v1/256-dim project. Use train-index-v2.py "
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


def main():
    inp_root, output, index_dir = parse_args()
    npys = []
    for name in sorted(list(os.listdir(inp_root))):
        phone = np.load(inp_root / name)
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    logger.debug(big_npy.shape)

    np.save(index_dir / "big_src_feature.npy", big_npy)

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


if __name__ == "__main__":
    main()
