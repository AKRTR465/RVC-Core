import argparse
import logging
import os
import sys
import traceback
from multiprocessing import cpu_count
from pathlib import Path

import faiss
import numpy as np
from sklearn.cluster import MiniBatchKMeans

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
            raise ValueError("train-index-v2.py expects a v2/768-dim project.")
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
    return Path(inp_root), Path(output), index_dir, n_cpu


def main():
    inp_root, output, index_dir, n_cpu = parse_args()
    npys = []
    for name in sorted(list(os.listdir(inp_root))):
        phone = np.load(inp_root / name)
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    logger.debug(big_npy.shape)

    if big_npy.shape[0] > 2e5:
        logger.info("Trying doing kmeans %s shape to 10k centers.", big_npy.shape[0])
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * n_cpu,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception:
            logger.warning(traceback.format_exc())

    np.save(index_dir / "big_src_feature.npy", big_npy)

    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(768, f"IVF{n_ivf},Flat")
    logger.info("Training...")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    trained_path = index_dir / f"trained_{output.name}"
    faiss.write_index(index, str(trained_path))
    logger.info("Adding...")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(index, str(output))


if __name__ == "__main__":
    main()
