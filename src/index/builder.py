import logging
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np

from src.index.common import load_feature_matrix, save_source_matrix

logger = logging.getLogger(__name__)


def build_faiss_index(
    inp_root,
    output,
    *,
    feature_dim,
    index_dir=None,
    n_cpu=None,
    shuffle=False,
    reduce_large=False,
    nprobe=1,
    add_batch_size=None,
):
    import faiss

    inp_root = Path(inp_root)
    output = Path(output)
    index_dir = Path(index_dir) if index_dir is not None else output.resolve().parent
    index_dir.mkdir(parents=True, exist_ok=True)

    n_cpu = n_cpu or cpu_count()
    if n_cpu < 1:
        raise ValueError("n_cpu must be >= 1")

    big_npy = load_feature_matrix(inp_root, feature_dim=feature_dim)
    if shuffle:
        big_npy = big_npy[np.random.permutation(big_npy.shape[0])]
    if reduce_large and big_npy.shape[0] > 2e5:
        from sklearn.cluster import MiniBatchKMeans

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
        except (RuntimeError, ValueError) as exc:
            logger.warning("KMeans reduction failed; using full matrix: %s", exc)

    save_source_matrix(index_dir, big_npy)

    n_ivf = _index_cluster_count(feature_dim, big_npy.shape[0])
    index = faiss.index_factory(feature_dim, f"IVF{n_ivf},Flat")
    logger.info("Training...")
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = nprobe
    index.train(big_npy)

    trained_path = index_dir / f"trained_{output.name}"
    faiss.write_index(index, str(trained_path))

    logger.info("Adding...")
    if add_batch_size is None:
        index.add(big_npy)
    else:
        for i in range(0, big_npy.shape[0], add_batch_size):
            index.add(big_npy[i : i + add_batch_size])
    faiss.write_index(index, str(output))
    return output


def _index_cluster_count(feature_dim, rows):
    if feature_dim == 256:
        return max(1, min(512, rows))
    if feature_dim == 768:
        return max(1, min(int(16 * np.sqrt(rows)), rows // 39))
    raise ValueError(f"Unsupported feature_dim: {feature_dim}")
