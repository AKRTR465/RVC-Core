from pathlib import Path

import numpy as np


def load_feature_matrix(feature_dir):
    root = Path(feature_dir)
    features = [np.load(path) for path in sorted(root.iterdir())]
    if not features:
        raise ValueError(f"No feature files found under: {root}")
    return np.concatenate(features, 0)


def save_source_matrix(index_dir, matrix):
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    np.save(index_path / "big_src_feature.npy", matrix)
