import numpy as np
from pathlib import Path


def load_feature_matrix(feature_dir, feature_dim=None):
    root = Path(feature_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Feature directory does not exist: {root}")

    features = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".npy":
            continue
        feature = np.load(path, allow_pickle=False)
        if feature.ndim != 2:
            raise ValueError(f"Feature file must be 2-D: {path} got shape={feature.shape}")
        if feature_dim is not None and feature.shape[1] != feature_dim:
            raise ValueError(
                f"Feature dim mismatch in {path}: expected {feature_dim}, got {feature.shape[1]}"
            )
        if not np.isfinite(feature).all():
            raise ValueError(f"Feature file contains NaN or inf: {path}")
        features.append(np.ascontiguousarray(feature, dtype=np.float32))
    if not features:
        raise ValueError(f"No .npy feature files found under: {root}")
    matrix = np.concatenate(features, 0)
    if matrix.shape[0] == 0:
        raise ValueError(f"Feature matrix is empty under: {root}")
    return matrix


def save_source_matrix(index_dir, matrix):
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    np.save(index_path / "big_src_feature.npy", matrix)
