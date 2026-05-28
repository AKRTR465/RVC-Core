import numpy as np
from pathlib import Path

SOURCE_MATRIX_NAME = "big_src_feature.npy"


def _validate_feature_array(matrix, source, *, feature_dim=None):
    if matrix.ndim != 2:
        raise ValueError(f"Feature matrix must be 2-D: {source} got shape={matrix.shape}")
    if feature_dim is not None and matrix.shape[1] != feature_dim:
        raise ValueError(
            f"Feature dim mismatch in {source}: expected {feature_dim}, got {matrix.shape[1]}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"Feature matrix contains NaN or inf: {source}")
    if matrix.shape[0] == 0:
        raise ValueError(f"Feature matrix is empty under: {source}")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def load_feature_matrix(feature_dir, feature_dim=None):
    root = Path(feature_dir)
    if not root.is_dir():
        raise NotADirectoryError(f"Feature directory does not exist: {root}")

    features = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".npy":
            continue
        features.append(
            _validate_feature_array(
                np.load(path, allow_pickle=False),
                path,
                feature_dim=feature_dim,
            )
        )
    if not features:
        raise ValueError(f"No .npy feature files found under: {root}")
    return _validate_feature_array(np.concatenate(features, 0), root, feature_dim=feature_dim)


def load_source_matrix(index_dir, *, feature_dim=None):
    source_path = Path(index_dir) / SOURCE_MATRIX_NAME
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Retrieval source matrix is required next to the index: {source_path}"
        )
    return _validate_feature_array(
        np.load(source_path, allow_pickle=False),
        source_path,
        feature_dim=feature_dim,
    )


def save_source_matrix(index_dir, matrix):
    index_path = Path(index_dir)
    index_path.mkdir(parents=True, exist_ok=True)
    np.save(index_path / SOURCE_MATRIX_NAME, matrix)
