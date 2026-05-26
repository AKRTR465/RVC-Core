from pathlib import Path

import numpy as np


def load_retrieval_index(index_file):
    import faiss

    index_path = Path(index_file)
    index = faiss.read_index(str(index_path))
    if index.ntotal <= 0:
        raise ValueError(f"Retrieval index is empty: {index_path}")

    source_path = index_path.resolve().parent / "big_src_feature.npy"
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Retrieval source matrix is required next to the index: {source_path}"
        )
    source = np.load(source_path, allow_pickle=False)

    source = np.ascontiguousarray(source, dtype=np.float32)
    if source.ndim != 2:
        raise ValueError(f"Source feature matrix must be 2-D: {source_path}")
    if not np.isfinite(source).all():
        raise ValueError(f"Source feature matrix contains non-finite values: {source_path}")
    if source.shape[0] < index.ntotal:
        raise ValueError(
            f"Source feature rows {source.shape[0]} fewer than index entries {index.ntotal}"
        )
    if hasattr(index, "d") and source.shape[1] != index.d:
        raise ValueError(
            f"Source feature dim {source.shape[1]} does not match index dim {index.d}"
        )
    return index, source


def blend_search_features(index, source, features, top_k=8):
    k = min(int(top_k), int(index.ntotal), int(source.shape[0]))
    if k <= 0:
        return features

    score, ix = index.search(features, k=k)
    valid = (ix >= 0) & np.isfinite(score)
    safe_ix = np.where(valid, ix, 0)
    safe_score = np.where(valid, score, np.inf)
    exact = valid & (safe_score <= 1e-8)
    weight = np.zeros_like(safe_score, dtype=np.float32)

    exact_rows = exact.any(axis=1)
    if exact_rows.any():
        exact_weight = exact[exact_rows].astype(np.float32)
        exact_weight /= exact_weight.sum(axis=1, keepdims=True)
        weight[exact_rows] = exact_weight

    non_exact_rows = ~exact_rows
    if non_exact_rows.any():
        inv = np.where(
            valid[non_exact_rows],
            1.0 / np.maximum(safe_score[non_exact_rows], 1e-6),
            0.0,
        )
        inv = np.square(inv)
        denom = inv.sum(axis=1, keepdims=True)
        good = denom.squeeze(1) > 0
        if good.any():
            weight[non_exact_rows] = np.where(denom > 0, inv / denom, 0.0)

    blended = np.sum(source[safe_ix] * np.expand_dims(weight, axis=2), axis=1)
    has_neighbor = weight.sum(axis=1) > 0
    return np.where(has_neighbor[:, None], blended, features)
