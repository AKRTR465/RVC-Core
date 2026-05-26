import os
from pathlib import Path

from src.features.hubert import load_hubert_model


def _iter_paths(root, suffix):
    root_path = Path(root)
    if not root_path.exists():
        return []
    return sorted(path for path in root_path.rglob(f"*{suffix}") if path.is_file())


def get_model_path_from_sid(sid, ckpt_root):
    sid_path = Path(sid)
    root_path = Path(ckpt_root)
    root_resolved = root_path.resolve()
    if sid_path.is_absolute():
        absolute_path = sid_path.resolve()
        try:
            absolute_path.relative_to(root_resolved)
        except ValueError:
            return ""
        if absolute_path.is_file() and absolute_path.suffix == ".pth":
            return str(absolute_path)
        return ""

    direct_path = (root_path / sid_path).resolve()
    try:
        direct_path.relative_to(root_resolved)
    except ValueError:
        return ""
    if direct_path.is_file() and direct_path.suffix == ".pth":
        return str(direct_path)

    matches = []
    for path in _iter_paths(root_path, ".pth"):
        if path.name.startswith(("G_", "D_")):
            continue
        if path.name == sid_path.name or path.stem == sid_path.stem:
            matches.append(path)

    export_matches = [path for path in matches if "export" in path.parts]
    picked = export_matches[0] if export_matches else (matches[0] if matches else None)
    return str(picked) if picked is not None else ""


def get_index_path_from_model(sid, ckpt_root):
    model_path = get_model_path_from_sid(sid, ckpt_root)
    model_stem = Path(sid).stem
    candidate_roots = []
    if model_path != "":
        model_dir = Path(model_path).resolve()
        if model_dir.parent.name == "export":
            candidate_roots.append(model_dir.parents[1] / "index")
    candidate_roots.append(Path(ckpt_root))

    for root in candidate_roots:
        for path in _iter_paths(root, ".index"):
            if "trained" in path.name:
                continue
            if path.stem == model_stem or model_stem in path.stem:
                return str(path)

    return ""


def load_hubert(config):
    model_path = getattr(
        config,
        "hubert_path",
        os.path.join(getattr(config, "pretrain_root", "pretrain"), "hubert", "hubert_base.pt"),
    )
    hubert_model, _ = load_hubert_model(model_path, config.device, config.is_half)
    if hubert_model is None:
        raise FileNotFoundError(model_path)
    return hubert_model

