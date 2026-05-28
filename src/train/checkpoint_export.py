import os
from collections import OrderedDict
from pathlib import Path

import torch
from src.models.models import build_export_model_config

CHECKPOINT_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
)


def _export_weight_payload(ckpt, sr, if_f0, epoch, version, hps):
    weights = OrderedDict(
        (key, value.half())
        for key, value in ckpt.items()
        if "enc_q" not in key
    )
    return OrderedDict(
        (
            ("weight", weights),
            ("config", build_export_model_config(hps)),
            ("info", f"{epoch}epoch"),
            ("sr", sr),
            ("f0", if_f0),
            ("version", version),
        )
    )


def _guess_project_name(name):
    stem = Path(name).stem
    if "_e" in stem and "_s" in stem:
        return stem.split("_e", 1)[0]
    return stem


def _export_path(name, suffix=".pth", hps=None):
    filename = Path(name).name
    filename = filename if filename.endswith(suffix) else f"{filename}{suffix}"
    if hps is not None and getattr(hps, "export_dir", None):
        export_dir = Path(hps.export_dir)
    else:
        export_dir = (
            Path(os.getenv("ckpt_root", "ckpt")) / _guess_project_name(name) / "export"
        )
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir / filename


def savee(ckpt, sr, if_f0, name, epoch, version, hps):
    try:
        torch.save(
            _export_weight_payload(ckpt, sr, if_f0, epoch, version, hps),
            _export_path(name, hps=hps),
        )
        return "Success."
    except CHECKPOINT_ERRORS as exc:
        raise RuntimeError(f"Failed to save exported checkpoint: {name}") from exc
