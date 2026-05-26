import os
import pickle
from collections import OrderedDict
from pathlib import Path

import torch

CHECKPOINT_ERRORS = (
    OSError,
    RuntimeError,
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    pickle.UnpicklingError,
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
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = "%sepoch" % epoch
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version
        torch.save(opt, _export_path(name, hps=hps))
        return "Success."
    except CHECKPOINT_ERRORS as exc:
        raise RuntimeError(f"Failed to save exported checkpoint: {name}") from exc


def show_info(path):
    try:
        a = torch.load(path, map_location="cpu")
        return "模型信息:%s\n采样率:%s\n模型是否输入音高引导:%s\n版本:%s" % (
            a.get("info", "None"),
            a.get("sr", "None"),
            a.get("f0", "None"),
            a.get("version", "None"),
        )
    except CHECKPOINT_ERRORS as exc:
        raise RuntimeError(f"Failed to read checkpoint info: {path}") from exc


def extract_small_model(path, name, sr, if_f0, info, version):
    try:
        ckpt = torch.load(path, map_location="cpu")
        source_config = ckpt.get("config") if isinstance(ckpt, dict) else None
        if "model" in ckpt:
            weights = ckpt["model"]
        elif "weight" in ckpt:
            weights = ckpt["weight"]
        else:
            weights = ckpt
        opt = OrderedDict()
        opt["weight"] = {}
        for key in weights.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = weights[key].half()
        if source_config is None:
            raise ValueError("Checkpoint is missing exported model config.")
        opt["config"] = source_config
        if info == "":
            info = "Extracted model."
        opt["info"] = info
        opt["version"] = version
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        torch.save(opt, _export_path(name))
        return "Success."
    except CHECKPOINT_ERRORS as exc:
        raise RuntimeError(f"Failed to extract small model: {path}") from exc


def change_info(path, info, name):
    try:
        ckpt = torch.load(path, map_location="cpu")
        ckpt["info"] = info
        if name == "":
            name = os.path.basename(path)
        torch.save(ckpt, _export_path(name, suffix=""))
        return "Success."
    except CHECKPOINT_ERRORS as exc:
        raise RuntimeError(f"Failed to change checkpoint info: {path}") from exc


def _normalize_f0_flag(f0):
    if isinstance(f0, bool):
        return int(f0)
    if isinstance(f0, (int, float)):
        return 1 if int(f0) != 0 else 0
    if isinstance(f0, str):
        value = f0.strip().lower()
        if value in {"1", "true", "yes", "y"}:
            return 1
        if value in {"0", "false", "no", "n"}:
            return 0
    return 0


def merge(path1, path2, alpha1, sr, f0, info, name, version):
    try:

        def extract(ckpt):
            a = ckpt["model"] if "model" in ckpt else ckpt["weight"]
            weight = OrderedDict()
            for key in a:
                if "enc_q" in key:
                    continue
                weight[key] = a[key]
            return weight

        ckpt1 = torch.load(path1, map_location="cpu")
        ckpt2 = torch.load(path2, map_location="cpu")
        cfg = ckpt1.get("config") or ckpt2.get("config")
        if cfg is None:
            raise ValueError("Cannot merge models without exported model config.")
        ckpt1 = extract(ckpt1)
        ckpt2 = extract(ckpt2)
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            raise ValueError("Cannot merge models with different architectures.")
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key][:min_shape0].float())
                    + (1 - alpha1) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()
        opt["config"] = cfg
        opt["sr"] = sr
        opt["f0"] = _normalize_f0_flag(f0)
        opt["version"] = version
        opt["info"] = info
        torch.save(opt, _export_path(name))
        return "Success."
    except CHECKPOINT_ERRORS as exc:
        raise RuntimeError(f"Failed to merge checkpoints: {path1}, {path2}") from exc

