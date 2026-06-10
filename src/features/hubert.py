from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.features.hubert_fairseq_compat import (
    build_hubert_model_from_checkpoint,
    checkpoint_cfg_to_namespace,
)


def prepare_hubert_waveform(
    waveform,
    normalize=False,
    *,
    source_label="audio",
    mono_error_template="{label} must be mono after downmix, got dim={dim}",
):
    feats = torch.as_tensor(waveform).float()
    if feats.dim() == 2:
        if feats.shape[0] == 1:
            feats = feats.squeeze(0)
        else:
            feats = feats.mean(-1)
    if feats.dim() != 1:
        raise ValueError(mono_error_template.format(label=source_label, dim=feats.dim()))
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats.view(1, -1)


def build_hubert_inputs(
    waveform,
    device,
    is_half,
    *,
    normalize=False,
    source_label="audio",
    mono_error_template="{label} must be mono after downmix, got dim={dim}",
):
    feats = prepare_hubert_waveform(
        waveform,
        normalize=normalize,
        source_label=source_label,
        mono_error_template=mono_error_template,
    )
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    source = feats.half() if is_half and torch.device(device).type != "cpu" else feats
    return {
        "source": source.to(device),
        "padding_mask": padding_mask.to(device),
    }


def extract_hubert_features(
    model,
    waveform,
    version,
    device,
    is_half,
    *,
    normalize=None,
    source_label="audio",
    mono_error_template="{label} must be mono after downmix, got dim={dim}",
):
    if normalize is None:
        normalize = bool(getattr(model, "task_normalize", False))
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported HuBERT feature version: {version}")

    inputs = build_hubert_inputs(
        waveform,
        device,
        is_half,
        normalize=normalize,
        source_label=source_label,
        mono_error_template=mono_error_template,
    )
    inputs["output_layer"] = 9 if version == "v1" else 12
    with torch.no_grad():
        logits = model.extract_features(**inputs)
        return model.final_proj(logits[0]) if version == "v1" else logits[0]


def read_wave_16k(wav_path, soundfile_module):
    wav, sr = soundfile_module.read(wav_path)
    if sr != 16000:
        raise ValueError(f"{wav_path} sampling rate must be 16000, got {sr}")
    return wav


def _torch_load_trusted_checkpoint(model_path):
    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        return torch.load(model_path, **kwargs)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        kwargs.pop("weights_only", None)
        return torch.load(model_path, **kwargs)


def load_hubert_model(model_path, device, is_half, log_fn=None):
    if not Path(model_path).is_file():
        if log_fn is not None:
            log_fn(
                "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
                % model_path
            )
        return None, None

    if log_fn is not None:
        log_fn(f"load model(s) from {model_path}")

    checkpoint = _torch_load_trusted_checkpoint(str(model_path))
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"HuBERT checkpoint does not contain a model state: {model_path}")

    cfg = checkpoint.get("cfg", {})
    saved_cfg = checkpoint_cfg_to_namespace(cfg)
    if not hasattr(saved_cfg, "task"):
        saved_cfg.task = SimpleNamespace(normalize=False)
    elif not hasattr(saved_cfg.task, "normalize"):
        saved_cfg.task.normalize = False

    model = build_hubert_model_from_checkpoint(cfg, checkpoint["model"]).to(device)
    if log_fn is not None:
        log_fn(f"move model to {device}")
    if is_half and torch.device(device).type != "cpu":
        model = model.half()
    else:
        model = model.float()
    return model.eval(), saved_cfg
