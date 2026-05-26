from pathlib import Path

import torch
import torch.nn.functional as F


def read_wave_16k(wav_path, soundfile_module, normalize=False):
    wav, sr = soundfile_module.read(wav_path)
    if sr != 16000:
        raise ValueError(f"{wav_path} sampling rate must be 16000, got {sr}")
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:
        feats = feats.mean(-1)
    if feats.dim() != 1:
        raise ValueError(f"{wav_path} must be mono after downmix, got dim={feats.dim()}")
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats.view(1, -1)


def load_hubert_model(model_path, device, is_half, log_fn=None):
    import fairseq

    if not Path(model_path).is_file():
        if log_fn is not None:
            log_fn(
                "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
                % model_path
            )
        return None, None

    if log_fn is not None:
        log_fn(f"load model(s) from {model_path}")

    original_torch_load = torch.load

    def torch_load_trusted_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        try:
            return original_torch_load(*args, **kwargs)
        except TypeError as exc:
            if "weights_only" not in str(exc):
                raise
            kwargs.pop("weights_only", None)
            return original_torch_load(*args, **kwargs)

    torch.load = torch_load_trusted_checkpoint
    try:
        models, saved_cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [str(model_path)],
            suffix="",
        )
    finally:
        torch.load = original_torch_load

    model = models[0].to(device)
    if log_fn is not None:
        log_fn(f"move model to {device}")
    if is_half and torch.device(device).type != "cpu":
        model = model.half()
    else:
        model = model.float()
    model.task_normalize = bool(getattr(saved_cfg.task, "normalize", False))
    return model.eval(), saved_cfg
