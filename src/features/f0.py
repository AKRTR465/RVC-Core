import os

import numpy as np

F0_MIN = 50.0
F0_MAX = 1100.0
F0_BIN = 256


def compute_pm_f0(x, sr, p_len, f0_min=F0_MIN, f0_max=F0_MAX, hop_size=160):
    import parselmouth

    time_step = hop_size / sr * 1000
    f0 = (
        parselmouth.Sound(x, sr)
        .to_pitch_ac(
            time_step=time_step / 1000,
            voicing_threshold=0.6,
            pitch_floor=f0_min,
            pitch_ceiling=f0_max,
        )
        .selected_array["frequency"]
    )
    pad_size = (p_len - len(f0) + 1) // 2
    if pad_size > 0 or p_len - len(f0) - pad_size > 0:
        f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
    return f0


def compute_world_f0(x, sr, hop_size, method, f0_min=F0_MIN, f0_max=F0_MAX):
    import pyworld

    audio = x.astype(np.double)
    frame_period = 1000 * hop_size / sr
    if method == "harvest":
        f0, t = pyworld.harvest(
            audio,
            fs=sr,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=frame_period,
        )
    elif method == "dio":
        f0, t = pyworld.dio(
            audio,
            fs=sr,
            f0_ceil=f0_max,
            f0_floor=f0_min,
            frame_period=frame_period,
        )
    else:
        raise ValueError(f"Unsupported pyworld f0 method: {method}")
    return pyworld.stonemask(audio, f0, t, sr)


def compute_crepe_f0(x, sr, hop_size, device, f0_min=F0_MIN, f0_max=F0_MAX):
    import torch
    import torchcrepe

    audio = torch.tensor(np.copy(x))[None].float()
    f0, pd = torchcrepe.predict(
        audio,
        sr,
        hop_size,
        f0_min,
        f0_max,
        "full",
        batch_size=512,
        device=device,
        return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)
    f0 = torchcrepe.filter.mean(f0, 3)
    f0[pd < 0.1] = 0
    return f0[0].cpu().numpy()


def load_rmvpe_model(model_path, device, is_half, log_fn=None):
    from src.utils.rmvpe import RMVPE

    if log_fn is not None:
        log_fn(f"Loading rmvpe model,{model_path}")
    return RMVPE(model_path, is_half=is_half, device=device)


def compute_f0_by_method(
    x,
    sr,
    p_len,
    hop_size,
    method,
    *,
    device="cpu",
    is_half=False,
    pretrain_root="pretrain",
    rmvpe_path=None,
    rmvpe_model=None,
    log_fn=None,
    f0_min=F0_MIN,
    f0_max=F0_MAX,
):
    if method == "pm":
        return (
            compute_pm_f0(x, sr, p_len, f0_min=f0_min, f0_max=f0_max, hop_size=hop_size),
            rmvpe_model,
        )
    if method in {"harvest", "dio"}:
        return (
            compute_world_f0(x, sr, hop_size, method, f0_min=f0_min, f0_max=f0_max),
            rmvpe_model,
        )
    if method == "crepe":
        return (
            compute_crepe_f0(x, sr, hop_size, device, f0_min=f0_min, f0_max=f0_max),
            rmvpe_model,
        )
    if method == "rmvpe":
        model = rmvpe_model
        if model is None:
            model = load_rmvpe_model(
                rmvpe_path or os.path.join(pretrain_root, "rmvpe", "rmvpe.pt"),
                device,
                is_half,
                log_fn,
            )
        return model.infer_from_audio(x, thred=0.03), model
    raise ValueError(f"Unsupported f0 method: {method}")


def f0_to_coarse(f0, f0_min=F0_MIN, f0_max=F0_MAX, f0_bin=F0_BIN):
    if f0.size == 0:
        raise ValueError("empty f0 sequence")
    if not np.isfinite(f0).all():
        raise ValueError("f0 sequence contains NaN or inf")

    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    voiced = f0_mel > 0
    f0_mel[voiced] = (f0_mel[voiced] - f0_mel_min) * (f0_bin - 2) / (
        f0_mel_max - f0_mel_min
    ) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > f0_bin - 1] = f0_bin - 1
    f0_coarse = np.rint(f0_mel).astype(np.int32)
    if f0_coarse.max() > 255 or f0_coarse.min() < 1:
        raise ValueError(
            f"coarse f0 out of range: min={f0_coarse.min()}, max={f0_coarse.max()}"
        )
    return f0_coarse
