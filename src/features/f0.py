import os

import numpy as np

F0_MIN = 50.0
F0_MAX = 1100.0
F0_BIN = 256
F0_METHODS = ("rmvpe",)


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
    if method not in F0_METHODS:
        raise ValueError(f"Unsupported f0 method: {method}")
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
