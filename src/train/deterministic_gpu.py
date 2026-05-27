from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import amp
from torch.nn import functional as F


_MEL_BASIS_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_HANN_WINDOW_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_REFLECT_INDEX_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_PREFIX_SUM_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def reset_deterministic_caches() -> None:
    _MEL_BASIS_CACHE.clear()
    _HANN_WINDOW_CACHE.clear()
    _REFLECT_INDEX_CACHE.clear()
    _PREFIX_SUM_CACHE.clear()


def _patch_attr(target: Any, name: str, value: Any) -> None:
    original_name = f"_rvc_rebuild_original_{name}"
    if not hasattr(target, original_name):
        setattr(target, original_name, getattr(target, name))
    setattr(target, name, value)


def _restore_attr(target: Any, name: str) -> None:
    original_name = f"_rvc_rebuild_original_{name}"
    if hasattr(target, original_name):
        setattr(target, name, getattr(target, original_name))


def reset_backend_runtime_overrides() -> None:
    from src.models import models as models_module
    from src.train import data_utils, mel_processing

    for name in ("spectrogram_torch", "spec_to_mel_torch", "mel_spectrogram_torch"):
        _restore_attr(mel_processing, name)
    _restore_attr(data_utils, "spectrogram_torch")
    _restore_attr(models_module.SineGen, "_f02sine")
    _restore_attr(models_module.DiscriminatorP, "forward")


def _hz_to_mel(frequencies, htk: bool = False):
    frequencies = np.asanyarray(frequencies, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)

    f_sp = 200.0 / 3.0
    mels = frequencies.copy()
    mels /= f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels, htk: bool = False):
    mels = np.asanyarray(mels, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    f_sp = 200.0 / 3.0
    freqs = mels.copy()
    freqs *= f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def _mel_frequencies(n_mels: int, fmin: float, fmax: float, htk: bool = False):
    min_mel = _hz_to_mel(fmin, htk=htk)
    max_mel = _hz_to_mel(fmax, htk=htk)
    return _mel_to_hz(np.linspace(min_mel, max_mel, n_mels), htk=htk)


def _build_mel_basis(
    sampling_rate: int,
    n_fft: int,
    num_mels: int,
    fmin: float,
    fmax: float | None,
) -> np.ndarray:
    if fmax is None:
        fmax = float(sampling_rate) / 2.0
    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sampling_rate)
    mel_f = _mel_frequencies(num_mels + 2, fmin, fmax, htk=False)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    lower = -ramps[:-2] / fdiff[:-1, np.newaxis]
    upper = ramps[2:] / fdiff[1:, np.newaxis]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2 : num_mels + 2] - mel_f[:num_mels])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32, copy=False)


def _autocast_device_type(device: torch.device) -> str:
    return "cuda" if device.type == "cuda" else "cpu"


def _reflect_indices(length: int, left: int, right: int, device: torch.device) -> torch.Tensor:
    key = (length, left, right, device.type, device.index)
    if key not in _REFLECT_INDEX_CACHE:
        if length <= 1:
            raise ValueError(f"reflect padding requires length > 1, got length={length}")
        if left >= length or right >= length:
            raise ValueError(
                f"reflect padding requires pad < length, got left={left} right={right} length={length}"
            )
        base = torch.arange(-left, length + right, device=device, dtype=torch.long)
        period = 2 * (length - 1)
        reflected = torch.remainder(base, period)
        reflected = (length - 1) - torch.abs(reflected - (length - 1))
        _REFLECT_INDEX_CACHE[key] = reflected.to(dtype=torch.long)
    return _REFLECT_INDEX_CACHE[key]


def reflect_pad_last(x: torch.Tensor, left: int, right: int) -> torch.Tensor:
    if left == 0 and right == 0:
        return x
    indices = _reflect_indices(x.size(-1), left, right, x.device)
    return x.index_select(-1, indices)


def _hann_window(n_fft: int, win_size: int, device: torch.device) -> torch.Tensor:
    key = (n_fft, win_size, device.type, device.index)
    if key not in _HANN_WINDOW_CACHE:
        window = torch.hann_window(win_size, dtype=torch.float32, device=device)
        if win_size != n_fft:
            padded = torch.zeros(n_fft, dtype=torch.float32, device=device)
            start = (n_fft - win_size) // 2
            padded[start : start + win_size] = window
            window = padded
        _HANN_WINDOW_CACHE[key] = window
    return _HANN_WINDOW_CACHE[key]


def _mel_basis(
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float | None,
    device: torch.device,
) -> torch.Tensor:
    key = (n_fft, num_mels, sampling_rate, fmin, fmax, device.type, device.index)
    if key not in _MEL_BASIS_CACHE:
        basis = _build_mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax)
        _MEL_BASIS_CACHE[key] = torch.from_numpy(basis).to(device=device, dtype=torch.float32)
    return _MEL_BASIS_CACHE[key]


def _prefix_sum_matrix(length: int, device: torch.device) -> torch.Tensor:
    key = (length, device.type, device.index)
    if key not in _PREFIX_SUM_CACHE:
        _PREFIX_SUM_CACHE[key] = torch.tril(
            torch.ones((length, length), device=device, dtype=torch.float32)
        )
    return _PREFIX_SUM_CACHE[key]


def spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> torch.Tensor:
    del sampling_rate
    if center:
        raise ValueError("deterministic spectrogram_torch only supports center=False")

    device_type = _autocast_device_type(y.device)
    with amp.autocast(device_type, enabled=False):
        y = y.float()
        pad = int((n_fft - hop_size) / 2)
        if y.size(-1) <= pad:
            raise ValueError(
                f"Audio is too short for reflect padding: length={y.size(-1)}, pad={pad}"
            )
        y = reflect_pad_last(y, pad, pad)
        frames = y.unfold(-1, n_fft, hop_size)
        window = _hann_window(n_fft, win_size, y.device).view(1, 1, -1)
        frames = frames * window
        spec = torch.fft.rfft(frames, n=n_fft, dim=-1)
        magnitude = torch.sqrt(spec.real.pow(2) + spec.imag.pow(2) + 1.0e-6)
        return magnitude.transpose(1, 2).contiguous()


def spec_to_mel_torch(
    spec: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    fmin: float,
    fmax: float | None,
) -> torch.Tensor:
    device_type = _autocast_device_type(spec.device)
    with amp.autocast(device_type, enabled=False):
        spec = spec.float()
        mel = torch.matmul(
            _mel_basis(n_fft, num_mels, sampling_rate, fmin, fmax, spec.device),
            spec,
        )
        return torch.log(torch.clamp(mel, min=1.0e-5))


def mel_spectrogram_torch(
    y: torch.Tensor,
    n_fft: int,
    num_mels: int,
    sampling_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float | None,
    center: bool = False,
) -> torch.Tensor:
    spec = spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=center)
    return spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)


def deterministic_f02sine(self, f0: torch.Tensor, upp: int) -> torch.Tensor:
    a = torch.arange(1, upp + 1, dtype=f0.dtype, device=f0.device)
    rad = f0 / self.sampling_rate * a
    rad2 = torch.fmod(rad[:, :-1, -1:].float() + 0.5, 1.0) - 0.5
    if rad2.size(1) > 0:
        prefix = _prefix_sum_matrix(rad2.size(1), rad2.device).unsqueeze(0)
        rad_acc = torch.matmul(prefix, rad2).fmod(1.0).to(f0)
        rad = rad + F.pad(rad_acc, (0, 0, 1, 0), mode="constant")
    rad = rad.reshape(f0.shape[0], -1, 1)
    b = torch.arange(1, self.dim + 1, dtype=f0.dtype, device=f0.device).reshape(1, 1, -1)
    rad = rad * b
    rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
    rand_ini[..., 0] = 0
    rad = rad + rand_ini
    return torch.sin(2 * np.pi * rad)


def build_discriminator_p_forward(models_module):
    lrelu_slope = float(models_module.modules.LRELU_SLOPE)

    def deterministic_forward(self, x: torch.Tensor):
        fmap = []

        batch_size, channels, total_length = x.shape
        if total_length % self.period != 0:
            n_pad = self.period - (total_length % self.period)
            x = reflect_pad_last(x, 0, n_pad)
            total_length = total_length + n_pad
        x = x.view(batch_size, channels, total_length // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap

    return deterministic_forward


def apply_deterministic_gpu_patches(hps) -> list[str]:
    mode = str(
        getattr(hps.train, "numeric_backend", getattr(hps.train, "mel_loss_device", "native"))
    ).strip().lower()
    if mode != "deterministic_gpu":
        return []

    from src.models import models as models_module
    from src.train import data_utils, mel_processing

    _patch_attr(mel_processing, "spectrogram_torch", spectrogram_torch)
    _patch_attr(mel_processing, "spec_to_mel_torch", spec_to_mel_torch)
    _patch_attr(mel_processing, "mel_spectrogram_torch", mel_spectrogram_torch)
    _patch_attr(data_utils, "spectrogram_torch", spectrogram_torch)
    _patch_attr(models_module.SineGen, "_f02sine", deterministic_f02sine)
    _patch_attr(
        models_module.DiscriminatorP,
        "forward",
        build_discriminator_p_forward(models_module),
    )
    return [
        "deterministic_gpu_spec",
        "deterministic_gpu_mel",
        "deterministic_gpu_sine",
        "deterministic_gpu_discriminator_pad",
    ]
