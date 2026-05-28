from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from src.features.mel import build_mel_basis
from torch import amp
from torch.nn import functional as F


@dataclass(frozen=True)
class RuntimeBackend:
    name: str
    spectrogram_torch: Callable
    spec_to_mel_torch: Callable
    mel_spectrogram_torch: Callable
    deterministic_sine: bool = False
    deterministic_discriminator_pad: bool = False


_MEL_BASIS_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_HANN_WINDOW_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_REFLECT_INDEX_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}
_PREFIX_SUM_CACHE: dict[tuple[Any, ...], torch.Tensor] = {}


def reset_deterministic_caches() -> None:
    _MEL_BASIS_CACHE.clear()
    _HANN_WINDOW_CACHE.clear()
    _REFLECT_INDEX_CACHE.clear()
    _PREFIX_SUM_CACHE.clear()


def resolve_runtime_backend(mode: str, native_module) -> RuntimeBackend:
    resolved_mode = str(mode).strip().lower()
    if resolved_mode == "native":
        return RuntimeBackend(
            name="native",
            spectrogram_torch=native_module.spectrogram_torch,
            spec_to_mel_torch=native_module.spec_to_mel_torch,
            mel_spectrogram_torch=native_module.mel_spectrogram_torch,
        )
    if resolved_mode == "deterministic_gpu":
        return RuntimeBackend(
            name="deterministic_gpu",
            spectrogram_torch=spectrogram_torch,
            spec_to_mel_torch=spec_to_mel_torch,
            mel_spectrogram_torch=mel_spectrogram_torch,
            deterministic_sine=True,
            deterministic_discriminator_pad=True,
        )
    raise ValueError(f"Unsupported numeric backend: {mode!r}")


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
        basis = build_mel_basis(sampling_rate, n_fft, num_mels, fmin, fmax)
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
