import os
import platform
import re
import subprocess

import numpy as np
import resampy


def load_audio(file, sr):
    file = clean_path(file)
    if not os.path.exists(file):
        raise RuntimeError(
            "You input a wrong audio path that does not exists, please fix it!"
        )
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        completed = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg executable was not found in PATH") from exc
    except (subprocess.CalledProcessError, OSError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to load audio {file!r}: {exc}") from exc

    return np.frombuffer(completed.stdout, np.float32).flatten()


def resample_audio(audio, orig_sr, target_sr):
    audio = np.asarray(audio)
    if int(orig_sr) == int(target_sr):
        return audio
    if not np.issubdtype(audio.dtype, np.floating):
        raise ValueError("resample_audio expects floating-point audio")
    if not np.isfinite(audio).all():
        raise ValueError("resample_audio received non-finite audio")

    ratio = float(target_sr) / orig_sr
    target_len = int(np.ceil(audio.shape[-1] * ratio))
    resampled = resampy.resample(
        audio,
        orig_sr,
        target_sr,
        filter="kaiser_best",
        axis=-1,
    )
    resampled = fix_length(resampled, target_len, axis=-1)
    return resampled.astype(audio.dtype, copy=False)


def audio_rms(audio, frame_length, hop_length):
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError(f"audio_rms expects mono audio, got shape {audio.shape}")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("frame_length and hop_length must be positive")

    padded = np.pad(audio, frame_length // 2, mode="constant")
    frames = frame_audio(padded, frame_length=frame_length, hop_length=hop_length)
    power = np.mean(np.abs(frames) ** 2, axis=-2, keepdims=True)
    return np.sqrt(power)


def clean_path(path_str):
    path_str = os.fspath(path_str)
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    path_str = re.sub(r"[\u202a\u202b\u202c\u202d\u202e]", "", path_str)
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


def frame_audio(audio, frame_length, hop_length):
    n_frames = 1 + (audio.shape[-1] - frame_length) // hop_length
    if n_frames < 1:
        raise ValueError("Input is too short for the requested frame_length")
    shape = audio.shape[:-1] + (frame_length, n_frames)
    strides = audio.strides[:-1] + (audio.strides[-1], hop_length * audio.strides[-1])
    return np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)


def fix_length(data, size, axis=-1):
    current = data.shape[axis]
    if current > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]
    if current < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - current)
        return np.pad(data, lengths, mode="constant")
    return data
