import os
import platform
import re

import ffmpeg
import numpy as np


def load_audio(file, sr):
    try:
        # This launches a subprocess to decode audio while down-mixing and resampling.
        file = clean_path(file)
        if not os.path.exists(file):
            raise RuntimeError(
                "You input a wrong audio path that does not exists, please fix it!"
            )
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except (ffmpeg.Error, OSError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to load audio {file!r}: {exc}") from exc

    return np.frombuffer(out, np.float32).flatten()


def clean_path(path_str):
    path_str = os.fspath(path_str)
    if platform.system() == "Windows":
        path_str = path_str.replace("/", "\\")
    path_str = re.sub(r"[\u202a\u202b\u202c\u202d\u202e]", "", path_str)
    return path_str.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
