import os
from io import BytesIO
from pathlib import Path

import soundfile as sf

from src.utils.audio import clean_path, wav2


def resolve_batch_input_paths(dir_path, paths):
    dir_path = clean_path(dir_path)
    if dir_path:
        return [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
    return [path.name for path in paths]


def output_path_for_input(opt_root, input_path, audio_format):
    opt_root = clean_path(opt_root)
    return os.path.join(opt_root, f"{Path(input_path).stem}.{audio_format}")


def save_converted_audio(opt_root, input_path, audio, sample_rate, audio_format):
    output_path = output_path_for_input(opt_root, input_path, audio_format)
    if audio_format in {"wav", "flac"}:
        sf.write(output_path, audio, sample_rate)
        return output_path

    with BytesIO() as wavf:
        sf.write(wavf, audio, sample_rate, format="wav")
        wavf.seek(0, 0)
        with open(output_path, "wb") as outf:
            wav2(wavf, outf, audio_format)
    return output_path
