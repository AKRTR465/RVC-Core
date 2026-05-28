import logging
import os

import numpy as np
import torch
from scipy.io.wavfile import read

from src.train.checkpoints import (
    latest_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    set_checkpoint_logger,
)
from src.train.hparams import HParams, get_hparams
from src.train.visualization import (
    plot_alignment_to_numpy,
    plot_spectrogram_to_numpy,
    plot_validation_mels_to_numpy,
)

logger = logging.getLogger(__name__)
set_checkpoint_logger(logger)


def summarize(
    writer,
    global_step,
    scalars=None,
    histograms=None,
    images=None,
    audios=None,
    audio_sampling_rate=22050,
):
    scalars = scalars or {}
    histograms = histograms or {}
    images = images or {}
    audios = audios or {}
    for key, value in scalars.items():
        writer.add_scalar(key, value, global_step)
    for key, value in histograms.items():
        writer.add_histogram(key, value, global_step)
    for key, value in images.items():
        writer.add_image(key, value, global_step, dataformats="HWC")
    for key, value in audios.items():
        writer.add_audio(key, value, global_step, audio_sampling_rate)


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8-sig") as handle:
        return [line.strip().split(split) for line in handle if line.strip()]


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    handler = logging.FileHandler(os.path.join(model_dir, filename))
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    target = os.path.abspath(os.path.join(model_dir, filename))
    if not any(
        isinstance(existing, logging.FileHandler)
        and os.path.abspath(existing.baseFilename) == target
        for existing in logger.handlers
    ):
        logger.addHandler(handler)
    else:
        handler.close()
    set_checkpoint_logger(logger)
    return logger
