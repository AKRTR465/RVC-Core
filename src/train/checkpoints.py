import glob
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def set_checkpoint_logger(checkpoint_logger):
    global logger
    logger = checkpoint_logger


def _model_state_dict(model):
    if hasattr(model, "module"):
        return model.module.state_dict()
    return model.state_dict()


def _load_matching_state_dict(model, saved_state_dict, mismatch_message):
    state_dict = _model_state_dict(model)
    new_state_dict = {}
    for key, value in state_dict.items():
        try:
            loaded_value = saved_state_dict[key]
            new_state_dict[key] = loaded_value
            if loaded_value.shape != value.shape:
                logger.warning(mismatch_message, key, value.shape, loaded_value.shape)
                raise KeyError
        except KeyError:
            logger.info("%s is not in the checkpoint", key)
            new_state_dict[key] = value
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    return model


def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1, scaler=None):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    _load_matching_state_dict(
        model,
        checkpoint_dict["model"],
        "shape-%s-mismatch|need-%s|get-%s",
    )
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if optimizer is not None and load_opt == 1:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    if scaler is not None and load_opt == 1:
        scaler_state = checkpoint_dict.get("scaler")
        if scaler_state is None:
            logger.warning(
                "Checkpoint '%s' is missing GradScaler state; resuming with a fresh scaler.",
                checkpoint_path,
            )
        else:
            scaler.load_state_dict(scaler_state)
    logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(
    model, optimizer, learning_rate, iteration, checkpoint_path, scaler=None
):
    logger.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
    )
    torch.save(
        {
            "model": _model_state_dict(model),
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        checkpoint_path,
    )


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    if not f_list:
        raise FileNotFoundError(f"No checkpoint matching {regex} under {dir_path}")
    f_list.sort(key=lambda path: int("".join(filter(str.isdigit, Path(path).stem)) or -1))
    return f_list[-1]
