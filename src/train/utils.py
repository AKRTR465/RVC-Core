import argparse
import copy
import glob
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from scipy.io.wavfile import read

from configs.project_config import (
    load_project_config,
    parse_hparams_overrides,
    save_project_config_snapshot,
)

MATPLOTLIB_FLAG = False

logger = logging.getLogger(__name__)


def load_checkpoint_d(checkpoint_path, combd, sbd, optimizer=None, load_opt=1):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    ##################
    def go(model, bkey):
        saved_state_dict = checkpoint_dict[bkey]
        if hasattr(model, "module"):
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():  # 模型需要的shape
            try:
                new_state_dict[k] = saved_state_dict[k]
                if saved_state_dict[k].shape != state_dict[k].shape:
                    logger.warning(
                        "shape-%s-mismatch. need: %s, get: %s",
                        k,
                        state_dict[k].shape,
                        saved_state_dict[k].shape,
                    )  #
                    raise KeyError
            except KeyError:
                logger.info("%s is not in the checkpoint", k)  # pretrain缺失的
                new_state_dict[k] = v  # 模型自带的随机值
        if hasattr(model, "module"):
            model.module.load_state_dict(new_state_dict, strict=False)
        else:
            model.load_state_dict(new_state_dict, strict=False)
        return model

    go(combd, "combd")
    model = go(sbd, "sbd")
    #############
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  ###加载不了，如果是空的的话，重新初始化，可能还会影响lr时间表的更新，因此在train文件最外围catch
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration

def load_checkpoint(checkpoint_path, model, optimizer=None, load_opt=1, scaler=None):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():  # 模型需要的shape
        try:
            new_state_dict[k] = saved_state_dict[k]
            if saved_state_dict[k].shape != state_dict[k].shape:
                logger.warning(
                    "shape-%s-mismatch|need-%s|get-%s",
                    k,
                    state_dict[k].shape,
                    saved_state_dict[k].shape,
                )  #
                raise KeyError
        except KeyError:
            logger.info("%s is not in the checkpoint", k)  # pretrain缺失的
            new_state_dict[k] = v  # 模型自带的随机值
    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    logger.info("Loaded model weights")

    iteration = checkpoint_dict["iteration"]
    learning_rate = checkpoint_dict["learning_rate"]
    if (
        optimizer is not None and load_opt == 1
    ):  ###加载不了，如果是空的的话，重新初始化，可能还会影响lr时间表的更新，因此在train文件最外围catch
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


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path, scaler=None):
    logger.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        checkpoint_path,
    )


def save_checkpoint_d(combd, sbd, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info(
        "Saving model and optimizer state at epoch {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(combd, "module"):
        state_dict_combd = combd.module.state_dict()
    else:
        state_dict_combd = combd.state_dict()
    if hasattr(sbd, "module"):
        state_dict_sbd = sbd.module.state_dict()
    else:
        state_dict_sbd = sbd.state_dict()
    torch.save(
        {
            "combd": state_dict_combd,
            "sbd": state_dict_sbd,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


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
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    if not f_list:
        raise FileNotFoundError(f"No checkpoint matching {regex} under {dir_path}")
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, Path(f).stem)) or -1))
    return f_list[-1]


def _figure_canvas_to_rgb_array(fig, np):
    fig.canvas.draw()
    if hasattr(fig.canvas, "tostring_rgb"):
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()


def plot_spectrogram_to_numpy(spectrogram):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    data = _figure_canvas_to_rgb_array(fig, np)
    plt.close()
    return data


def plot_validation_mels_to_numpy(gt_mel, pred_mel, diff_mel):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    gt_mel = np.asarray(gt_mel)
    pred_mel = np.asarray(pred_mel)
    diff_mel = np.asarray(diff_mel)

    mel_min = float(min(np.min(gt_mel), np.min(pred_mel)))
    mel_max = float(max(np.max(gt_mel), np.max(pred_mel)))
    if mel_min == mel_max:
        mel_max = mel_min + 1e-6

    diff_abs_max = float(np.max(np.abs(diff_mel)))
    if diff_abs_max == 0.0:
        diff_abs_max = 1e-6

    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    panels = (
        ("GT", gt_mel, "viridis", mel_min, mel_max),
        ("PRED", pred_mel, "viridis", mel_min, mel_max),
        ("DIFF", diff_mel, "seismic", -diff_abs_max, diff_abs_max),
    )

    for ax, (title, mel, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(
            mel,
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        ax.set_ylabel("Channels")

    axes[-1].set_xlabel("Frames")
    plt.tight_layout()

    data = _figure_canvas_to_rgb_array(fig, np)
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    data = _figure_canvas_to_rgb_array(fig, np)
    plt.close()
    return data


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding="utf-8-sig") as f:
        filepaths_and_text = [line.strip().split(split) for line in f if line.strip()]
    
    return filepaths_and_text


def _first_not_empty(*values):
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _required_value(field_name, value):
    if value in (None, ""):
        raise ValueError(f"Missing required training setting: {field_name}")
    return value


def _normalize_save_every_weights(value):
    if value in (None, ""):
        return "0"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return "1" if int(value) != 0 else "0"
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return "1"
    return "0"


def _sync_train_aliases(config):
    train = config["train"]
    config["save_every_epoch"] = int(train["save_every_epoch"])
    config["total_epoch"] = int(train["epochs"])
    config["pretrainG"] = train.get("pretrainG", "")
    config["pretrainD"] = train.get("pretrainD", "")
    config["if_latest"] = int(train["if_latest"])
    config["if_cache_data_in_gpu"] = int(train["if_cache_data_in_gpu"])
    config["save_every_weights"] = _normalize_save_every_weights(
        train.get("save_every_weights")
    )


def _apply_training_cli_overrides(config, args):
    train = config["train"]
    replayable_train = config.get("replayable_config", {}).get("train")

    train["save_every_epoch"] = int(
        _required_value(
            "save_every_epoch",
            _first_not_empty(args.save_every_epoch, train.get("save_every_epoch")),
        )
    )
    train["epochs"] = int(
        _required_value(
            "total_epoch",
            _first_not_empty(args.total_epoch, train.get("epochs")),
        )
    )
    train["pretrainG"] = (
        args.pretrainG if args.pretrainG != "" else train.get("pretrainG", "") or ""
    )
    train["pretrainD"] = (
        args.pretrainD if args.pretrainD != "" else train.get("pretrainD", "") or ""
    )
    train["if_latest"] = int(
        _required_value(
            "if_latest",
            _first_not_empty(args.if_latest, train.get("if_latest")),
        )
    )
    train["if_cache_data_in_gpu"] = int(
        _required_value(
            "if_cache_data_in_gpu",
            _first_not_empty(args.if_cache_data_in_gpu, train.get("if_cache_data_in_gpu")),
        )
    )
    train["save_every_weights"] = (
        _normalize_save_every_weights(
            _first_not_empty(
                args.save_every_weights,
                train.get("save_every_weights"),
                "0",
            )
        )
        == "1"
    )
    train["batch_size"] = int(
        _required_value(
            "batch_size",
            _first_not_empty(args.batch_size, train.get("batch_size")),
        )
    )

    if isinstance(replayable_train, dict):
        replayable_train["save_every_epoch"] = train["save_every_epoch"]
        replayable_train["epochs"] = train["epochs"]
        replayable_train["pretrainG"] = train["pretrainG"]
        replayable_train["pretrainD"] = train["pretrainD"]
        replayable_train["if_latest"] = train["if_latest"]
        replayable_train["if_cache_data_in_gpu"] = train["if_cache_data_in_gpu"]
        replayable_train["save_every_weights"] = train["save_every_weights"]
        replayable_train["batch_size"] = train["batch_size"]

    if args.gpus is not None:
        config["gpus"] = args.gpus
    _sync_train_aliases(config)
    return config


def _snapshot_path_for_project(config):
    return Path(config["work_dir"]) / "config.yaml"


def _build_hparams(config):
    hparams = HParams(**config)
    hparams.model_dir = config["train_dir"]
    hparams.experiment_dir = config["work_dir"]
    hparams.gpus = config.get("gpus", "0")
    hparams.data.training_files = config["training_files"]
    hparams.data.validation_files = config["validation_files"]
    return hparams


def get_hparams(init=True):
    """
    todo:
      结尾七人组：
        保存频率、总epoch                     done
        bs                                    done
        pretrainG、pretrainD                  done
        卡号：os.en["CUDA_VISIBLE_DEVICES"]   done
        if_latest                             done
      模型：if_f0                             done
      采样率：自动选择config                  done
      是否缓存数据集进GPU:if_cache_data_in_gpu done

      -m:
        自动决定training_files路径,改掉train_nsf_load_pretrain.py里的hps.data.training_files    done
      -c不要了
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-se",
        "--save_every_epoch",
        type=int,
        default=None,
        help="checkpoint save frequency (epoch)",
    )
    parser.add_argument(
        "-te", "--total_epoch", type=int, default=None, help="total_epoch"
    )
    parser.add_argument(
        "-pg", "--pretrainG", type=str, default="", help="Pretrained Generator path"
    )
    parser.add_argument(
        "-pd", "--pretrainD", type=str, default="", help="Pretrained Discriminator path"
    )
    parser.add_argument("-g", "--gpus", type=str, default=None, help="split by -")
    parser.add_argument(
        "-bs", "--batch_size", type=int, default=None, help="batch size"
    )
    parser.add_argument("--config", type=str, default="", help="task config path or name")
    parser.add_argument(
        "--hparams",
        type=str,
        default="",
        help="comma-separated scalar overrides, e.g. train.batch_size=1",
    )
    parser.add_argument("--reset", action="store_true", help="ignore work_dir/config.yaml")
    parser.add_argument(
        "-sw",
        "--save_every_weights",
        type=str,
        default="0",
        help="save the extracted model in weights directory when saving checkpoints",
    )
    parser.add_argument(
        "-l",
        "--if_latest",
        type=int,
        default=None,
        help="if only save the latest G/D pth file, 1 or 0",
    )
    parser.add_argument(
        "-c",
        "--if_cache_data_in_gpu",
        type=int,
        default=None,
        help="if caching the dataset in GPU memory, 1 or 0",
    )

    args = parser.parse_args()
    if args.config == "":
        raise ValueError("Please provide --config.")

    project_config = load_project_config(
        args.config,
        overrides=parse_hparams_overrides(args.hparams),
        reset=args.reset,
    )
    config = copy.deepcopy(project_config)

    config.setdefault("data", {})
    config["data"]["training_files"] = project_config["training_files"]
    config["data"]["validation_files"] = project_config["validation_files"]
    config = _apply_training_cli_overrides(config, args)

    config_save_path = _snapshot_path_for_project(config)
    if args.reset or not config_save_path.exists():
        save_project_config_snapshot(config, config_save_path)

    return _build_hparams(config)


def get_hparams_from_dir(model_dir):
    model_dir_path = Path(model_dir)
    config_save_path = model_dir_path.parent / "config.yaml"
    config = load_project_config(config_save_path, reset=True)
    _sync_train_aliases(config)
    return _build_hparams(config)


def get_hparams_from_file(config_path):
    path = Path(config_path)
    config = load_project_config(path, reset=path.name == "config.yaml")
    _sync_train_aliases(config)
    return _build_hparams(config)


def check_git_hash(model_dir):
    source_dir = Path(__file__).resolve().parents[2]
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path, encoding="utf-8").read()
        if saved_hash != cur_hash:
            logger.warning(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w", encoding="utf-8").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    target = os.path.abspath(os.path.join(model_dir, filename))
    if not any(
        isinstance(handler, logging.FileHandler)
        and os.path.abspath(handler.baseFilename) == target
        for handler in logger.handlers
    ):
        logger.addHandler(h)
    else:
        h.close()
    return logger


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

