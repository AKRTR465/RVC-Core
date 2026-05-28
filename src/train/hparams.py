import argparse
import copy
from pathlib import Path

from configs.project_config import (
    load_project_config,
    parse_hparams_overrides,
    save_project_config_snapshot,
)

_INT_TRAIN_OVERRIDES = (
    ("save_every_epoch", "save_every_epoch", "save_every_epoch"),
    ("epochs", "total_epoch", "total_epoch"),
    ("if_latest", "if_latest", "if_latest"),
    ("if_cache_data_in_gpu", "if_cache_data_in_gpu", "if_cache_data_in_gpu"),
    ("batch_size", "batch_size", "batch_size"),
)

_TEXT_TRAIN_OVERRIDES = (
    ("pretrainG", "pretrainG"),
    ("pretrainD", "pretrainD"),
)

_REPLAYABLE_TRAIN_KEYS = (
    "save_every_epoch",
    "epochs",
    "pretrainG",
    "pretrainD",
    "if_latest",
    "if_cache_data_in_gpu",
    "save_every_weights",
    "batch_size",
)


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


def _apply_training_cli_overrides(config, args):
    train = config["train"]
    replayable_train = config.get("replayable_config", {}).get("train")

    for train_key, cli_attr, required_name in _INT_TRAIN_OVERRIDES:
        current = train.get(train_key)
        if train_key == "epochs":
            current = train.get("epochs")
        train[train_key] = int(
            _required_value(
                required_name,
                _first_not_empty(getattr(args, cli_attr), current),
            )
        )

    for train_key, cli_attr in _TEXT_TRAIN_OVERRIDES:
        cli_value = getattr(args, cli_attr)
        train[train_key] = cli_value if cli_value != "" else train.get(train_key, "") or ""

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

    if isinstance(replayable_train, dict):
        for key in _REPLAYABLE_TRAIN_KEYS:
            replayable_train[key] = train[key]

    if args.gpus is not None:
        config["gpus"] = args.gpus
    return config


def _snapshot_path_for_project(config):
    return Path(config["paths"]["work_dir"]) / "config.yaml"


def _build_hparams(config):
    hparams = HParams(**config)
    selectors = config["selectors"]
    runtime = config["runtime"]
    runtime_slice = runtime["slice"]
    train = config["train"]
    paths = config["paths"]

    hparams.version = selectors["version"]
    hparams.sample_rate = selectors["sample_rate"]
    hparams.if_f0 = selectors["if_f0"]
    hparams.device = runtime["device"]
    hparams.is_half = runtime["is_half"]
    hparams.n_cpu = runtime["n_cpu"]
    hparams.x_pad = runtime_slice["x_pad"]
    hparams.x_query = runtime_slice["x_query"]
    hparams.x_center = runtime_slice["x_center"]
    hparams.x_max = runtime_slice["x_max"]
    hparams.save_every_epoch = int(train["save_every_epoch"])
    hparams.total_epoch = int(train["epochs"])
    hparams.pretrainG = train.get("pretrainG", "")
    hparams.pretrainD = train.get("pretrainD", "")
    hparams.if_latest = int(train["if_latest"])
    hparams.if_cache_data_in_gpu = int(train["if_cache_data_in_gpu"])
    hparams.save_every_weights = _normalize_save_every_weights(
        train.get("save_every_weights")
    )
    hparams.export_dir = paths["export_dir"]
    hparams.model_dir = paths["train_dir"]
    hparams.experiment_dir = paths["work_dir"]
    hparams.gpus = config.get("gpus", "0")
    hparams.data.training_files = paths["training_files"]
    hparams.data.validation_files = paths["validation_files"]
    return hparams


def _build_parser():
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
    parser.add_argument(
        "--config", type=str, default="", help="task config path or name"
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default="",
        help="comma-separated scalar overrides, e.g. train.batch_size=1",
    )
    parser.add_argument(
        "--reset", action="store_true", help="ignore work_dir/config.yaml"
    )
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
    return parser


def get_hparams(init=True):
    _ = init
    parser = _build_parser()
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
    config["data"]["training_files"] = project_config["paths"]["training_files"]
    config["data"]["validation_files"] = project_config["paths"]["validation_files"]
    config = _apply_training_cli_overrides(config, args)

    config_save_path = _snapshot_path_for_project(config)
    if args.reset or not config_save_path.exists():
        save_project_config_snapshot(config, config_save_path)

    return _build_hparams(config)


class HParams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                value = HParams(**value)
            self[key] = value

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
