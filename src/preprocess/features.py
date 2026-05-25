import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from configs.project_config import load_project_config, parse_hparams_overrides


def log_message(log_path, message):
    print(message)
    with open(log_path, "a+", encoding="utf-8") as handle:
        handle.write(f"{message}\n")
        handle.flush()


def read_wave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats.view(1, -1)


def resolve_device(device_request, i_gpu=""):
    if i_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(i_gpu)
    if device_request not in {"auto", "cpu", "cuda"}:
        raise ValueError("--device must be one of: auto, cpu, cuda")
    if device_request == "cpu":
        return "cpu"
    if device_request == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA for feature extraction, but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_hubert_model(model_path, device, is_half, log_path):
    import fairseq

    if not Path(model_path).is_file():
        log_message(
            log_path,
            "Error: Extracting is shut down because %s does not exist, you may download it from https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            % model_path,
        )
        return None, None

    log_message(log_path, f"load model(s) from {model_path}")
    models, saved_cfg, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [str(model_path)],
        suffix="",
    )
    model = models[0].to(device)
    log_message(log_path, f"move model to {device}")
    if is_half and device != "cpu":
        model = model.half()
    return model.eval(), saved_cfg


def output_dir_for_version(exp_dir, version):
    name = "3_feature256" if version == "v1" else "3_feature768"
    return Path(exp_dir) / name


def extract_features(
    exp_dir,
    version,
    n_part,
    i_part,
    device,
    is_half,
    model_path,
    log_path=None,
):
    exp_path = Path(exp_dir)
    wav_dir = exp_path / "1_16k_wavs"
    out_dir = output_dir_for_version(exp_path, version)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_path or exp_path / "extract_f0_feature.log"

    log_message(log_path, f"exp_dir: {exp_path}")
    model, saved_cfg = load_hubert_model(model_path, device, is_half, log_path)
    if model is None:
        return

    todo = sorted(os.listdir(wav_dir))[i_part::n_part]
    every = max(1, len(todo) // 10)
    if not todo:
        log_message(log_path, "no-feature-todo")
        return

    log_message(log_path, f"all-feature-{len(todo)}")
    output_layer = 9 if version == "v1" else 12
    for idx, file in enumerate(todo):
        try:
            if not file.endswith(".wav"):
                continue
            wav_path = wav_dir / file
            out_path = out_dir / file.replace("wav", "npy")
            if out_path.exists():
                continue

            feats = read_wave(wav_path, normalize=saved_cfg.task.normalize)
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
            source = feats.half() if is_half and device != "cpu" else feats
            inputs = {
                "source": source.to(device),
                "padding_mask": padding_mask.to(device),
                "output_layer": output_layer,
            }
            with torch.no_grad():
                logits = model.extract_features(**inputs)
                feats = model.final_proj(logits[0]) if version == "v1" else logits[0]

            feature_npy = feats.squeeze(0).float().cpu().numpy()
            if np.isnan(feature_npy).sum() == 0:
                np.save(out_path, feature_npy, allow_pickle=False)
            else:
                log_message(log_path, f"{file}-contains nan")
            if idx % every == 0:
                log_message(
                    log_path,
                    f"now-{len(todo)},all-{idx},{file},{feature_npy.shape}",
                )
        except Exception:
            log_message(log_path, traceback.format_exc())
    log_message(log_path, "all-feature-done")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy_args", nargs="*")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--hparams", type=str, default="")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--exp-dir", type=str, default="")
    parser.add_argument("--version", choices=["v1", "v2"], default="")
    parser.add_argument("--n-part", type=int, default=1)
    parser.add_argument("--i-part", type=int, default=0)
    parser.add_argument("--i-gpu", type=str, default="")
    parser.add_argument("--is-half", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--pretrain-root", type=str, default=os.getenv("pretrain_root", "pretrain"))
    parser.add_argument("--model-path", type=str, default="")
    args = parser.parse_args()

    if args.legacy_args:
        if len(args.legacy_args) not in {6, 7}:
            parser.error(
                "legacy mode requires: device n_part i_part [i_gpu] exp_dir version is_half"
            )
        if any([args.config, args.exp_dir, args.version, args.i_gpu, args.is_half]):
            parser.error("legacy mode cannot be mixed with config or named feature options")
        args.n_part = int(args.legacy_args[1])
        args.i_part = int(args.legacy_args[2])
        if len(args.legacy_args) == 6:
            args.exp_dir = args.legacy_args[3]
            args.version = args.legacy_args[4]
            args.is_half = args.legacy_args[5].lower() == "true"
        else:
            args.i_gpu = args.legacy_args[3]
            args.exp_dir = args.legacy_args[4]
            args.version = args.legacy_args[5]
            args.is_half = args.legacy_args[6].lower() == "true"
        args.device = "auto"

    if args.config:
        if args.exp_dir or args.version or args.pretrain_root != os.getenv("pretrain_root", "pretrain"):
            parser.error("config mode only accepts --config, --hparams, --reset, partition, device, and model options")
        project = load_project_config(
            args.config,
            overrides=parse_hparams_overrides(args.hparams),
            reset=args.reset,
        )
        args.exp_dir = project["preprocess_dir"]
        args.version = project["version"]
        args.pretrain_root = project["pretrain_root"]
        args.device = "cuda" if str(project["device"]).startswith("cuda") else "cpu"
        args.is_half = bool(project["is_half"])

    if args.exp_dir == "" or args.version == "":
        parser.error("provide --config or both --exp-dir and --version")
    if args.n_part < 1:
        parser.error("--n-part must be >= 1")
    if args.i_part < 0 or args.i_part >= args.n_part:
        parser.error("--i-part must satisfy 0 <= i_part < n_part")
    if args.model_path == "":
        args.model_path = str(Path(args.pretrain_root) / "hubert" / "hubert_base.pt")
    return args


def main():
    args = parse_args()
    device = resolve_device(args.device, args.i_gpu)
    log_path = Path(args.exp_dir) / "extract_f0_feature.log"
    log_message(log_path, " ".join(sys.argv))
    extract_features(
        exp_dir=args.exp_dir,
        version=args.version,
        n_part=args.n_part,
        i_part=args.i_part,
        device=device,
        is_half=bool(args.is_half),
        model_path=args.model_path,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
