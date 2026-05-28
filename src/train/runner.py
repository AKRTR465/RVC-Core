import os
import logging
import sys
import datetime
import math
import random
from pathlib import Path

import numpy as np
from src.train import utils

from random import randint, shuffle

import torch
from torch import amp
from tqdm.auto import tqdm

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.models import commons
from src.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
    TrainingBatch,
)

from src.models.models import (
    MultiPeriodDiscriminator,
    MultiPeriodDiscriminatorV2,
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)

from src.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from src.train import mel_processing
from src.train.checkpoint_export import savee
from src.train.checkpoints import model_state_dict as checkpoint_model_state_dict
from src.train.deterministic_gpu import (
    reset_deterministic_caches,
    resolve_runtime_backend,
)

global_step = 0


def _coerce_auto_bool(value, field_name):
    if value in (None, "", "auto"):
        return "auto"
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    text = str(value).strip().lower()
    if text == "auto":
        return "auto"
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"{field_name} must be auto|true|false, got: {value!r}")


def _coerce_auto_non_negative_int(value, field_name):
    if value in (None, "", "auto"):
        return "auto"
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be auto or integer, got: {value!r}") from exc
    if number < 0:
        raise ValueError(f"{field_name} must be >= 0, got: {number}")
    return number


def _coerce_auto_positive_int(value, field_name):
    if value in (None, "", "auto"):
        return "auto"
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be auto or integer, got: {value!r}") from exc
    if number < 1:
        raise ValueError(f"{field_name} must be >= 1, got: {number}")
    return number


def resolve_dataloader_kwargs(hps, n_gpus):
    cpu_count = max(1, int(getattr(hps.runtime, "n_cpu", os.cpu_count() or 1)))
    worker_request = _coerce_auto_non_negative_int(
        getattr(hps.train, "num_workers", "auto"), "train.num_workers"
    )
    if worker_request == "auto":
        if os.name == "nt":
            num_workers = 0
        else:
            num_workers = max(1, min(4, cpu_count // max(1, n_gpus)))
    else:
        num_workers = worker_request

    persistent_request = _coerce_auto_bool(
        getattr(hps.train, "persistent_workers", "auto"), "train.persistent_workers"
    )
    if persistent_request == "auto":
        persistent_workers = num_workers > 0 and os.name != "nt"
    else:
        persistent_workers = bool(persistent_request)
    if persistent_workers and num_workers == 0:
        raise ValueError("train.persistent_workers=true requires train.num_workers > 0")

    prefetch_request = _coerce_auto_positive_int(
        getattr(hps.train, "prefetch_factor", "auto"), "train.prefetch_factor"
    )
    if prefetch_request == "auto":
        prefetch_factor = 2 if num_workers > 0 else None
    else:
        prefetch_factor = prefetch_request
    if prefetch_factor is not None and num_workers == 0:
        raise ValueError("train.prefetch_factor requires train.num_workers > 0")

    kwargs = {
        "num_workers": num_workers,
        "shuffle": False,
        "pin_memory": True,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def resolve_progress_bar_enabled(hps, rank):
    if rank != 0:
        return False
    use_tqdm_request = _coerce_auto_bool(
        getattr(hps.train, "use_tqdm", "auto"), "train.use_tqdm"
    )
    if use_tqdm_request == "auto":
        return sys.stderr.isatty()
    return bool(use_tqdm_request)


def format_progress_metrics(global_step, lr, loss_disc, loss_gen, loss_mel, loss_kl):
    return {
        "step": int(global_step),
        "lr": f"{float(lr):.2e}",
        "d": f"{loss_disc.detach().item():.3f}",
        "g": f"{loss_gen.detach().item():.3f}",
        "mel": f"{loss_mel.detach().item():.3f}",
        "kl": f"{loss_kl.detach().item():.3f}",
    }


def _numeric_backend(hps) -> str:
    return str(getattr(hps.train, "numeric_backend", "native")).strip().lower()


def _strict_repro_mode(hps) -> bool:
    return _numeric_backend(hps) == "deterministic_gpu"


def _runtime_backend(hps):
    return resolve_runtime_backend(_numeric_backend(hps), mel_processing)


def _step_seed(base_seed: int, epoch: int, batch_idx: int) -> int:
    return int(base_seed) + int(epoch) * 100000 + int(batch_idx)


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _seed_torch_for_step(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def configure_torch_runtime(runtime) -> None:
    deterministic_mode = str(
        getattr(runtime, "deterministic_algorithms", "off")
    ).strip().lower()
    if deterministic_mode not in {"off", "warn_only", "error"}:
        raise ValueError(
            f"Unsupported runtime.deterministic_algorithms={deterministic_mode!r}"
        )

    workspace_config = getattr(runtime, "cublas_workspace_config", None)
    if workspace_config:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(workspace_config)

    disable_tf32 = bool(getattr(runtime, "disable_tf32", False))
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = not disable_tf32
    torch.backends.cudnn.allow_tf32 = not disable_tf32
    torch.backends.cudnn.deterministic = deterministic_mode != "off"
    torch.backends.cudnn.benchmark = False

    if deterministic_mode == "off":
        torch.use_deterministic_algorithms(False)
    elif deterministic_mode == "warn_only":
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(True)


def _compute_y_hat_mel(hps, runtime_backend, y_hat, amp_device_type):
    with amp.autocast(amp_device_type, enabled=False):
        return runtime_backend.mel_spectrogram_torch(
            y_hat.float().squeeze(1),
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )


def _compute_loss_mel(y_mel, y_hat_mel, hps):
    if _numeric_backend(hps) == "native" and hps.train.fp16_run:
        y_hat_mel = y_hat_mel.half()
    return F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel


def _raise_if_non_finite_loss(name, value):
    scalar = value.detach().float()
    if not torch.isfinite(scalar).all():
        raise FloatingPointError(f"Non-finite {name}: {scalar.item()}")


def _raise_if_non_finite_gradients(model, label):
    nan_params = 0
    inf_params = 0
    first_nan: list[str] = []
    first_inf: list[str] = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if torch.isnan(grad).any():
            nan_params += 1
            if len(first_nan) < 5:
                first_nan.append(name)
        elif torch.isinf(grad).any():
            inf_params += 1
            if len(first_inf) < 5:
                first_inf.append(name)
    if nan_params or inf_params:
        raise FloatingPointError(
            f"Non-finite gradients in {label}: nan_param_count={nan_params} "
            f"inf_param_count={inf_params} first_nan={first_nan} first_inf={first_inf}"
        )


def _raise_if_non_finite_norm(label, value):
    if not math.isfinite(float(value)):
        raise FloatingPointError(f"Non-finite {label}: {value}")


def resolve_model_classes(version):
    if version == "v1":
        return (
            SynthesizerTrnMs256NSFsid,
            SynthesizerTrnMs256NSFsid_nono,
            MultiPeriodDiscriminator,
        )
    if version == "v2":
        return (
            SynthesizerTrnMs768NSFsid,
            SynthesizerTrnMs768NSFsid_nono,
            MultiPeriodDiscriminatorV2,
        )
    raise ValueError(f"Unsupported RVC version: {version}")


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def record(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time_str = str(datetime.timedelta(seconds=elapsed_time))
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{current_time}] | ({elapsed_time_str})"


def main(hps=None):
    if hps is None:
        hps = utils.get_hparams()
    os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
    configure_torch_runtime(hps.runtime)
    n_gpus = torch.cuda.device_count()
    if _strict_repro_mode(hps):
        if not torch.cuda.is_available():
            raise RuntimeError("train.numeric_backend=deterministic_gpu requires CUDA")
        if n_gpus != 1:
            raise RuntimeError(
                "train.numeric_backend=deterministic_gpu only supports a single visible GPU"
            )
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    logger = utils.get_logger(hps.model_dir)
    if n_gpus == 1:
        logger.info("running training in single-process mode")
        run(0, n_gpus, hps, logger)
        return

    children = []
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger),
        )
        children.append(subproc)
        subproc.start()

    for i in range(n_gpus):
        children[i].join()
        if children[i].exitcode != 0:
            raise RuntimeError(f"training worker {i} exited with code {children[i].exitcode}")


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _batch_uses_f0(batch: TrainingBatch) -> bool:
    return batch.pitch is not None and batch.pitchf is not None


def trim_audio_to_length(audio, length):
    return audio[..., : int(length)]


def trim_mel_to_length(mel, length):
    return mel[..., : int(length)]


def build_ddsp_validation_audio_dict(sample_name, gt_audio, pred_audio):
    return {
        f"{sample_name}/gt.wav": gt_audio.float().detach().cpu(),
        f"{sample_name}/pred.wav": pred_audio.float().detach().cpu(),
    }


def build_ddsp_validation_image_dict(sample_name, gt_mel, pred_mel):
    gt_mel = gt_mel.float().detach().cpu()
    pred_mel = pred_mel.float().detach().cpu()
    diff_mel = pred_mel - gt_mel
    return {
        sample_name: utils.plot_validation_mels_to_numpy(
            gt_mel.numpy(),
            pred_mel.numpy(),
            diff_mel.numpy(),
        )
    }


def extract_validation_sample_names(batch: TrainingBatch):
    return batch.sample_names


def infer_full_validation_audio(
    raw_g,
    batch: TrainingBatch,
):
    if _batch_uses_f0(batch):
        audio, _, _ = raw_g.infer(
            batch.phone,
            batch.phone_lengths,
            batch.pitch,
            batch.pitchf,
            batch.sid,
            return_length2=batch.spec_lengths,
        )
    else:
        audio, _, _ = raw_g.infer(
            batch.phone,
            batch.phone_lengths,
            batch.sid,
            return_length2=batch.spec_lengths,
        )
    return audio


def _forward_generator_batch(model, batch: TrainingBatch, *, validation: bool):
    if _batch_uses_f0(batch):
        if validation:
            return model.forward_val(
                batch.phone,
                batch.phone_lengths,
                batch.pitch,
                batch.pitchf,
                batch.spec,
                batch.spec_lengths,
                batch.sid,
            )
        return model(
            batch.phone,
            batch.phone_lengths,
            batch.pitch,
            batch.pitchf,
            batch.spec,
            batch.spec_lengths,
            batch.sid,
        )
    if validation:
        return model.forward_val(
            batch.phone,
            batch.phone_lengths,
            batch.spec,
            batch.spec_lengths,
            batch.sid,
        )
    return model(
        batch.phone,
        batch.phone_lengths,
        batch.spec,
        batch.spec_lengths,
        batch.sid,
    )


def _build_segment_targets(hps, runtime_backend, batch: TrainingBatch, ids_slice, y_hat, amp_device_type):
    mel = runtime_backend.spec_to_mel_torch(
        batch.spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax,
    )
    y_mel = commons.slice_segments(
        mel, ids_slice, hps.train.segment_size // hps.data.hop_length
    )
    y_hat_mel = _compute_y_hat_mel(hps, runtime_backend, y_hat, amp_device_type)
    wave_slice = commons.slice_segments(
        batch.wave, ids_slice * hps.data.hop_length, hps.train.segment_size
    )
    return mel, y_mel, y_hat_mel, wave_slice


def validate(
    epoch,
    hps,
    nets,
    val_loader,
    writer,
    logger,
    global_step,
    amp_device_type,
    runtime_backend,
):
    net_g, net_d = nets
    net_g.eval()
    net_d.eval()
    if _strict_repro_mode(hps):
        reset_deterministic_caches()

    loss_disc_sum = 0.0
    loss_gen_sum = 0.0
    loss_fm_sum = 0.0
    loss_mel_sum = 0.0
    loss_kl_sum = 0.0
    mel_val_mse_sum = 0.0
    n_batches = 0

    raw_g = _unwrap_model(net_g)
    raw_d = _unwrap_model(net_d)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if _strict_repro_mode(hps):
                _seed_torch_for_step(_step_seed(hps.train.seed, epoch, batch_idx))
            batch = move_batch_to_device(batch, 0)
            sample_names = extract_validation_sample_names(batch)
            sample_name = (
                sample_names[0]
                if sample_names is not None and len(sample_names) > 0
                else f"sample_{batch_idx}"
            )

            with amp.autocast(amp_device_type, enabled=hps.train.fp16_run):
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = _forward_generator_batch(raw_g, batch, validation=True)

                mel, y_mel, y_hat_mel, wave_slice = _build_segment_targets(
                    hps,
                    runtime_backend,
                    batch,
                    ids_slice,
                    y_hat,
                    amp_device_type,
                )

                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = raw_d(wave_slice, y_hat)
                with amp.autocast(amp_device_type, enabled=False):
                    loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                    loss_mel = _compute_loss_mel(y_mel, y_hat_mel, hps)
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, _ = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
                full_pred_audio = infer_full_validation_audio(raw_g, batch)
                full_pred_audio = full_pred_audio.float()
                full_gt_audio = batch.wave[0, 0].float()
                target_wave_length = int(batch.wave_lengths[0].item())
                compare_wave_length = min(target_wave_length, int(full_pred_audio.size(-1)))
                full_gt_audio = trim_audio_to_length(full_gt_audio, compare_wave_length)
                full_pred_audio = trim_audio_to_length(
                    full_pred_audio[0, 0],
                    compare_wave_length,
                )
                full_gt_mel = runtime_backend.mel_spectrogram_torch(
                    full_gt_audio.unsqueeze(0),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )[0]
                full_pred_mel = runtime_backend.mel_spectrogram_torch(
                    full_pred_audio.unsqueeze(0),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )[0]
                mel_frame_length = min(full_gt_mel.size(-1), full_pred_mel.size(-1))
                full_gt_mel = trim_mel_to_length(full_gt_mel, mel_frame_length)
                full_pred_mel = trim_mel_to_length(full_pred_mel, mel_frame_length)
                mel_val_mse = F.mse_loss(full_pred_mel, full_gt_mel)

            loss_disc_sum += loss_disc.item()
            loss_gen_sum += loss_gen_all.item()
            loss_fm_sum += loss_fm.item()
            loss_mel_sum += loss_mel.item()
            loss_kl_sum += loss_kl.item()
            mel_val_mse_sum += mel_val_mse.item()
            n_batches += 1

            if writer is not None:
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=build_ddsp_validation_image_dict(
                        sample_name,
                        full_gt_mel,
                        full_pred_mel,
                    ),
                    audios=build_ddsp_validation_audio_dict(
                        sample_name,
                        full_gt_audio,
                        full_pred_audio,
                    ),
                    audio_sampling_rate=hps.data.sampling_rate,
                )

    net_g.train()
    net_d.train()

    if n_batches == 0:
        return

    scalar_dict = {
        "validation/loss_d": loss_disc_sum / n_batches,
        "validation/loss_g": loss_gen_sum / n_batches,
        "validation/loss_fm": loss_fm_sum / n_batches,
        "validation/loss_mel": loss_mel_sum / n_batches,
        "validation/loss_kl": loss_kl_sum / n_batches,
        "validation/mel_val_mse": mel_val_mse_sum / n_batches,
    }

    if writer is not None:
        utils.summarize(
            writer=writer,
            global_step=global_step,
            scalars=scalar_dict,
        )

    if logger is not None:
        logger.info(
            "Validation Epoch %d: loss_d=%.4f, loss_g=%.4f, loss_fm=%.4f, loss_mel=%.4f, loss_kl=%.4f, mel_val_mse=%.4f",
            epoch,
            loss_disc_sum / n_batches,
            loss_gen_sum / n_batches,
            loss_fm_sum / n_batches,
            loss_mel_sum / n_batches,
            loss_kl_sum / n_batches,
            mel_val_mse_sum / n_batches,
        )


def run(rank, n_gpus, hps, logger: logging.Logger):
    global global_step
    use_distributed = n_gpus > 1
    if _strict_repro_mode(hps) and use_distributed:
        raise RuntimeError(
            "train.numeric_backend=deterministic_gpu only supports single-process training"
        )
    RVC_Model_f0, RVC_Model_nof0, Discriminator = resolve_model_classes(hps.version)
    writer = None
    try:
        configure_torch_runtime(hps.runtime)
        reset_deterministic_caches()
        runtime_backend = _runtime_backend(hps)
        if rank == 0:
            logger.info(hps)
            writer = SummaryWriter(log_dir=hps.model_dir)
            logger.info("using runtime backend: %s", runtime_backend.name)

        if use_distributed:
            dist.init_process_group(
                backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
            )
        _seed_everything(hps.train.seed)
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)

        val_loader = None
        training_files = hps.data.training_files
        val_filelist_path = hps.data.validation_files

        if hps.if_f0 == 1:
            train_dataset = TextAudioLoaderMultiNSFsid(
                training_files,
                hps.data,
                spectrogram_fn=runtime_backend.spectrogram_torch,
                spectrogram_cache_tag=runtime_backend.name,
            )
        else:
            train_dataset = TextAudioLoader(
                training_files,
                hps.data,
                spectrogram_fn=runtime_backend.spectrogram_torch,
                spectrogram_cache_tag=runtime_backend.name,
            )
        train_sampler = DistributedBucketSampler(
            train_dataset,
            hps.train.batch_size,
            [100, 200, 300, 400, 500, 600, 700, 800, 900],
            num_replicas=n_gpus,
            rank=rank,
            shuffle=True,
        )
        if hps.if_f0 == 1:
            collate_fn = TextAudioCollateMultiNSFsid()
        else:
            collate_fn = TextAudioCollate()
        dataloader_kwargs = resolve_dataloader_kwargs(hps, n_gpus)
        if rank == 0:
            logger.info("resolved train DataLoader kwargs: %s", dataloader_kwargs)
        train_loader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            batch_sampler=train_sampler,
            **dataloader_kwargs,
        )

        if rank == 0:
            val_filelist = Path(val_filelist_path)
            if not val_filelist.is_file():
                raise FileNotFoundError(
                    f"Missing validation filelist: {val_filelist}. "
                    "Run preprocessing to generate val_filelist.txt."
                )
            if hps.if_f0 == 1:
                val_dataset = TextAudioLoaderMultiNSFsid(
                    val_filelist_path,
                    hps.data,
                    spectrogram_fn=runtime_backend.spectrogram_torch,
                    spectrogram_cache_tag=runtime_backend.name,
                    return_sample_name=True,
                    filelist_label="f0 validation",
                )
                val_collate_fn = TextAudioCollateMultiNSFsid(return_sample_names=True)
            else:
                val_dataset = TextAudioLoader(
                    val_filelist_path,
                    hps.data,
                    spectrogram_fn=runtime_backend.spectrogram_torch,
                    spectrogram_cache_tag=runtime_backend.name,
                    return_sample_name=True,
                    filelist_label="validation",
                )
                val_collate_fn = TextAudioCollate(return_sample_names=True)
            val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=val_collate_fn,
                num_workers=0,
                pin_memory=False,
            )

        if hps.if_f0 == 1:
            net_g = RVC_Model_f0(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model,
                is_half=hps.train.fp16_run,
                sr=hps.sample_rate,
                numeric_backend=runtime_backend.name,
            )
        else:
            net_g = RVC_Model_nof0(
                hps.data.filter_length // 2 + 1,
                hps.train.segment_size // hps.data.hop_length,
                **hps.model,
                is_half=hps.train.fp16_run,
                numeric_backend=runtime_backend.name,
            )
        if torch.cuda.is_available():
            net_g = net_g.cuda(rank)
        net_d = Discriminator(
            hps.model.use_spectral_norm,
            deterministic_pad=runtime_backend.deterministic_discriminator_pad,
        )
        if torch.cuda.is_available():
            net_d = net_d.cuda(rank)
        optim_g = torch.optim.AdamW(
            net_g.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
        optim_d = torch.optim.AdamW(
            net_d.parameters(),
            hps.train.learning_rate,
            betas=hps.train.betas,
            eps=hps.train.eps,
        )
        if use_distributed:
            if torch.cuda.is_available():
                net_g = DDP(net_g, device_ids=[rank])
                net_d = DDP(net_d, device_ids=[rank])
            else:
                net_g = DDP(net_g)
                net_d = DDP(net_d)

        amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
        scaler = amp.GradScaler(
            amp_device_type,
            enabled=hps.train.fp16_run,
            init_scale=float(hps.train.grad_scaler_init_scale),
        )

        try:
            d_checkpoint = utils.latest_checkpoint_path(hps.model_dir, "D_*.pth")
            g_checkpoint = utils.latest_checkpoint_path(hps.model_dir, "G_*.pth")
        except FileNotFoundError:
            d_checkpoint = g_checkpoint = None

        if d_checkpoint is not None and g_checkpoint is not None:
            _, _, _, epoch_str = utils.load_checkpoint(
                d_checkpoint, net_d, optim_d
            )
            if rank == 0:
                logger.info("loaded D")
            _, _, _, epoch_str = utils.load_checkpoint(
                g_checkpoint, net_g, optim_g, scaler=scaler
            )
            global_step = (epoch_str - 1) * len(train_loader)
        else:
            epoch_str = 1
            global_step = 0
            if hps.pretrainG != "":
                if rank == 0:
                    logger.info("loaded pretrained %s" % (hps.pretrainG))
                if hasattr(net_g, "module"):
                    logger.info(
                        net_g.module.load_state_dict(
                            torch.load(hps.pretrainG, map_location="cpu")["model"]
                        )
                    )  ##测试不加载优化器
                else:
                    logger.info(
                        net_g.load_state_dict(
                            torch.load(hps.pretrainG, map_location="cpu")["model"]
                        )
                    )  ##测试不加载优化器
            if hps.pretrainD != "":
                if rank == 0:
                    logger.info("loaded pretrained %s" % (hps.pretrainD))
                if hasattr(net_d, "module"):
                    logger.info(
                        net_d.module.load_state_dict(
                            torch.load(hps.pretrainD, map_location="cpu")["model"]
                        )
                    )
                else:
                    logger.info(
                        net_d.load_state_dict(
                            torch.load(hps.pretrainD, map_location="cpu")["model"]
                        )
                    )

        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
        )
        if rank == 0:
            logger.info(
                "start training numeric_backend=%s deterministic_algorithms=%s grad_scaler_init_scale=%.1f",
                _numeric_backend(hps),
                getattr(hps.runtime, "deterministic_algorithms", "off"),
                float(hps.train.grad_scaler_init_scale),
            )

        cache = []
        for epoch in range(epoch_str, hps.train.epochs + 1):
            if rank == 0:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d],
                    [optim_g, optim_d],
                    scaler,
                    train_loader,
                    logger,
                    writer,
                    cache,
                    amp_device_type,
                    runtime_backend,
                )
            else:
                train_and_evaluate(
                    rank,
                    epoch,
                    hps,
                    [net_g, net_d],
                    [optim_g, optim_d],
                    scaler,
                    train_loader,
                    None,
                    None,
                    cache,
                    amp_device_type,
                    runtime_backend,
                )
            if rank == 0 and epoch % hps.save_every_epoch == 0:
                validate(
                    epoch, hps, [net_g, net_d], val_loader,
                    writer, logger, global_step, amp_device_type, runtime_backend,
                )
            scheduler_g.step()
            scheduler_d.step()
    finally:
        reset_deterministic_caches()
        if writer is not None:
            writer.flush()
            writer.close()
        if use_distributed and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _cuda_if_tensor(value, rank):
    if value is None or not torch.cuda.is_available():
        return value
    if isinstance(value, torch.Tensor):
        return value.cuda(rank, non_blocking=True)
    return value


def move_batch_to_device(batch: TrainingBatch, rank):
    if not torch.cuda.is_available():
        return batch
    return batch._replace(
        phone=_cuda_if_tensor(batch.phone, rank),
        phone_lengths=_cuda_if_tensor(batch.phone_lengths, rank),
        pitch=_cuda_if_tensor(batch.pitch, rank),
        pitchf=_cuda_if_tensor(batch.pitchf, rank),
        spec=_cuda_if_tensor(batch.spec, rank),
        spec_lengths=_cuda_if_tensor(batch.spec_lengths, rank),
        wave=_cuda_if_tensor(batch.wave, rank),
        wave_lengths=_cuda_if_tensor(batch.wave_lengths, rank),
        sid=_cuda_if_tensor(batch.sid, rank),
        ids_sorted_decreasing=_cuda_if_tensor(batch.ids_sorted_decreasing, rank),
    )


def prepare_epoch_iterator(train_loader, cache, rank, hps):
    if not hps.if_cache_data_in_gpu:
        return enumerate(train_loader)

    if not cache:
        for batch_idx, batch in enumerate(train_loader):
            cache.append(
                (
                    batch_idx,
                    move_batch_to_device(batch, rank),
                )
            )
    else:
        shuffle(cache)
    return cache


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    scaler,
    train_loader,
    logger,
    writer,
    cache,
    amp_device_type,
    runtime_backend,
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    strict_mode = _strict_repro_mode(hps)

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    data_iterator = prepare_epoch_iterator(train_loader, cache, rank, hps)
    progress = tqdm(
        total=len(train_loader),
        desc=f"Epoch {epoch}/{hps.train.epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
        disable=not resolve_progress_bar_enabled(hps, rank),
    )

    # Run steps
    epoch_recorder = EpochRecorder()
    try:
        for batch_idx, batch in data_iterator:
            if strict_mode:
                _seed_torch_for_step(_step_seed(hps.train.seed, epoch, batch_idx))
            if not hps.if_cache_data_in_gpu:
                batch = move_batch_to_device(batch, rank)

            # Calculate
            with amp.autocast(amp_device_type, enabled=hps.train.fp16_run):
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = _forward_generator_batch(net_g, batch, validation=False)
                mel, y_mel, y_hat_mel, wave_slice = _build_segment_targets(
                    hps,
                    runtime_backend,
                    batch,
                    ids_slice,
                    y_hat,
                    amp_device_type,
                )

                # Discriminator
                y_d_hat_r, y_d_hat_g, _, _ = net_d(wave_slice, y_hat.detach())
                with amp.autocast(amp_device_type, enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        y_d_hat_r, y_d_hat_g
                    )
            _raise_if_non_finite_loss("loss_disc", loss_disc)
            optim_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optim_d)
            grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
            _raise_if_non_finite_gradients(net_d, "discriminator")
            _raise_if_non_finite_norm("grad_norm_d", grad_norm_d)
            scaler.step(optim_d)

            with amp.autocast(amp_device_type, enabled=hps.train.fp16_run):
                # Generator
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave_slice, y_hat)
                with amp.autocast(amp_device_type, enabled=False):
                    loss_mel = _compute_loss_mel(y_mel, y_hat_mel, hps)
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_r, fmap_g)
                    loss_gen, losses_gen = generator_loss(y_d_hat_g)
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            _raise_if_non_finite_loss("loss_gen", loss_gen)
            _raise_if_non_finite_loss("loss_fm", loss_fm)
            _raise_if_non_finite_loss("loss_mel", loss_mel)
            _raise_if_non_finite_loss("loss_kl", loss_kl)
            _raise_if_non_finite_loss("loss_gen_all", loss_gen_all)
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
            _raise_if_non_finite_gradients(net_g, "generator")
            _raise_if_non_finite_norm("grad_norm_g", grad_norm_g)
            scaler.step(optim_g)
            scaler.update()

            progress.update(1)

            if rank == 0:
                if global_step % hps.train.log_interval == 0:
                    lr = optim_g.param_groups[0]["lr"]
                    progress.set_postfix(
                        format_progress_metrics(
                            global_step, lr, loss_disc, loss_gen, loss_mel, loss_kl
                        ),
                        refresh=False,
                    )
                    if logger is not None:
                        logger.info(
                            "Train Epoch: {} [{:.0f}%]".format(
                                epoch, 100.0 * batch_idx / len(train_loader)
                            )
                        )
                    # Amor For Tensorboard display
                    if loss_mel > 75:
                        loss_mel = 75
                    if loss_kl > 9:
                        loss_kl = 9

                    if logger is not None:
                        logger.info([global_step, lr])
                        logger.info(
                            f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                        )
                    scalar_dict = {
                        "loss/g/total": loss_gen_all,
                        "loss/d/total": loss_disc,
                        "learning_rate": lr,
                        "grad_norm_d": grad_norm_d,
                        "grad_norm_g": grad_norm_g,
                    }
                    scalar_dict.update(
                        {
                            "loss/g/fm": loss_fm,
                            "loss/g/mel": loss_mel,
                            "loss/g/kl": loss_kl,
                        }
                    )

                    scalar_dict.update(
                        {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                    )
                    scalar_dict.update(
                        {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                    )
                    scalar_dict.update(
                        {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                    )
                    image_dict = {
                        "slice/mel_org": utils.plot_spectrogram_to_numpy(
                            y_mel[0].detach().cpu().numpy()
                        ),
                        "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                            y_hat_mel[0].detach().cpu().numpy()
                        ),
                        "all/mel": utils.plot_spectrogram_to_numpy(
                            mel[0].detach().cpu().numpy()
                        ),
                    }
                    if writer is not None:
                        utils.summarize(
                            writer=writer,
                            global_step=global_step,
                            images=image_dict,
                            scalars=scalar_dict,
                        )
            global_step += 1
    finally:
        progress.close()
    # /Run steps

    save_epoch_checkpoints(
        rank, epoch, hps, net_g, net_d, optim_g, optim_d, scaler, logger, global_step
    )

    if rank == 0 and logger is not None:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    if save_final_checkpoint(rank, epoch, hps, net_g, logger):
        sleep(1)
        return


def save_epoch_checkpoints(
    rank, epoch, hps, net_g, net_d, optim_g, optim_d, scaler, logger, step
):
    if epoch % hps.save_every_epoch != 0 or rank != 0:
        return

    checkpoint_step = 2333333 if hps.if_latest != 0 else step
    utils.save_checkpoint(
        net_g,
        optim_g,
        hps.train.learning_rate,
        epoch,
        os.path.join(hps.model_dir, "G_{}.pth".format(checkpoint_step)),
        scaler=scaler,
    )
    utils.save_checkpoint(
        net_d,
        optim_d,
        hps.train.learning_rate,
        epoch,
        os.path.join(hps.model_dir, "D_{}.pth".format(checkpoint_step)),
        scaler=scaler,
    )
    if hps.save_every_weights == "1" and logger is not None:
        logger.info(
            "saving ckpt %s_e%s:%s",
            hps.name,
            epoch,
            savee(
                checkpoint_model_state_dict(net_g),
                hps.sample_rate,
                hps.if_f0,
                hps.name + "_e%s_s%s" % (epoch, step),
                epoch,
                hps.version,
                hps,
            ),
        )


def save_final_checkpoint(rank, epoch, hps, net_g, logger):
    if epoch < hps.total_epoch or rank != 0:
        return False
    if logger is not None:
        logger.info("Training is done. The program is closed.")
        logger.info(
            "saving final ckpt:%s",
            savee(
                checkpoint_model_state_dict(net_g),
                hps.sample_rate,
                hps.if_f0,
                hps.name,
                epoch,
                hps.version,
                hps,
            ),
        )
    else:
        savee(
            checkpoint_model_state_dict(net_g),
            hps.sample_rate,
            hps.if_f0,
            hps.name,
            epoch,
            hps.version,
            hps,
        )
    return True


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
