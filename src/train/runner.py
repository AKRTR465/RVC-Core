import os
import logging

import datetime

from src.train import utils

from random import randint, shuffle

import torch
from torch.cuda.amp import GradScaler, autocast

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
from src.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from src.train.checkpoint_export import savee

global_step = 0


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
    n_gpus = torch.cuda.device_count()
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    logger = utils.get_logger(hps.model_dir)
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


def run(rank, n_gpus, hps, logger: logging.Logger):
    global global_step
    RVC_Model_f0, RVC_Model_nof0, Discriminator = resolve_model_classes(hps.version)
    if rank == 0:
        logger.info(hps)
        writer = SummaryWriter(log_dir=hps.model_dir)

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
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
    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = Discriminator(hps.model.use_spectral_norm)
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
    if torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    else:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

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
            g_checkpoint, net_g, optim_g
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

    scaler = GradScaler(enabled=hps.train.fp16_run)

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
            )
        scheduler_g.step()
        scheduler_d.step()


def move_batch_to_device(info, rank, use_f0, include_wave_lengths):
    if not torch.cuda.is_available():
        return info

    values = list(info)
    if use_f0:
        cuda_indices = [0, 1, 2, 3, 4, 5, 6, 8]
        if include_wave_lengths:
            cuda_indices.append(7)
    else:
        cuda_indices = [0, 1, 2, 3, 4, 6]
        if include_wave_lengths:
            cuda_indices.append(5)

    for index in cuda_indices:
        values[index] = values[index].cuda(rank, non_blocking=True)
    return tuple(values)


def prepare_epoch_iterator(train_loader, cache, rank, hps):
    if not hps.if_cache_data_in_gpu:
        return enumerate(train_loader)

    use_f0 = hps.if_f0 == 1
    if not cache:
        for batch_idx, info in enumerate(train_loader):
            cache.append(
                (
                    batch_idx,
                    move_batch_to_device(
                        info, rank, use_f0, include_wave_lengths=True
                    ),
                )
            )
    else:
        shuffle(cache)
    return cache


def unpack_training_batch(info, use_f0):
    if use_f0:
        return info

    phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
    return phone, phone_lengths, None, None, spec, spec_lengths, wave, wave_lengths, sid


def train_and_evaluate(
    rank, epoch, hps, nets, optims, scaler, train_loader, logger, writer, cache
):
    net_g, net_d = nets
    optim_g, optim_d = optims

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    use_f0 = hps.if_f0 == 1
    data_iterator = prepare_epoch_iterator(train_loader, cache, rank, hps)

    # Run steps
    epoch_recorder = EpochRecorder()
    for batch_idx, info in data_iterator:
        if not hps.if_cache_data_in_gpu:
            info = move_batch_to_device(
                info, rank, use_f0, include_wave_lengths=False
            )
        (
            phone,
            phone_lengths,
            pitch,
            pitchf,
            spec,
            spec_lengths,
            wave,
            wave_lengths,
            sid,
        ) = unpack_training_batch(info, use_f0)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            if use_f0:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
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
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1
    # /Run steps

    save_epoch_checkpoints(
        rank, epoch, hps, net_g, net_d, optim_g, optim_d, logger, global_step
    )

    if rank == 0:
        logger.info("====> Epoch: {} {}".format(epoch, epoch_recorder.record()))
    if save_final_checkpoint(rank, epoch, hps, net_g, logger):
        sleep(1)
        return


def model_state_dict(model):
    return model.module.state_dict() if hasattr(model, "module") else model.state_dict()


def save_epoch_checkpoints(
    rank, epoch, hps, net_g, net_d, optim_g, optim_d, logger, step
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
    )
    utils.save_checkpoint(
        net_d,
        optim_d,
        hps.train.learning_rate,
        epoch,
        os.path.join(hps.model_dir, "D_{}.pth".format(checkpoint_step)),
    )
    if hps.save_every_weights == "1":
        logger.info(
            "saving ckpt %s_e%s:%s",
            hps.name,
            epoch,
            savee(
                model_state_dict(net_g),
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
    logger.info("Training is done. The program is closed.")
    logger.info(
        "saving final ckpt:%s",
        savee(
            model_state_dict(net_g),
            hps.sample_rate,
            hps.if_f0,
            hps.name,
            epoch,
            hps.version,
            hps,
        ),
    )
    return True


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()

