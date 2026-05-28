import math
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm
from src.models import attentions, commons, modules
from src.models.commons import get_padding, init_weights
from src.train.deterministic_gpu import deterministic_f02sine, reflect_pad_last

__all__ = [
    "TextEncoder",
    "ResidualCouplingBlock",
    "PosteriorEncoder",
    "Generator",
    "SineGen",
    "SourceModuleHnNSF",
    "GeneratorNSF",
    "SynthesizerTrnMs256NSFsid",
    "SynthesizerTrnMs768NSFsid",
    "SynthesizerTrnMs256NSFsid_nono",
    "SynthesizerTrnMs768NSFsid_nono",
    "MultiPeriodDiscriminator",
    "MultiPeriodDiscriminatorV2",
    "DiscriminatorS",
    "DiscriminatorP",
]


def build_export_model_config(hps) -> list:
    return [
        hps.data.filter_length // 2 + 1,
        32,
        hps.model.inter_channels,
        hps.model.hidden_channels,
        hps.model.filter_channels,
        hps.model.n_heads,
        hps.model.n_layers,
        hps.model.kernel_size,
        hps.model.p_dropout,
        hps.model.resblock,
        hps.model.resblock_kernel_sizes,
        hps.model.resblock_dilation_sizes,
        hps.model.upsample_rates,
        hps.model.upsample_initial_channel,
        hps.model.upsample_kernel_sizes,
        hps.model.spk_embed_dim,
        hps.model.gin_channels,
        hps.data.sampling_rate,
    ]


class TextEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        f0=True,
    ):
        super(TextEncoder, self).__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.emb_phone = nn.Linear(in_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self,
        phone: torch.Tensor,
        pitch: torch.Tensor,
        lengths: torch.Tensor,
        skip_head: Optional[torch.Tensor] = None,
    ):
        if pitch is None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask)
        if skip_head is not None:
            if not isinstance(skip_head, torch.Tensor):
                raise TypeError("skip_head must be a torch.Tensor")
            head = int(skip_head.item())
            x = x[:, :, head:]
            x_mask = x_mask[:, :, head:]
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in self.flows[::-1]:
                x, _ = flow.forward(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super(PosteriorEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(
        self, x: torch.Tensor, x_lengths: torch.Tensor, g: Optional[torch.Tensor] = None
    ):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


def apply_resblock_stack(resblocks, num_kernels, x, upsample_index, module_name):
    start = upsample_index * num_kernels
    xs: Optional[torch.Tensor] = None
    for resblock in resblocks[start : start + num_kernels]:
        y = resblock(x)
        xs = y if xs is None else xs + y
    if xs is None:
        raise RuntimeError(f"{module_name} has no residual blocks for upsample layer")
    return xs / num_kernels


def maybe_resize_latent(x: torch.Tensor, n_res: Optional[torch.Tensor]) -> torch.Tensor:
    if n_res is None:
        return x
    if not isinstance(n_res, torch.Tensor):
        raise TypeError("n_res must be a torch.Tensor")
    n = int(n_res.item())
    if n != x.shape[-1]:
        x = F.interpolate(x, size=n, mode="linear")
    return x


def run_generator_stack(
    module,
    x: torch.Tensor,
    g: Optional[torch.Tensor],
    *,
    module_name: str,
    source_provider=None,
) -> torch.Tensor:
    x = module.conv_pre(x)
    if g is not None:
        x = x + module.cond(g)
    for i in range(module.num_upsamples):
        x = F.leaky_relu(x, modules.LRELU_SLOPE)
        x = module.ups[i](x)
        if source_provider is not None:
            x = x + source_provider(i)
        x = apply_resblock_stack(module.resblocks, module.num_kernels, x, i, module_name)
    x = F.leaky_relu(x)
    x = module.conv_post(x)
    return torch.tanh(x)


def remove_generator_weight_norm(module) -> None:
    for layer in module.ups:
        remove_weight_norm(layer)
    for layer in module.resblocks:
        layer.remove_weight_norm()


def init_generator_backbone(
    module,
    initial_channel,
    resblock,
    resblock_kernel_sizes,
    resblock_dilation_sizes,
    upsample_rates,
    upsample_initial_channel,
    upsample_kernel_sizes,
    gin_channels=0,
):
    module.num_kernels = len(resblock_kernel_sizes)
    module.num_upsamples = len(upsample_rates)
    module.conv_pre = Conv1d(
        initial_channel, upsample_initial_channel, 7, 1, padding=3
    )
    resblock_cls = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

    channels = []
    module.ups = nn.ModuleList()
    for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
        c_cur = upsample_initial_channel // (2 ** (i + 1))
        channels.append(c_cur)
        module.ups.append(
            weight_norm(
                ConvTranspose1d(
                    upsample_initial_channel // (2**i),
                    c_cur,
                    k,
                    u,
                    padding=(k - u) // 2,
                )
            )
        )

    module.resblocks = nn.ModuleList()
    for ch in channels:
        for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
            module.resblocks.append(resblock_cls(ch, k, d))

    module.conv_post = Conv1d(channels[-1], 1, 7, 1, padding=3, bias=False)
    module.ups.apply(init_weights)

    if gin_channels != 0:
        module.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
    return channels


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        init_generator_backbone(
            self,
            initial_channel,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        n_res: Optional[torch.Tensor] = None,
    ):
        return run_generator_stack(
            self,
            maybe_resize_latent(x, n_res),
            g,
            module_name="Generator",
        )

    def remove_weight_norm(self):
        remove_generator_weight_norm(self)


class SineGen(torch.nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)
    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)
    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(torch.pi) or cos(0)
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
        deterministic_backend=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self._deterministic_backend = bool(deterministic_backend)

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv
    
    def _native_f02sine(self, f0, upp):
        """ f0: (batchsize, length, dim)
            where dim indicates fundamental tone and overtones
        """
        a = torch.arange(1, upp + 1, dtype=f0.dtype, device=f0.device)
        rad = f0 / self.sampling_rate * a
        rad2 = torch.fmod(rad[:, :-1, -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad += F.pad(rad_acc, (0, 0, 1, 0), mode='constant')
        rad = rad.reshape(f0.shape[0], -1, 1)
        b = torch.arange(1, self.dim + 1, dtype=f0.dtype, device=f0.device).reshape(1, 1, -1)
        rad *= b
        rand_ini = torch.rand(1, 1, self.dim, device=f0.device)
        rand_ini[..., 0] = 0
        rad += rand_ini
        sines = torch.sin(2 * np.pi * rad)
        return sines

    def _f02sine(self, f0, upp):
        if self._deterministic_backend:
            return deterministic_f02sine(self, f0, upp)
        return self._native_f02sine(f0, upp)
        
    def forward(self, f0: torch.Tensor, upp: int):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0 = f0.unsqueeze(-1)
            sine_waves = self._f02sine(f0, upp) * self.sine_amp
            uv = self._f02uv(f0)
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


class SourceModuleHnNSF(torch.nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)
    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
        is_half=True,
        deterministic_backend=False,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std
        self.is_half = is_half
        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshod,
            deterministic_backend=deterministic_backend,
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, x: torch.Tensor, upp: int = 1):
        sine_wavs, uv, _ = self.l_sin_gen(x, upp)
        #     sine_wavs = sine_wavs.half()
        # sine_merge = self.l_tanh(self.l_linear(sine_wavs.to(x)))
        sine_wavs = sine_wavs.to(dtype=self.l_linear.weight.dtype)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))
        return sine_merge, None, None  # noise, uv


class GeneratorNSF(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        sr,
        is_half=False,
        deterministic_backend=False,
    ):
        super(GeneratorNSF, self).__init__()
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sr,
            harmonic_num=0,
            is_half=is_half,
            deterministic_backend=deterministic_backend,
        )
        channels = init_generator_backbone(
            self,
            initial_channel,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.noise_convs = nn.ModuleList()
        for i, c_cur in enumerate(channels):
            if i + 1 < len(upsample_rates):
                stride_f0 = math.prod(upsample_rates[i + 1 :])
                self.noise_convs.append(
                    Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(Conv1d(1, c_cur, kernel_size=1))

        self.upp = math.prod(upsample_rates)

        self.lrelu_slope = modules.LRELU_SLOPE

    def forward(
        self,
        x,
        f0,
        g: Optional[torch.Tensor] = None,
        n_res: Optional[torch.Tensor] = None,
    ):
        har_source, noi_source, uv = self.m_source(f0, self.upp)
        har_source = har_source.transpose(1, 2)
        if n_res is not None:
            if not isinstance(n_res, torch.Tensor):
                raise TypeError("n_res must be a torch.Tensor")
            n = int(n_res.item())
            if n * self.upp != har_source.shape[-1]:
                har_source = F.interpolate(har_source, size=n * self.upp, mode="linear")
        return run_generator_stack(
            self,
            maybe_resize_latent(x, n_res),
            g,
            module_name="GeneratorNSF",
            source_provider=lambda index: self.noise_convs[index](har_source),
        )

    def remove_weight_norm(self):
        remove_generator_weight_norm(self)


sr2sr = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def init_synthesizer_modules(
    module,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock,
    resblock_kernel_sizes,
    resblock_dilation_sizes,
    upsample_rates,
    upsample_initial_channel,
    upsample_kernel_sizes,
    spk_embed_dim,
    gin_channels,
    *,
    sr=None,
    feature_dim=256,
    use_f0=True,
    is_half=False,
    numeric_backend="native",
):
    if use_f0 and isinstance(sr, str):
        sr = sr2sr[sr]
    module.spec_channels = spec_channels
    module.inter_channels = inter_channels
    module.hidden_channels = hidden_channels
    module.filter_channels = filter_channels
    module.n_heads = n_heads
    module.n_layers = n_layers
    module.kernel_size = kernel_size
    module.p_dropout = float(p_dropout)
    module.resblock = resblock
    module.resblock_kernel_sizes = resblock_kernel_sizes
    module.resblock_dilation_sizes = resblock_dilation_sizes
    module.upsample_rates = upsample_rates
    module.upsample_initial_channel = upsample_initial_channel
    module.upsample_kernel_sizes = upsample_kernel_sizes
    module.segment_size = segment_size
    module.gin_channels = gin_channels
    module.spk_embed_dim = spk_embed_dim
    module.use_f0 = bool(use_f0)
    module.numeric_backend = str(numeric_backend)
    module.enc_p = TextEncoder(
        feature_dim,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        float(p_dropout),
        f0=use_f0,
    )
    if use_f0:
        module.dec = GeneratorNSF(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
            sr=sr,
            is_half=is_half,
            deterministic_backend=module.numeric_backend == "deterministic_gpu",
        )
    else:
        module.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
    module.enc_q = PosteriorEncoder(
        spec_channels,
        inter_channels,
        hidden_channels,
        5,
        1,
        16,
        gin_channels=gin_channels,
    )
    module.flow = ResidualCouplingBlock(
        inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
    )
    module.emb_g = nn.Embedding(module.spk_embed_dim, gin_channels)


class SynthesizerTrnMs256NSFsid(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
        feature_dim=256,
        use_f0=True,
        **kwargs
    ):
        super(SynthesizerTrnMs256NSFsid, self).__init__()
        numeric_backend = kwargs.get("numeric_backend", "native")
        init_synthesizer_modules(
            self,
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr=sr,
            feature_dim=feature_dim,
            use_f0=use_f0,
            is_half=kwargs.get("is_half", False),
            numeric_backend=numeric_backend,
        )

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        if hasattr(self, "enc_q"):
            self.enc_q.remove_weight_norm()

    def _speaker_embedding(self, ds: Optional[torch.Tensor]) -> torch.Tensor:
        return self.emb_g(ds).unsqueeze(-1)

    def _forward_common(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: Optional[torch.Tensor],
        *,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        center: bool = False,
    ):
        g = self._speaker_embedding(ds)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        slicer = commons.center_slice_segments if center else commons.rand_slice_segments
        z_slice, ids_slice = slicer(z, y_lengths, self.segment_size)
        if self.use_f0:
            if pitchf is None:
                raise ValueError("pitchf is required when use_f0=True")
            pitchf = commons.slice_segments2(pitchf, ids_slice, self.segment_size)
            o = self.dec(z_slice, pitchf, g=g)
        else:
            o = self.dec(z_slice, g=g)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def _reconstruct_full_common(
        self,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: Optional[torch.Tensor],
        *,
        pitchf: Optional[torch.Tensor] = None,
    ):
        g = self._speaker_embedding(ds)
        z, _, _, y_mask = self.enc_q(y, y_lengths, g=g)
        if self.use_f0:
            if pitchf is None:
                raise ValueError("pitchf is required when use_f0=True")
            return self.dec(z * y_mask, pitchf, g=g, n_res=y_lengths)
        return self.dec(z * y_mask, g=g, n_res=y_lengths)

    def _infer_common(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        sid: torch.Tensor,
        *,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        skip_head: Optional[torch.Tensor] = None,
        return_length: Optional[torch.Tensor] = None,
        return_length2: Optional[torch.Tensor] = None,
    ):
        g = self._speaker_embedding(sid)
        if skip_head is not None and return_length is not None:
            if not isinstance(skip_head, torch.Tensor):
                raise TypeError("skip_head must be a torch.Tensor")
            if not isinstance(return_length, torch.Tensor):
                raise TypeError("return_length must be a torch.Tensor")
            head = int(skip_head.item())
            length = int(return_length.item())
            flow_head = torch.clamp(skip_head - 24, min=0)
            dec_head = head - int(flow_head.item())
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths, flow_head)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            z = z[:, :, dec_head : dec_head + length]
            x_mask = x_mask[:, :, dec_head : dec_head + length]
            if self.use_f0 and nsff0 is not None:
                nsff0 = nsff0[:, head : head + length]
        else:
            m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
            z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
            z = self.flow(z_p, x_mask, g=g, reverse=True)
        if self.use_f0:
            if nsff0 is None:
                raise ValueError("nsff0 is required when use_f0=True")
            o = self.dec(z * x_mask, nsff0, g=g, n_res=return_length2)
        else:
            o = self.dec(z * x_mask, g=g, n_res=return_length2)
        return o, x_mask, (z, z_p, m_p, logs_p)

    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: Optional[torch.Tensor] = None,
    ):
        return self._forward_common(
            phone,
            phone_lengths,
            y,
            y_lengths,
            ds,
            pitch=pitch,
            pitchf=pitchf,
        )

    def forward_val(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: Optional[torch.Tensor] = None,
    ):
        return self._forward_common(
            phone,
            phone_lengths,
            y,
            y_lengths,
            ds,
            pitch=pitch,
            pitchf=pitchf,
            center=True,
        )

    def reconstruct_full(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        ds: Optional[torch.Tensor] = None,
    ):
        return self._reconstruct_full_common(y, y_lengths, ds, pitchf=pitchf)

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: torch.Tensor,
        nsff0: torch.Tensor,
        sid: torch.Tensor,
        skip_head: Optional[torch.Tensor] = None,
        return_length: Optional[torch.Tensor] = None,
        return_length2: Optional[torch.Tensor] = None,
    ):
        return self._infer_common(
            phone,
            phone_lengths,
            sid,
            pitch=pitch,
            nsff0=nsff0,
            skip_head=skip_head,
            return_length=return_length,
            return_length2=return_length2,
        )


class SynthesizerTrnMs768NSFsid(SynthesizerTrnMs256NSFsid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, feature_dim=768, **kwargs)


class SynthesizerTrnMs256NSFsid_nono(SynthesizerTrnMs256NSFsid):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr=None,
        feature_dim=256,
        **kwargs
    ):
        super().__init__(
            spec_channels,
            segment_size,
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            spk_embed_dim,
            gin_channels,
            sr,
            feature_dim=feature_dim,
            use_f0=False,
            **kwargs
        )

    def forward(self, phone, phone_lengths, y, y_lengths, ds):
        return self._forward_common(phone, phone_lengths, y, y_lengths, ds)

    def forward_val(self, phone, phone_lengths, y, y_lengths, ds):
        return self._forward_common(phone, phone_lengths, y, y_lengths, ds, center=True)

    def reconstruct_full(self, phone, phone_lengths, y, y_lengths, ds):
        return self._reconstruct_full_common(y, y_lengths, ds)

    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        sid: torch.Tensor,
        skip_head: Optional[torch.Tensor] = None,
        return_length: Optional[torch.Tensor] = None,
        return_length2: Optional[torch.Tensor] = None,
    ):
        return self._infer_common(
            phone,
            phone_lengths,
            sid,
            skip_head=skip_head,
            return_length=return_length,
            return_length2=return_length2,
        )


class SynthesizerTrnMs768NSFsid_nono(SynthesizerTrnMs256NSFsid_nono):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, feature_dim=768, **kwargs)


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, periods=None, deterministic_pad=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = periods or [2, 3, 5, 7, 11, 17]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(
                i,
                use_spectral_norm=use_spectral_norm,
                deterministic_pad=deterministic_pad,
            )
            for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiPeriodDiscriminatorV2(MultiPeriodDiscriminator):
    def __init__(self, use_spectral_norm=False, deterministic_pad=False):
        super(MultiPeriodDiscriminatorV2, self).__init__(
            use_spectral_norm=use_spectral_norm,
            periods=[2, 3, 5, 7, 11, 17, 23, 37],
            deterministic_pad=deterministic_pad,
        )


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        period,
        kernel_size=5,
        stride=3,
        use_spectral_norm=False,
        deterministic_pad=False,
    ):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        self.deterministic_pad = bool(deterministic_pad)
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            if self.deterministic_pad:
                x = reflect_pad_last(x, 0, n_pad)
            else:
                x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

