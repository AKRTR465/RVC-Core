"""Minimal fairseq-compatible HuBERT inference implementation.

This module implements only the HuBERT Base inference path used by RVC.  The
module layout and parameter names intentionally match fairseq's MIT-licensed
HuBERT/Wav2Vec2 implementation so existing ``hubert_base.pt`` checkpoints load
with ``strict=True`` without depending on fairseq at runtime.
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_CONV_FEATURE_LAYERS = (
    (512, 10, 5),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 3, 2),
    (512, 2, 2),
    (512, 2, 2),
)
DEFAULT_CONV_FEATURE_LAYERS_EXPR = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"


def _get_mapping_value(mapping, key, default=None):
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _nested_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _nested_namespace(v) for k, v in value.items()})
    return value


def checkpoint_cfg_to_namespace(cfg):
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        return cfg
    return _nested_namespace(cfg)


def _parse_conv_feature_layers(value):
    if value is None:
        return DEFAULT_CONV_FEATURE_LAYERS
    if isinstance(value, str):
        compact = value.replace(" ", "")
        if compact == DEFAULT_CONV_FEATURE_LAYERS_EXPR.replace(" ", ""):
            return DEFAULT_CONV_FEATURE_LAYERS
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Unsupported HuBERT conv_feature_layers: {value!r}") from exc
    layers = tuple(tuple(int(part) for part in layer) for layer in value)
    if not layers or any(len(layer) != 3 for layer in layers):
        raise ValueError(f"Invalid HuBERT conv_feature_layers: {value!r}")
    return layers


def _count_encoder_layers(state_dict):
    indices = set()
    pattern = re.compile(r"^encoder\.layers\.(\d+)\.")
    for key in state_dict:
        match = pattern.match(key)
        if match is not None:
            indices.add(int(match.group(1)))
    return max(indices) + 1 if indices else None


def _activation_fn(name):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported HuBERT activation_fn: {name!r}")


@dataclass(frozen=True)
class HubertConfig:
    conv_layers: tuple[tuple[int, int, int], ...]
    extractor_mode: str = "default"
    conv_bias: bool = False
    encoder_layers: int = 12
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 3072
    encoder_attention_heads: int = 12
    activation_fn: str = "gelu"
    layer_norm_first: bool = False
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.0
    encoder_layerdrop: float = 0.0
    dropout_input: float = 0.0
    dropout_features: float = 0.0
    final_dim: int = 256
    label_embs_num: int = 0
    conv_pos: int = 128
    conv_pos_groups: int = 16
    required_seq_len_multiple: int = 2
    mask_prob: float = 0.8
    mask_length: int = 10
    mask_channel_prob: float = 0.0


def hubert_config_from_checkpoint(cfg, state_dict):
    if not isinstance(state_dict, dict):
        raise ValueError("HuBERT checkpoint does not contain a valid model state dict")

    model_cfg = _get_mapping_value(cfg, "model", {}) if cfg is not None else {}
    model_name = _get_mapping_value(model_cfg, "_name", "hubert")
    if model_name not in (None, "hubert"):
        raise ValueError(f"Unsupported HuBERT checkpoint model: {model_name!r}")

    layer_type = _get_mapping_value(model_cfg, "layer_type", "transformer")
    if layer_type != "transformer":
        raise ValueError(f"Unsupported HuBERT layer_type: {layer_type!r}")

    extractor_mode = _get_mapping_value(model_cfg, "extractor_mode", "default")
    if extractor_mode != "default":
        raise ValueError(f"Unsupported HuBERT extractor_mode: {extractor_mode!r}")

    if bool(_get_mapping_value(model_cfg, "target_glu", False)):
        raise ValueError("Unsupported HuBERT target_glu checkpoint")
    if int(_get_mapping_value(model_cfg, "pos_conv_depth", 1)) != 1:
        raise ValueError("Unsupported HuBERT pos_conv_depth checkpoint")

    conv_layers = _parse_conv_feature_layers(
        _get_mapping_value(model_cfg, "conv_feature_layers", None)
    )
    inferred_layers = _count_encoder_layers(state_dict)
    final_proj_weight = state_dict.get("final_proj.weight")
    if final_proj_weight is None:
        raise ValueError("HuBERT checkpoint is missing final_proj.weight")

    label_embs = state_dict.get("label_embs_concat")
    return HubertConfig(
        conv_layers=conv_layers,
        extractor_mode=extractor_mode,
        conv_bias=bool(_get_mapping_value(model_cfg, "conv_bias", False)),
        encoder_layers=int(
            _get_mapping_value(model_cfg, "encoder_layers", inferred_layers or 12)
        ),
        encoder_embed_dim=int(_get_mapping_value(model_cfg, "encoder_embed_dim", 768)),
        encoder_ffn_embed_dim=int(
            _get_mapping_value(model_cfg, "encoder_ffn_embed_dim", 3072)
        ),
        encoder_attention_heads=int(
            _get_mapping_value(model_cfg, "encoder_attention_heads", 12)
        ),
        activation_fn=_get_mapping_value(model_cfg, "activation_fn", "gelu"),
        layer_norm_first=bool(_get_mapping_value(model_cfg, "layer_norm_first", False)),
        dropout=float(_get_mapping_value(model_cfg, "dropout", 0.1)),
        attention_dropout=float(_get_mapping_value(model_cfg, "attention_dropout", 0.1)),
        activation_dropout=float(_get_mapping_value(model_cfg, "activation_dropout", 0.0)),
        encoder_layerdrop=float(_get_mapping_value(model_cfg, "encoder_layerdrop", 0.0)),
        dropout_input=float(_get_mapping_value(model_cfg, "dropout_input", 0.0)),
        dropout_features=float(_get_mapping_value(model_cfg, "dropout_features", 0.0)),
        final_dim=int(final_proj_weight.shape[0]),
        label_embs_num=int(label_embs.shape[0]) if label_embs is not None else 0,
        conv_pos=int(_get_mapping_value(model_cfg, "conv_pos", 128)),
        conv_pos_groups=int(_get_mapping_value(model_cfg, "conv_pos_groups", 16)),
        required_seq_len_multiple=int(
            _get_mapping_value(model_cfg, "required_seq_len_multiple", 2)
        ),
        mask_prob=float(_get_mapping_value(model_cfg, "mask_prob", 0.8)),
        mask_length=int(_get_mapping_value(model_cfg, "mask_length", 10)),
        mask_channel_prob=float(_get_mapping_value(model_cfg, "mask_channel_prob", 0.0)),
    )


def pad_to_multiple(x, multiple, dim=-1, value=0):
    if x is None:
        return None, 0
    size = x.size(dim)
    if size % multiple == 0:
        return x, 0
    remainder = math.ceil(size / multiple) * multiple - size
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(x, (*pad_offset, 0, remainder), value=value), remainder


def index_put(tensor, indices, value):
    tensor[indices] = value
    return tensor


class SamePad(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        if self.remove > 0:
            x = x[:, :, : -self.remove]
        return x


class Fp32GroupNorm(nn.GroupNorm):
    def forward(self, input):
        output = F.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class ConvFeatureExtractionModel(nn.Module):
    def __init__(self, conv_layers, conv_bias=False):
        super().__init__()
        in_dim = 1
        blocks = []
        for index, (dim, kernel, stride) in enumerate(conv_layers):
            conv = nn.Conv1d(in_dim, dim, kernel, stride=stride, bias=conv_bias)
            if index == 0:
                blocks.append(
                    nn.Sequential(
                        conv,
                        nn.Dropout(p=0.0),
                        Fp32GroupNorm(dim, dim, affine=True),
                        nn.GELU(),
                    )
                )
            else:
                blocks.append(nn.Sequential(conv, nn.Dropout(p=0.0), nn.GELU()))
            in_dim = dim
        self.conv_layers = nn.ModuleList(blocks)

    def forward(self, x):
        x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x


def make_conv_pos(embed_dim, kernel_size, groups):
    pos_conv = nn.Conv1d(
        embed_dim,
        embed_dim,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        groups=groups,
    )
    pos_conv = nn.utils.weight_norm(pos_conv, name="weight", dim=2)
    return nn.Sequential(pos_conv, SamePad(kernel_size), nn.GELU())


class FairseqMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("HuBERT encoder_embed_dim must be divisible by heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=False,
        attn_mask=None,
    ):
        return F.multi_head_attention_forward(
            query,
            key,
            value,
            self.embed_dim,
            self.num_heads,
            query.new_empty((0,)),
            torch.cat((self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
            None,
            None,
            False,
            self.dropout,
            self.out_proj.weight,
            self.out_proj.bias,
            self.training,
            key_padding_mask,
            need_weights,
            attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
        )


class TransformerSentenceEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm_first = config.layer_norm_first
        self.activation_fn = _activation_fn(config.activation_fn)
        self.self_attn = FairseqMultiheadAttention(
            config.encoder_embed_dim,
            config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.activation_dropout)
        self.dropout3 = nn.Dropout(config.dropout)
        self.self_attn_layer_norm = nn.LayerNorm(config.encoder_embed_dim)
        self.fc1 = nn.Linear(config.encoder_embed_dim, config.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_embed_dim, config.encoder_embed_dim)
        self.final_layer_norm = nn.LayerNorm(config.encoder_embed_dim)

    def forward(
        self,
        x,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_weights=False,
        att_args=None,
    ):
        residual = x
        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                attn_mask=self_attn_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x
            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            layer_result = x
            x = self.dropout3(x)
            x = residual + x
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=False,
            )
            x = self.dropout1(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)
            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            layer_result = x
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)
        return x, (attn, layer_result)


class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config.dropout
        self.embedding_dim = config.encoder_embed_dim
        self.required_seq_len_multiple = config.required_seq_len_multiple
        self.pos_conv = make_conv_pos(
            config.encoder_embed_dim,
            config.conv_pos,
            config.conv_pos_groups,
        )
        self.layers = nn.ModuleList(
            [TransformerSentenceEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm_first = config.layer_norm_first
        self.layer_norm = nn.LayerNorm(config.encoder_embed_dim)
        self.layerdrop = config.encoder_layerdrop

    def forward(self, x, padding_mask=None, layer=None):
        x, layer_results = self.extract_features(x, padding_mask, layer)
        if self.layer_norm_first and layer is None:
            x = self.layer_norm(x)
        return x, layer_results

    def extract_features(self, x, padding_mask=None, tgt_layer=None, min_layer=0):
        if padding_mask is not None:
            x = index_put(x, padding_mask, 0)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x = x + x_conv

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x, pad_length = pad_to_multiple(
            x, self.required_seq_len_multiple, dim=-2, value=0
        )
        if pad_length > 0 and padding_mask is None:
            padding_mask = x.new_zeros((x.size(0), x.size(1)), dtype=torch.bool)
            padding_mask[:, -pad_length:] = True
        else:
            padding_mask, _ = pad_to_multiple(
                padding_mask, self.required_seq_len_multiple, dim=-1, value=True
            )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)

        layer_results = []
        result = None
        for index, layer in enumerate(self.layers):
            if not self.training or self.layerdrop == 0.0 or torch.rand(()) > self.layerdrop:
                x, (attn, layer_result) = layer(
                    x, self_attn_padding_mask=padding_mask, need_weights=False
                )
                if index >= min_layer:
                    layer_results.append((x, attn, layer_result))
            if index == tgt_layer:
                result = x
                break

        if result is not None:
            x = result

        x = x.transpose(0, 1)
        if pad_length > 0:
            x = x[:, :-pad_length]

            def undo_pad(hidden, attn, layer_result):
                return (
                    hidden[:-pad_length],
                    attn[:-pad_length] if attn is not None else attn,
                    layer_result[:-pad_length],
                )

            layer_results = [undo_pad(*item) for item in layer_results]
        return x, layer_results


class HubertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed = config.conv_layers[-1][0]
        self.feature_extractor = ConvFeatureExtractionModel(
            config.conv_layers,
            conv_bias=config.conv_bias,
        )
        self.post_extract_proj = (
            nn.Linear(self.embed, config.encoder_embed_dim)
            if self.embed != config.encoder_embed_dim
            else None
        )
        self.mask_prob = config.mask_prob
        self.mask_length = config.mask_length
        self.mask_channel_prob = config.mask_channel_prob
        self.dropout_input = nn.Dropout(config.dropout_input)
        self.dropout_features = nn.Dropout(config.dropout_features)
        self.mask_emb = nn.Parameter(torch.empty(config.encoder_embed_dim))
        self.encoder = TransformerEncoder(config)
        self.layer_norm = nn.LayerNorm(self.embed)
        self.final_proj = nn.Linear(config.encoder_embed_dim, config.final_dim)
        if config.label_embs_num > 0:
            self.label_embs_concat = nn.Parameter(
                torch.empty(config.label_embs_num, config.final_dim)
            )
        self.task_normalize = False

    def forward_features(self, source):
        return self.feature_extractor(source)

    def forward_padding_mask(self, features, padding_mask):
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        return padding_mask.all(-1)

    def forward(
        self,
        source,
        target_list=None,
        padding_mask=None,
        mask=False,
        features_only=False,
        output_layer=None,
    ):
        if target_list is not None:
            raise NotImplementedError("HuBERT target_list forward is not implemented")
        if mask:
            raise NotImplementedError("HuBERT mask=True inference is not implemented")
        if not features_only:
            raise NotImplementedError("HuBERT contrastive pretraining forward is not implemented")

        features = self.forward_features(source)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if padding_mask is not None:
            padding_mask = self.forward_padding_mask(features, padding_mask)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
        x, _ = self.encoder(
            features,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )
        return {"x": x, "padding_mask": padding_mask, "features": features}

    def extract_features(
        self,
        source,
        padding_mask=None,
        mask=False,
        ret_conv=False,
        output_layer=None,
    ):
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]


def build_hubert_model_from_checkpoint(cfg, state_dict):
    config = hubert_config_from_checkpoint(cfg, state_dict)
    model = HubertModel(config)
    model.load_state_dict(state_dict, strict=True)
    task_cfg = _get_mapping_value(cfg, "task", {}) if cfg is not None else {}
    model.task_normalize = bool(_get_mapping_value(task_cfg, "normalize", False))
    return model
