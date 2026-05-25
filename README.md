# RVC Core

这个仓库保留了 RVC 的训练、索引构建、离线推理和模型处理链路，当前目录布局已经统一为：

- `pretrain/`：HuBERT、RMVPE、预训练权重
- `data/<name>/dataset/`：原始训练音频
- `data/<name>/preprocess_data/`：预处理产物
- `ckpt/<name>/train|export|index/`：训练日志、导出模型和索引
- `ckpt/<name>/config.yaml`：训练入口生成的可重放任务快照
- `configs/base.yaml` + `configs/<task>.yaml`：SOAP 风格配置入口
- `src/`：训练、预处理、索引构建、离线推理和共享模型组件的新入口

## 环境准备

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## 配置方式

用户配置入口已经回切为两层 YAML：

- `configs/base.yaml`：共享默认项
- `configs/<task>.yaml`：任务配置，包含 `base_config`、上层根字段、`selectors` 和 `variants`

解析顺序固定为：

1. `base_config` 继承链
2. task YAML
3. `work_dir/config.yaml`，除非传 `--reset`
4. `--hparams` 标量 dotted overrides
5. `selectors.version/sample_rate` 对应的 `variants` patch
6. runtime auto 补全

其中：

- `selectors.version` 控制 `v1|v2`
- `selectors.sample_rate` 控制 `32k|40k|48k`
- `selectors.if_f0` 控制 `0|1`
- `runtime.is_half` 支持 `auto|true|false`
- `train.fp16_run` 支持 `auto|true|false`
- `--hparams` 只支持标量覆盖；列表和字典请直接回 YAML

推荐写法：

- 常规项目只显式写 `work_dir`、`data_root`、`ckpt_root`、`pretrain_root`
- `dataset_dir`、`preprocess_dir`、`train_dir`、`export_dir`、`index_dir` 默认由 resolver 自动推导
- `final_model_name`、`final_index_name` 默认由 `name` 自动推导
- 只有偏离默认目录布局或默认输出文件名时，才显式覆盖这些派生字段

### 旧字段已移除

以下旧配置入口已经不再支持，出现时会直接报错：

- `version -> selectors.version`
- `sample_rate -> selectors.sample_rate`
- `if_f0 -> selectors.if_f0`
- `experiment_dir/ckpt_dir -> work_dir`
- `preprocess_per -> preprocess.per`
- `noparallel -> preprocess.noparallel`
- `train_common/data_common/model_common -> train/data/model`
- `paths -> 顶层路径字段`

## 常用命令

预处理：

```bash
python -m src.preprocess --config configs/mute.yaml
```

带覆盖的预处理：

```bash
python -m src.preprocess --config configs/mute.yaml --hparams dataset_dir=data/mute/dataset,preprocess_dir=data/mute/preprocess_data_smoke,preprocess.noparallel=true,runtime.n_cpu=1
```

训练：

```bash
python -m src.train --config configs/mute.yaml
```

切换 selector：

```bash
python -m src.train --config configs/mute.yaml --hparams selectors.version=v2,selectors.sample_rate=48k
```

忽略已有快照重新解析：

```bash
python -m src.train --config configs/mute.yaml --reset
```

构建索引：

```bash
# v1 / 256-dim
python -m src.index.build_v1 --config configs/mute.yaml

# v2 / 768-dim
python -m src.index.build_v2 --config configs/mute.yaml --hparams selectors.version=v2,selectors.sample_rate=48k
```

手工路径模式仍保留：

```bash
python -m src.preprocess -i data/mute/dataset -o data/mute/preprocess_data_manual -sr 48000 -n 1 --per 3.7 --noparallel
python -m src.index.build_v1 -i data/mute/preprocess_data/3_feature256 -o ckpt/mute/index/mute.index
```

## 输出位置

- 最终模型：`ckpt/<name>/export/<name>.pth`
- 最终索引：`ckpt/<name>/index/<name>.index`
- 训练快照：`ckpt/<name>/config.yaml`

只有训练入口会生成或刷新 `ckpt/<name>/config.yaml`；preprocess、index 和纯解析模式只读不写。

## 预训练资源准备

请手动准备以下运行时资源：

- `pretrain/hubert/hubert_base.pt`
- `pretrain/rmvpe/rmvpe.pt`

如需使用预训练生成器和判别器权重，请按版本手动放入：

- `pretrain/pretrained/`
- `pretrain/pretrained_v2/`
