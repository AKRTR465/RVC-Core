# RVC Core

这个仓库保留 RVC 的核心训练、预处理、索引构建、离线推理和模型处理链路，入口已经统一迁移到 `src/`。它不是原 WebUI 的完整界面工程；训练流程以命令行和 YAML 配置为主。

参考原项目 `Retrieval-based-Voice-Conversion-WebUI` 的训练思路，本仓库当前训练链路是：

1. 准备环境和预训练资源。
2. 准备项目 YAML。
3. 放入已经切好的训练音频。
4. 运行预处理流水线：音频规范化/重采样、F0、HuBERT 特征、`filelist.txt`。
5. 启动训练。
6. 构建检索索引。

## 目录约定

默认目录由 `configs/project_config.py` 根据 `name/data_root/work_dir` 推导。

```text
pretrain/
  hubert/hubert_base.pt
  rmvpe/rmvpe.pt
  pretrained/          # 可选：v1 预训练 G/D
  pretrained_v2/       # 可选：v2 预训练 G/D

data/<name>/
  dataset/             # 用户已切好的训练音频
  preprocess_data/
    0_gt_wavs/         # 目标采样率训练 wav
    1_16k_wavs/        # HuBERT/F0 输入 wav
    2a_f0/             # coarse F0
    2b-f0nsf/          # continuous F0
    3_feature256/      # v1 HuBERT 特征
    3_feature768/      # v2 HuBERT 特征
    preprocess_manifest.jsonl
    filelist.txt       # 完整样本总表
    train_filelist.txt # 训练 filelist
    val_filelist.txt   # 验证 filelist

ckpt/<name>/
  train/               # G/D checkpoint、日志、TensorBoard
  export/              # 导出的小模型
  index/               # FAISS index 和 big_src_feature.npy
  config.yaml          # 训练入口生成的可重放快照
```

示例配置是 `configs/mute.yaml`，默认项目名为 `mute`。

## 环境准备

建议使用 Python 3.10 附近的独立环境。先安装适配你显卡的 PyTorch，再安装项目依赖。

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

回归测试建议直接使用 `RVC` conda 环境里的解释器：
```bash
F:\Anaconda3\envs\RVC\python.exe -m unittest discover -s tests
```

Windows 上还需要能调用 `ffmpeg`/`ffprobe`。如果没有全局安装，可以把可执行文件放到项目根目录或加入 `PATH`。

最少需要准备：

- `pretrain/hubert/hubert_base.pt`
- `pretrain/rmvpe/rmvpe.pt`

如果要从官方预训练 G/D 开始训练，还需要下载对应版本/采样率的 G、D 权重，并在配置或命令行里填入：

- `train.pretrainG`
- `train.pretrainD`

## 配置方式

配置入口是两层 YAML：

- `configs/base.yaml`：共享默认项。
- `configs/<task>.yaml`：项目配置。

解析顺序：

1. `base_config` 继承链。
2. task YAML。
3. `work_dir/config.yaml`，除非传 `--reset`。
4. `--hparams` 标量 dotted overrides。
5. `selectors.version/sample_rate` 对应的 `variants` patch。
6. runtime auto 补全。

关键字段：

```yaml
base_config: base.yaml
name: my_voice

work_dir: ckpt/my_voice
data_root: data
ckpt_root: ckpt
pretrain_root: pretrain

selectors:
  version: v2          # v1 或 v2
  sample_rate: 48k     # v1: 32k/40k/48k；v2: 32k/48k
  if_f0: 1             # 1: 使用 F0；0: 不使用 F0

preprocess:
  validation_split: 0.1
  validation_seed: 1234

runtime:
  device: auto         # auto/cpu/cuda/cuda:0
  is_half: auto

train:
  batch_size: 4
  epochs: 20000
  save_every_epoch: 10
  pretrainG: ""
  pretrainD: ""

model:
  spk_embed_dim: 1     # 单说话人通常为 1；多说话人按 speaker id 数量设置
```

`dataset_dir`、`preprocess_dir`、`train_dir`、`export_dir`、`index_dir` 默认会自动推导。只有偏离默认目录布局时才需要显式写。

命令行临时覆盖用 `--hparams`：

```bash
python -m src.train --config configs/my_voice.yaml --hparams train.batch_size=2,train.epochs=300
```

`--hparams` 只支持标量。列表、字典、复杂模型结构请直接改 YAML。

## 训练完整流程

下面以 `configs/mute.yaml` 为例。你可以复制这份 YAML 改成自己的项目名和目录。

### 1. 放入训练音频

把已经切好的音频放到：

```text
data/mute/dataset/
```

建议：

- 使用干净的人声干声，尽量少混响、少伴奏、少噪声。
- 预处理入口不再做自动切片；请先自行切成适合训练的片段。
- 数据量建议至少 10 分钟；质量比时长更重要。
- 可以放 `.wav`、`.mp3` 等 `ffmpeg` 能读取的格式。
- 单说话人直接把音频放在 `dataset/` 下，sid 固定为 `0`。
- 多说话人使用 DDSP-SVC 风格的一级数字目录，例如 `dataset/1/`、`dataset/2/`；写入 filelist 时会转换为 0-based sid。

### 2. 运行预处理流水线

训练前只需要运行聚合入口。默认会依次执行音频规范化/重采样、F0、HuBERT 特征提取，并生成 `filelist.txt`、`train_filelist.txt` 和 `val_filelist.txt`：

```bash
python -m src.preprocess --config configs/mute.yaml
```

等价的显式入口是：

```bash
python -m src.preprocess.pipeline --config configs/mute.yaml
```

常用覆盖：

```bash
python -m src.preprocess --config configs/mute.yaml --hparams preprocess.noparallel=true,runtime.n_cpu=1
```

RMVPE 默认只使用 1 个 F0 worker；也可以临时切换 F0 方法：

```bash
python -m src.preprocess --config configs/mute.yaml --f0method harvest
```

只补跑某个阶段时使用 `--stages`，例如只重建 filelist：

```bash
python -m src.preprocess --config configs/mute.yaml --stages filelist
```

输出：

```text
data/mute/preprocess_data/0_gt_wavs/
data/mute/preprocess_data/1_16k_wavs/
data/mute/preprocess_data/2a_f0/
data/mute/preprocess_data/2b-f0nsf/
data/mute/preprocess_data/3_feature256/   # v1
data/mute/preprocess_data/3_feature768/   # v2
data/mute/preprocess_data/preprocess_manifest.jsonl
data/mute/preprocess_data/filelist.txt
data/mute/preprocess_data/train_filelist.txt
data/mute/preprocess_data/val_filelist.txt
data/mute/preprocess_data/preprocess.log
data/mute/preprocess_data/extract_f0_feature.log
```

训练阶段只消费 `train_filelist.txt` 和 `val_filelist.txt`；`filelist.txt` 保留为完整总表，便于调试和重建切分。

如果 `selectors.if_f0: 0`，聚合入口会跳过 F0 阶段，`filelist.txt` 会写 3 列：

```text
wav_path|feature_path|speaker_id
```

`selectors.if_f0: 1` 时写 5 列：

```text
wav_path|feature_path|coarse_f0_path|nsf_f0_path|speaker_id
```

### 3. 启动训练

```bash
python -m src.train --config configs/mute.yaml
```

常用训练参数：

```bash
python -m src.train --config configs/mute.yaml -g 0 -bs 4 -te 300 -se 10
```

含义：

- `-g/--gpus`：GPU 列表，用 `-` 分隔，例如 `0`、`0-1`。
- `-bs/--batch_size`：单进程 batch size。
- `-te/--total_epoch`：总 epoch，等价覆盖 `train.epochs`。
- `-se/--save_every_epoch`：每多少个 epoch 保存一次。
- `-pg/--pretrainG`：预训练生成器路径。
- `-pd/--pretrainD`：预训练判别器路径。
- `-sw/--save_every_weights`：保存 checkpoint 时同步导出小模型，`1` 或 `0`。
- `-l/--if_latest`：只保留 latest checkpoint，`1` 或 `0`。
- `-c/--if_cache_data_in_gpu`：把训练数据缓存进 GPU，显存足够时才用。

也可以用 YAML 覆盖：

```bash
python -m src.train --config configs/mute.yaml --hparams train.batch_size=2,train.epochs=300,train.save_every_epoch=5
```

使用预训练 G/D：

```bash
python -m src.train --config configs/mute.yaml -pg pretrain/pretrained_v2/f0G48k.pth -pd pretrain/pretrained_v2/f0D48k.pth
```

实际文件名以你下载的预训练权重为准。`if_f0=0` 时应使用不带 F0 的预训练权重。

训练输出：

```text
ckpt/mute/train/
  train.log
  events.out.tfevents...
  G_*.pth
  D_*.pth

ckpt/mute/export/
  mute.pth

ckpt/mute/config.yaml
```

TensorBoard 查看方式：

```bash
tensorboard --logdir ckpt/mute/train --port 6006
```

启动后在浏览器打开 `http://localhost:6006/`。训练标量和 mel 图都会写到 `ckpt/<name>/train/events.out.tfevents...`。

训练会自动尝试从 `ckpt/mute/train/G_*.pth` 和 `D_*.pth` resume。想忽略已有 snapshot 重新解析配置，加 `--reset`：

```bash
python -m src.train --config configs/mute.yaml --reset
```

### 4. 构建检索索引

训练完成后构建 FAISS 索引：

```bash
python -m src.index --config configs/mute.yaml
```

输出：

```text
ckpt/mute/index/
  mute.index
  big_src_feature.npy
```

推理检索需要 `.index` 和同目录下的 `big_src_feature.npy` 同时存在。

## 手工模式

配置模式是推荐入口。手工模式保留给旧脚本或临时调试。

```bash
python -m src.preprocess.audio -i data/mute/dataset -o data/mute/preprocess_data_manual -sr 48000 -n 1 --noparallel
python -m src.preprocess.f0 --exp-dir data/mute/preprocess_data --workers 1 --f0method rmvpe
python -m src.preprocess.features --exp-dir data/mute/preprocess_data --version v1 --device auto
python -m src.index -i data/mute/preprocess_data/3_feature256 -o ckpt/mute/index/mute.index --feature-dim 256
```

## 常见问题

### `No valid audio files found in training filelist`

检查 `data/<name>/preprocess_data/train_filelist.txt` 是否存在、是否有行，以及每行路径是否真实存在。

### `Expected 5 columns` 或 `Expected 3 columns`

`selectors.if_f0` 和 filelist 格式不匹配：

- `if_f0=1` 需要 5 列。
- `if_f0=0` 需要 3 列。

### HuBERT 或 RMVPE 找不到

检查：

```text
pretrain/hubert/hubert_base.pt
pretrain/rmvpe/rmvpe.pt
```

也可以在 YAML 中覆盖：

```yaml
pretrain_root: pretrain
```

### 显存不够

优先降低：

- `train.batch_size`
- `runtime.is_half` 保持 `auto`
- F0 提取时 `--workers 1`

不要轻易打开 `--if_cache_data_in_gpu`，除非显存很宽裕。

### 多卡训练

```bash
python -m src.train --config configs/mute.yaml -g 0-1 -bs 4
```

`-bs` 是传给每个训练进程的 batch size。实际全局 batch 会随 GPU 数量增加。

### v1/v2 和采样率怎么选

- `v1` 支持 `32k/40k/48k`，特征维度 256。
- `v2` 支持 `32k/48k`，特征维度 768。
- 改版本或采样率时，需要重新跑预处理流水线和索引。

## 维护说明

- 新入口使用 `python -m src...`。
- 旧 `infer/` 源树不再作为入口。
- 模型 canonical 实现在 `src/models/`。
- 索引构建统一走 `python -m src.index`，不再保留 `build_v1/build_v2` 版本化入口。
- 文档、源码和测试统一使用 UTF-8 无 BOM。
