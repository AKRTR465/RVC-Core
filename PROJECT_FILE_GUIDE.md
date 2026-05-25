# 项目文件说明

本文档基于当前仓库代码结构整理目录职责，重点解释 `infer/` 目录内部的分层，以及配置、数据、训练产物之间的对应关系。

这套仓库的命名有一些历史包袱，先记住两点：

- `infer/` 不是“只有推理”，而是“训练与推理共用代码包 + 多个入口脚本”
- `infer/index/` 不是实时推理入口，而是“索引构建和权重辅助工具”

## 一句话看懂主链路

1. 在 `configs/base.yaml` 和 `configs/<task>.yaml` 里定义任务
2. `configs/project_config.py` 解析继承链、快照、`--hparams` 覆盖和 `variants`
3. 原始训练音频放到 `data/<name>/dataset/`
4. `infer/modules/train/preprocess.py` 把音频切片、归一化，并生成 16k 副本
5. `infer/modules/train/extract_f0_print.py` 提取 F0，`infer/modules/train/extract_feature_print.py` 提取 HuBERT 特征
6. 训练器固定从 `data/<name>/preprocess_data/filelist.txt` 读取训练清单
7. `infer/modules/train/train.py` 训练模型，日志和大 checkpoint 放到 `ckpt/<name>/train/`
8. `infer/lib/train/process_ckpt.py` 把训练权重导出成小模型到 `ckpt/<name>/export/`
9. `infer/index/train-index.py` 或 `infer/index/train-index-v2.py` 基于特征构建索引到 `ckpt/<name>/index/`
10. 离线变声运行时主要在 `infer/modules/vc/`

## 根目录

- `README.md`：项目总览、配置方式、常用命令示例
- `PROJECT_FILE_GUIDE.md`：本说明文件
- `requirements.txt`：pip 依赖列表
- `pyproject.toml`：项目元数据与 Poetry 依赖定义
- `CONTRIBUTING.md`：贡献说明
- `LICENSE`、`MIT协议暨相关引用库协议`：许可证文件
- `.env`：本地环境变量文件
- `configs/`：项目配置入口和配置解析器
- `data/`：原始数据和预处理产物
- `ckpt/`：训练、导出、索引和配置快照
- `pretrain/`：HuBERT、RMVPE、预训练权重
- `infer/`：训练、推理、索引构建和相关公共库

## 配置系统

### `configs/`

- `configs/project_config.py`：唯一配置解析器。负责解析 `base_config` 继承链、读取 `work_dir/config.yaml` 快照、应用 `--hparams` 标量覆盖、按 `selectors.version/sample_rate` 选择 `variants`，并完成 runtime 自动补全。
- `configs/base.yaml`：共享默认配置，主要放 `preprocess`、`runtime`、`infer`、`train`、`data`、`model` 的通用默认值。
- `configs/<task>.yaml`：任务配置入口。通常只需要覆盖 `name`、顶层路径字段、`selectors`、`model` 和 `variants`。

### 解析顺序

当前配置链路固定为：

1. `base_config` 继承链
2. 当前 task YAML
3. `work_dir/config.yaml` 快照，除非显式传 `--reset`
4. `--hparams` 标量 dotted overrides
5. `variants[version][sample_rate]` patch
6. runtime 自动推导

### 关键字段心智模型

- `name`：项目名，也是默认输出文件名和默认目录派生的基准
- `work_dir`：实验根目录，通常就是 `ckpt/<name>`
- `selectors.version`：选择 `v1` 或 `v2`
- `selectors.sample_rate`：选择 `32k`、`40k`、`48k`
- `selectors.if_f0`：控制训练/推理是否走带 F0 分支
- `variants`：只按 `version/sample_rate` 选择补丁，不按 `if_f0` 选分支
- `runtime.device`、`runtime.is_half`、`runtime.n_cpu`：支持 `auto`
- `train.fp16_run`：支持 `auto`

### 顶层路径字段

当前支持直接写在 task YAML 顶层的路径字段有：

- `work_dir`
- `data_root`
- `ckpt_root`
- `pretrain_root`
- `dataset_dir`
- `preprocess_dir`
- `train_dir`
- `export_dir`
- `index_dir`
- `final_model_name`
- `final_index_name`

常规项目通常只需要显式保留：

- `work_dir`
- `data_root`
- `ckpt_root`
- `pretrain_root`

其余目录和输出文件名一般交给 resolver 自动推导即可。

### 配置快照

- `ckpt/<name>/config.yaml`：训练入口写入的可重放快照
- 快照内容来自 `configs/project_config.py` 中的 `replayable_config`
- 只有训练入口会生成或刷新这个文件
- `preprocess.py`、索引脚本和纯解析模式只会读取它，不会写它

### 已移除的旧字段

以下旧入口已经不再支持，出现会直接报错：

- `version`，改为 `selectors.version`
- `sample_rate`，改为 `selectors.sample_rate`
- `if_f0`，改为 `selectors.if_f0`
- `experiment_dir` 或 `ckpt_dir`，改为 `work_dir`
- `preprocess_per`，改为 `preprocess.per`
- `noparallel`，改为 `preprocess.noparallel`
- `train_common`、`data_common`、`model_common`，分别并入 `train`、`data`、`model`
- `paths`，改为顶层路径字段

## 数据与产物布局

### `data/`

- `data/<name>/dataset/`：原始训练音频
- `data/<name>/preprocess_data/`：预处理阶段的中间产物目录

### `data/<name>/preprocess_data/` 常见内容

- `0_gt_wavs/`：按训练采样率保存的切片音频
- `1_16k_wavs/`：16k 重采样后的切片音频，供 HuBERT 和 F0 提取使用
- `2a_f0/`：粗量化后的 F0 `npy`
- `2b-f0nsf/`：连续 F0 `npy`
- `3_feature256/`：v1 使用的 256 维特征
- `3_feature768/`：v2 使用的 768 维特征
- `preprocess.log`：预处理日志
- `extract_f0_feature.log`：F0 / 特征提取日志
- `filelist.txt`：训练清单文件的约定路径

注意：

- 当前仓库里的 `preprocess.py` 只负责切片、归一化和 16k 重采样，不会直接生成 `filelist.txt`
- 但训练器和配置解析器会固定把训练清单路径解析为 `preprocess_dir/filelist.txt`

### `ckpt/`

- `ckpt/<name>/`：单个实验的工作目录，也就是默认的 `work_dir`
- `ckpt/<name>/train/`：训练日志、TensorBoard、`G_*.pth`、`D_*.pth`
- `ckpt/<name>/export/`：导出的轻量模型，通常是 `<name>.pth`
- `ckpt/<name>/index/`：FAISS 索引、中间特征聚合文件
- `ckpt/<name>/config.yaml`：训练阶段冻结出来的任务快照

### `ckpt/<name>/index/` 常见内容

- `big_src_feature.npy`：拼接后的大特征矩阵
- `trained_<name>.index`：训练完成但还未加库的索引
- `<name>.index`：最终可用于检索的索引

### `pretrain/`

- `pretrain/hubert/`：HuBERT 权重，默认文件是 `hubert_base.pt`
- `pretrain/rmvpe/`：RMVPE 权重，默认文件是 `rmvpe.pt`
- `pretrain/pretrained/`：v1 训练常用的预训练 G/D 权重
- `pretrain/pretrained_v2/`：v2 训练常用的预训练 G/D 权重

当前仓库不再提供内置下载脚本，运行前请手动把这些资源放到对应目录。

## `infer/` 的职责

`infer/` 是当前仓库的主代码目录。虽然名字叫 infer，但它同时承载了：

- 数据预处理入口
- F0 / 特征提取脚本
- 训练入口
- 离线推理运行时
- 索引构建脚本
- 训练与推理共享的底层模型和工具库

### `infer/modules/train/`

- `infer/modules/train/preprocess.py`
  训练数据预处理入口。负责静音切片、高通滤波、归一化、按目标采样率写出 `0_gt_wavs`，并额外写出 `1_16k_wavs`。支持三种调用方式：配置模式、手工路径模式、旧版位置参数模式。

- `infer/modules/train/train.py`
  主训练入口。启动分布式训练进程，构造数据集和判别器/生成器，尝试从 `train/` 下最新的 `G_*.pth`、`D_*.pth` 恢复；如果没有恢复点，则回退到 `train.pretrainG`、`train.pretrainD`。训练过程中按周期保存大 checkpoint，并通过 `process_ckpt.savee()` 导出小模型到 `export/`。

- `infer/modules/train/extract_feature_print.py`
  HuBERT 特征提取脚本。读取 `1_16k_wavs/`，根据 `version` 输出到 `3_feature256/` 或 `3_feature768/`。这是一个偏底层的批处理 worker，更像“提特征子进程”而不是顶层配置入口。

- `infer/modules/train/extract_f0_print.py`
  统一 F0 提取脚本。读取 `1_16k_wavs/`，输出 `2a_f0/` 和 `2b-f0nsf/`。同时支持 `pm`、`harvest`、`dio`、`rmvpe`，并把 RMVPE 的自动 CUDA/half 模式、GPU 指定模式、外部分片模式统一进一个入口。

### `infer/modules/vc/`

这一层是离线变声运行时，不负责训练。

- `infer/modules/vc/modules.py`
  `VC` 类所在文件。负责按 `sid` 从 `ckpt_root` 下定位模型、加载小模型、推断是否带 F0、初始化 `Pipeline`，并提供 `vc_single()`、`vc_multi()` 两个离线推理入口。

- `infer/modules/vc/pipeline.py`
  变声主流水线。负责：
  1. 音频预处理和分段
  2. HuBERT 特征提取
  3. 可选 FAISS 检索特征混合
  4. F0 提取和升降调
  5. 调用生成器推理
  6. RMS 混合和重采样

  当前这份实现里，推理阶段内置的 F0 方法分支是 `pm`、`harvest`、`crepe`、`rmvpe`。

- `infer/modules/vc/utils.py`
  推理辅助函数。负责在 `ckpt_root` 下查找模型和索引文件，并加载 HuBERT 模型。

### `infer/index/`

这一层负责索引构建和少量权重辅助处理，不负责实时推理。

- `infer/index/train-index.py`
  v1 / 256 维特征索引构建脚本。支持配置模式和手工路径模式。会把 `feature_dir` 下所有 `npy` 拼接成 `big_src_feature.npy`，训练 IVF Flat FAISS 索引，并写出最终 `.index`。

- `infer/index/train-index-v2.py`
  v2 / 768 维特征索引构建脚本。面向 768 维特征；样本很多时会尝试用 `MiniBatchKMeans` 压到 10000 个中心；之后训练索引并分批 `index.add()`。

- `infer/index/trans_weights.py`
  一个非常临时的权重转换脚本，里面仍然写着硬编码本地路径，作用是把某个训练 checkpoint 的权重转成 half 并另存。它不是当前项目主流程的一部分，更像历史调试/实验脚本。

### `infer/lib/`

这一层是训练和推理共享的底层库。

- `infer/lib/audio.py`
  音频 I/O 辅助。通过 `ffmpeg` 解码任意音频到单声道 float32，并提供 `wav2()` 做格式转码。

- `infer/lib/slicer2.py`
  静音切片器。`preprocess.py` 依赖它把长音频切成训练片段。

- `infer/lib/rmvpe.py`
  RMVPE 模型定义和推理实现。被 F0 提取脚本和 VC 推理管线共用。

### `infer/lib/infer_pack/`

这是 RVC 模型本体和神经网络积木所在目录。

- `infer/lib/infer_pack/models.py`
  生成器、判别器、`SynthesizerTrnMs256NSFsid`、`SynthesizerTrnMs768NSFsid` 等核心模型定义。

- `infer/lib/infer_pack/attentions.py`
  编码器/解码器注意力模块、多头注意力、FFN。

- `infer/lib/infer_pack/modules.py`
  各类共享网络模块，如 `WN`、`ResBlock`、flow coupling 等。

- `infer/lib/infer_pack/commons.py`
  通用 tensor 工具、segment 切片、mask、梯度裁剪等基础函数。

- `infer/lib/infer_pack/transforms.py`
  flow 相关的 spline 变换实现。

### `infer/lib/train/`

这是训练链路的公共库。

- `infer/lib/train/data_utils.py`
  训练数据集、`collate_fn`、bucket sampler。`train.py` 通过这里读取 `filelist.txt`。

- `infer/lib/train/losses.py`
  判别器损失、生成器损失、feature loss、KL loss。

- `infer/lib/train/mel_processing.py`
  频谱和 mel 相关运算。

- `infer/lib/train/utils.py`
  训练参数解析、`HParams` 组装、checkpoint 读写、TensorBoard 可视化、训练日志、配置快照落盘等工具。

- `infer/lib/train/process_ckpt.py`
  训练权重导出和模型后处理工具。包括：
  1. `savee()`：导出小模型到 `export/`
  2. `extract_small_model()`：从训练权重提取轻量模型
  3. `merge()`：合并两个模型
  4. `change_info()`、`show_info()`：修改或查看小模型元信息

## 当前目录里几个容易混淆的点

- `infer/` 是主代码目录，不只是推理
- `infer/modules/train/` 才是训练侧入口
- `infer/modules/vc/` 才是离线推理运行时
- `infer/index/` 不负责实时推理，主要负责索引和权重辅助处理
- `ckpt/<name>/` 是实验根目录，`train/`、`export/`、`index/` 都是它的子目录
- `preprocess.py` 只完成切片和重采样，F0、特征、索引都是后续独立步骤
- 训练清单路径约定存在于配置和训练器中，但当前仓库里的 `preprocess.py` 不会直接生成 `filelist.txt`

## 推荐阅读顺序

如果要快速理解项目，建议按下面顺序看：

1. `README.md`
2. `configs/base.yaml`
3. `configs/<task>.yaml`
4. `configs/project_config.py`
5. `infer/modules/train/preprocess.py`
6. `infer/modules/train/train.py`
7. `infer/lib/train/utils.py`
8. `infer/index/train-index.py` 或 `infer/index/train-index-v2.py`
9. `infer/modules/vc/modules.py`
10. `infer/modules/vc/pipeline.py`
