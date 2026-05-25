# RVC Rebuild 文件职责边界

本仓库的 Python 业务入口已经统一迁移到 `src/`。旧的 `infer/` 路径不再作为入口或兼容层使用；重构后的目标是保持原始功能严格等价，同时让每个模块只承担清晰的职责。

## 顶层目录

- `configs/`：项目配置解析和 YAML 默认配置。`configs/project_config.py` 负责配置继承、`--hparams` 覆盖、runtime 自动补全和路径派生。
- `src/`：训练、预处理、索引构建、离线推理和共享模型组件的新代码入口。
- `data/`：训练原始音频与预处理产物。
- `ckpt/`：训练日志、大 checkpoint、导出小模型和 FAISS 索引。
- `pretrain/`：HuBERT、RMVPE 和预训练 G/D 权重。

## `src/index/`

索引模块只负责从 HuBERT 特征构建 FAISS 索引，不负责训练、推理或特征提取。

- `src/index/__main__.py`：统一索引入口。配置模式下根据 `feature_dim` 自动分派到 v1 或 v2 构建器；手工模式要求显式传 `--feature-dim`。
- `src/index/build_v1.py`：v1/256 维特征索引构建。保持原始 `train-index.py` 的 IVF 参数、`nprobe=9`、`trained_*.index` 和最终 index 写出行为。
- `src/index/build_v2.py`：v2/768 维特征索引构建。保持原始 `train-index-v2.py` 的 shuffle、超大特征 MiniBatchKMeans 降采样、IVF 参数、批量 add 行为。
- `src/index/common.py`：索引构建公共边界，只做特征矩阵加载和 `big_src_feature.npy` 写出。

## `src/preprocess/`

预处理模块负责训练前数据产物：切片重采样、F0、HuBERT 特征。它不启动模型训练，也不构建索引。

- `src/preprocess/__main__.py`：音频切片预处理入口，调用 `src.preprocess.audio`。
- `src/preprocess/audio.py`：原始音频读取、静音切片、高通滤波、归一化，写出 `0_gt_wavs/` 和 `1_16k_wavs/`。保留原始配置模式、手工模式和 positional 参数语义。
- `src/preprocess/f0.py`：从 `1_16k_wavs/` 提取 F0，写出 `2a_f0/` 和 `2b-f0nsf/`。保留 `pm/harvest/dio/rmvpe` 方法、worker/shard 模式和日志语义。
- `src/preprocess/features.py`：加载 HuBERT，对 `1_16k_wavs/` 提取 v1/v2 特征，写出 `3_feature256/` 或 `3_feature768/`。模型只在入口运行时加载，避免 import 时产生副作用。
- `src/preprocess/utils/slicer.py`：预处理内部专用静音切片器，只供 `src.preprocess` 使用。

## `src/train/`

训练模块只负责训练流程和训练专属支撑代码，不再混入预处理或索引职责。

- `src/train/__main__.py`：训练入口，调用 `src.train.runner.main()`。
- `src/train/runner.py`：分布式训练主循环、模型/判别器创建、checkpoint 加载、epoch 训练与最终小模型导出触发。CLI 解析移到 `main()`，模块 import 不再启动训练。
- `src/train/utils.py`：训练配置转 `HParams`、checkpoint 读写、日志器、TensorBoard 汇总、训练文件读取等训练支撑函数。
- `src/train/data_utils.py`：训练 Dataset、BucketSampler、collate 函数。
- `src/train/losses.py`：判别器 loss、生成器 loss、feature matching loss、KL loss。
- `src/train/mel_processing.py`：spectrogram/mel 计算。
- `src/train/process_ckpt.py`：训练权重处理和小模型导出。

## `src/infer/`

推理模块只负责离线 voice conversion 运行时，不承担训练、预处理或索引构建。

- `src/infer/voice_converter.py`：`VC` 高层会话对象。负责加载导出模型、选择索引、调用 pipeline，提供单文件和批量转换方法。
- `src/infer/pipeline.py`：离线转换核心流水线。负责 F0 计算、HuBERT 特征提取、索引检索融合、模型 infer 调用、RMS 混合和重采样。
- `src/infer/model_utils.py`：推理侧模型路径、索引路径查找和 HuBERT 加载。

## `src/utils/`

共享工具模块服务于训练和推理等多个上层模块；预处理独有工具放在 `src/preprocess/utils/`。

- `src/utils/audio.py`：音频解码、重采样、格式转换。
- `src/utils/rmvpe.py`：RMVPE F0 模型实现，供推理和 F0 预处理共享。
- `src/utils/infer_pack/commons.py`：模型通用张量工具、初始化、segment 操作。
- `src/utils/infer_pack/transforms.py`：flow transform 数学函数。
- `src/utils/infer_pack/attentions.py`：Transformer/attention 模块。
- `src/utils/infer_pack/modules.py`：flow、resblock、WN、判别器子模块。
- `src/utils/infer_pack/models.py`：RVC 生成器、判别器和 Synthesizer 模型定义。

## 常用新入口

```bash
python -m src.preprocess --config configs/mute.yaml
python -m src.preprocess.f0 --exp-dir data/mute/preprocess_data --workers 1 --f0method rmvpe
python -m src.preprocess.features --config configs/mute.yaml
python -m src.train --config configs/mute.yaml
python -m src.index --config configs/mute.yaml
```

手工模式仍在 `src` 新入口下保留，用于保持功能等价；旧路径不保留兼容入口。
