# RVC Rebuild 文件职责边界

本仓库的业务入口统一在 `src/` 下。旧 `infer/` 源树不再作为入口或兼容层使用；需要保留旧导入路径时，只在 `src/utils/infer_pack/` 和 `src/train/process_ckpt.py` 这类显式 shim 中处理。

本文档按“谁负责什么、谁不负责什么”描述当前文件边界，便于后续清理冗余代码和避免职责回流。

## 顶层目录

- `configs/`：项目配置解析与默认 YAML。`configs/project_config.py` 负责配置继承、`--hparams` 覆盖、runtime 自动解析、路径派生和 snapshot。
- `src/`：训练、预处理、索引构建、离线推理、共享特征/模型组件的唯一源码入口。
- `tests/`：当前 `src/` 边界、兼容 shim、可视化工具和数值探针测试。
- `data/`：训练原始音频、预处理产物和 filelist。
- `ckpt/`：训练日志、大 checkpoint、导出小模型和 FAISS 索引。
- `pretrain/`：HuBERT、RMVPE、预训练 G/D 权重。
- `REDUNDANCY_STATIC_REVIEW.md`：冗余/非必要代码静态审查结果和验证记录。

## `configs/`

- `configs/base.yaml`：默认配置，只保留当前代码实际使用的默认项。
- `configs/mute.yaml`：示例项目配置。
- `configs/project_config.py`：配置加载核心。负责：
  - `base_config` 继承和深合并。
  - `--hparams` 字符串解析和 dotted key 覆盖。
  - `selectors/runtime/preprocess/infer/train/data/model` 分区归一化。
  - `work_dir/train_dir/export_dir/index_dir` 等路径解析。
  - `HParams` 兼容对象和 snapshot 读写。

配置模块不应导入训练、推理、预处理执行逻辑。

## `src/features/`

共享特征算法层，只放可被预处理和推理复用的纯功能模块。

- `src/features/f0.py`：F0 后端公共函数和 `f0_to_coarse()`，包括 `pm`、`harvest/dio`、`crepe`、RMVPE 加载入口。
- `src/features/hubert.py`：HuBERT 模型加载和 16k wav 读取/归一化。
- `src/features/mel.py`：librosa-free mel basis 构建。

该目录不处理 CLI、项目配置、文件遍历、日志文件或训练流程。

## `src/models/`

RVC 神经网络 canonical 实现。新代码应从这里导入模型、attention、flow、commons。

- `src/models/commons.py`：张量工具、初始化、segment 操作。
- `src/models/transforms.py`：flow transform 数学函数。
- `src/models/attentions.py`：Transformer/attention 模块。
- `src/models/modules.py`：flow、resblock、WN、判别器子模块。
- `src/models/models.py`：Synthesizer、Generator、Discriminator 等顶层模型。

这些模块定义 `__all__`，用于限制 legacy wildcard wrapper 的导出面。

## `src/utils/`

跨训练/推理共享的非模型工具。

- `src/utils/audio.py`：音频解码、重采样、格式转换、路径清理。
- `src/utils/rmvpe.py`：RMVPE 网络和推理封装，供 `src.features.f0` 调用。
- `src/utils/infer_pack/*.py`：旧 `infer_pack` 导入路径的兼容 wrapper。除 `__init__.py` 外，文件应只做 `from src.models... import *`，不承载新逻辑。

新功能不要继续加入 `src/utils/infer_pack/`；应放到 `src/models/` 或更具体的业务目录。

## `src/preprocess/`

训练前数据产物生成：音频规范化/重采样、F0、HuBERT 特征。预处理模块不启动训练，也不构建索引。

- `src/preprocess/__main__.py`：短入口，委托 `src.preprocess.pipeline`。
- `src/preprocess/pipeline.py`：训练前聚合预处理入口，顺序执行音频、F0、HuBERT 特征和 `filelist.txt` 生成，并写 `preprocess_manifest.jsonl`。
- `src/preprocess/audio.py`：读取用户已切好的音频、高通滤波、归一化，写出 `0_gt_wavs/` 和 `1_16k_wavs/`；不再做自动切片。
- `src/preprocess/f0.py`：从 `1_16k_wavs/` 提取 F0，写出 `2a_f0/` 和 `2b-f0nsf/`。
- `src/preprocess/features.py`：加载 HuBERT，提取 v1/v2 特征，写出 `3_feature256/` 或 `3_feature768/`。

训练前推荐只运行聚合入口；`audio.py`、`f0.py`、`features.py` 保留为阶段实现和调试入口，核心算法应尽量复用 `src/features/`。

## `src/index/`

索引模块只负责从 HuBERT 特征矩阵构建和读取检索索引，不负责特征提取、训练或推理。

- `src/index/__main__.py`：统一索引 CLI/config 入口，按 `feature_dim` 分派 v1/v2。
- `src/index/common.py`：特征矩阵加载和构建公共参数。
- `src/index/builder.py`：FAISS 训练、KMeans 降采样、写出 index 的共享实现。
- `src/index/build_v1.py`：v1/256 维索引入口。
- `src/index/build_v2.py`：v2/768 维索引入口。
- `src/index/retrieval.py`：推理侧读取 index、校验 `big_src_feature.npy`、检索特征加权融合。

推理代码只能通过 `src.index.retrieval` 消费索引，不应直接写 FAISS 构建逻辑。

## `src/train/`

训练模块只负责训练流程和训练专属支撑代码，不处理预处理或索引职责。

- `src/train/__main__.py`：训练入口，调用 `src.train.runner.main()`。
- `src/train/runner.py`：分布式训练主循环、模型/判别器创建、checkpoint 加载、epoch 训练、导出触发。
- `src/train/data_utils.py`：Dataset、BucketSampler、Collate。
- `src/train/losses.py`：判别器 loss、生成器 loss、feature matching loss、KL loss。
- `src/train/mel_processing.py`：spectrogram/mel 计算。
- `src/train/utils.py`：训练配置转 `HParams`、checkpoint 读写、日志器、TensorBoard 汇总、filelist 读取、可视化数组生成。
- `src/train/checkpoint_export.py`：小模型导出、信息查看、checkpoint merge/change info。
- `src/train/process_ckpt.py`：旧入口兼容 shim，向外 re-export `checkpoint_export`。

训练主路径应直接导入 `checkpoint_export.py`，不要再通过 `process_ckpt.py` shim 反向依赖。

## `src/infer/`

推理模块只负责离线 voice conversion 运行时，不负责训练、预处理或索引构建。

- `src/infer/service.py`：数组级 voice conversion 服务。负责模型状态、checkpoint 加载、HuBERT 加载、Pipeline 调用。
- `src/infer/voice_converter.py`：面向旧 UI/批处理调用习惯的高层适配器，处理单文件/批量路径、输出保存和 Gradio update dict。
- `src/infer/pipeline.py`：离线转换核心流水线。负责 F0、HuBERT 特征、检索融合、模型 infer、RMS 混合、重采样。
- `src/infer/model_utils.py`：推理侧模型路径、索引路径和 HuBERT 路径解析。
- `src/infer/batch.py`：批量输入枚举和输出音频保存。

新的非 UI 推理调用优先使用 `VoiceConversionService`；`VC` 保留旧调用形状。

## `tests/`

- `tests/test_equivalence_source_coverage.py`：当前 `src` 文件分类、旧 `infer/` 树移除、兼容 wrapper 边界、canonical `__all__` 检查。
- `tests/test_equivalence_visual.py`：`src.train.utils` 可视化输出形状和确定性检查。
- `tests/cuda_numeric_probe.py`：可选 CUDA 数值对比探针，会引用外部旧仓库路径；它不是当前 `src` 运行边界的一部分。
- `tests/equivalence_helpers.py`：测试辅助函数和 fake 依赖。

## 常用入口

```bash
python -m src.preprocess --config configs/mute.yaml
python -m src.preprocess.pipeline --config configs/mute.yaml
python -m src.train --config configs/mute.yaml
python -m src.index --config configs/mute.yaml
```

手工模式和 legacy positional 模式只用于兼容旧脚本。新增入口和内部调用应优先使用 config 模式。

## 维护规则

- 新模型代码放 `src/models/`，不要放 `src/utils/infer_pack/`。
- 新特征算法放 `src/features/`，不要复制到预处理和推理两边。
- 索引构建逻辑放 `src/index/builder.py`，推理检索消费放 `src/index/retrieval.py`。
- 训练导出逻辑放 `src/train/checkpoint_export.py`，`process_ckpt.py` 只保留兼容导出。
- 配置字段只有被生产代码读取时才进入 `configs/base.yaml`。
- 文档、源码和测试统一使用 UTF-8 无 BOM。
