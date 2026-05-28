# RVC_rebuild Codebase Slimming Report

日期：2026-05-28  
仓库：`F:\deeplearning\SVC\RVC_rebuild`

## 1. 结论摘要

这个仓库已经完成过一轮“旧 `infer/` 树迁移到 `src/`”的大重构，但还保留了不少兼容层、过渡入口和历史测试。  
如果目标是“尽量减少代码量，同时尽量不碰核心算法”，最佳路径不是先动 `src/models/` 或 `src/train/runner.py`，而是先清掉下面三类负担：

1. 已经失效的历史测试与手工对比脚本。
2. 只为兼容旧导入路径或旧 WebUI 形态保留的薄封装。
3. 预处理/索引/训练侧重复出现的脚本胶水代码。

按风险和收益综合排序，最值得优先做的是：

1. 删除依赖已移除旧 `infer/` 树的 5 个跳过测试，以及 `cuda_numeric_probe.py` 这类旧仓库对比脚本。
2. 评估并删除 `src/infer/voice_converter.py` 这种 WebUI 风格适配层。
3. 合并 `src/index/build_v1.py` / `src/index/build_v2.py`，保留一个参数化入口。
4. 把 HuBERT 特征提取、F0 method dispatch、预处理日志/多进程调度下沉成共享 helper。
5. 如果接受进一步降兼容，再裁掉 `src/train/process_ckpt.py`、部分旧 CLI positional 参数支持、以及 `configs/project_config.py` 中的旧字段兼容逻辑。

如果只做前两层低风险清理，粗略就能减少数百行，而且不需要先碰模型数值实现。

## 2. 调研方法

本次结论来自以下证据：

- 静态盘点 `src/`、`configs/`、`tests/` 结构和行数分布。
- 逐文件检查高体量模块、入口模块、兼容 wrapper、测试引用关系。
- 用 `RVC` conda 环境串行运行测试：
  - `F:\Anaconda3\envs\RVC\python.exe -m unittest tests.test_equivalence_source_coverage tests.test_preprocess_pipeline tests.test_train_strict_runtime tests.test_train_validation_config`
  - `F:\Anaconda3\envs\RVC\python.exe -m unittest discover -s tests`
- 并行做了三组子调查：
  - `src/models/` 与 `src/utils/infer_pack/` 的关系。
  - `preprocess` / `features` / `infer` 的职责交叉。
  - `train` / `index` 的复杂度热点与可裁剪入口。

说明：

- 本报告以当前工作树为准。
- 读写和测试均按 UTF-8 约束处理。
- 报告中的“可删除”指从仓库内部静态证据看具备很强候选性，不等于已经证明仓库外部没有用户依赖。

## 3. 仓库基线

### 3.1 tracked 文件规模

已跟踪的文本类文件共 71 个。

按目录统计 Python 代码行数：

| 目录 | 文件数 | 行数 |
| --- | ---: | ---: |
| `src/train` | 10 | 2508 |
| `src/models` | 6 | 2311 |
| `src/preprocess` | 6 | 1308 |
| `tests` | 13 | 1150 |
| `configs` | 1 | 889 |
| `src/infer` | 6 | 768 |
| `src/utils` | 3 | 426 |
| `src/index` | 7 | 309 |
| `src/features` | 4 | 190 |
| `src/utils/infer_pack` | 6 | 6 |

体量最大的单文件：

| 文件 | 行数 |
| --- | ---: |
| `src/train/runner.py` | 1060 |
| `src/models/models.py` | 1043 |
| `configs/project_config.py` | 889 |
| `src/preprocess/pipeline.py` | 593 |
| `src/train/utils.py` | 538 |
| `src/models/modules.py` | 510 |
| `src/models/attentions.py` | 407 |
| `src/infer/pipeline.py` | 380 |
| `src/utils/rmvpe.py` | 372 |

### 3.2 测试基线

`2026-05-28` 实测：

- `python -m unittest discover -s tests`
  - 40 tests
  - 1 failure
  - 7 skipped

唯一失败：

- `tests/test_equivalence_source_coverage.py::test_every_src_python_file_is_classified`
  - 失败原因：`src/train/deterministic_gpu.py` 已存在于 `src/`，但没有登记进 `EXPECTED_SRC_FILES`。

这说明测试集本身存在“仓库已经演化、测试清单没同步收口”的现象，属于典型可减负点。

7 个跳过测试全部绑定到已移除的旧 `infer/` 树：

- `tests/test_equivalence_preprocess.py`
- `tests/test_equivalence_features.py`
- `tests/test_equivalence_f0.py`
- `tests/test_equivalence_index.py`
- `tests/test_equivalence_infer.py`
- 以及相关旧树依赖路径上的跳过分支

这些跳过测试总计 322 行，已经不是活跃保护网，而是历史包袱。

## 4. 最高优先级机会

### 4.1 删除已经失效的历史等价测试与对比脚本

#### 证据

- 5 个旧等价测试都以 `@unittest.skipUnless((REPO_ROOT / "infer").exists(), "legacy infer tree removed")` 为前提：
  - `tests/test_equivalence_preprocess.py:15`
  - `tests/test_equivalence_features.py:17`
  - `tests/test_equivalence_f0.py:10`
  - `tests/test_equivalence_index.py:11`
  - `tests/test_equivalence_infer.py:8`
- 当前仓库又明确要求旧 `infer` 树已经移除：
  - `tests/test_equivalence_source_coverage.py:86-87`
- 这些跳过测试里已经出现“长期无人维护”的证据：
  - `tests/test_equivalence_preprocess.py:47-51` 调用了 6 参数的 `preprocess_trainset(...)`
  - 但当前 `src/preprocess/audio.py:140` 的 `preprocess_trainset` 只有 5 参数
  - 这类测试即使恢复执行也大概率已经过时
- `tests/cuda_numeric_probe.py:10-12`、`39-60` 直接依赖外部旧仓库 `Retrieval-based-Voice-Conversion-WebUI` 和 `infer.lib.infer_pack`

#### 收益

- 直接可删规模：
  - 5 个跳过测试共 322 行
  - `tests/cuda_numeric_probe.py` 143 行
  - 合计约 465 行
- 同时减少“仓库里看起来有回归保护，实际上完全不执行”的错觉。

#### 风险

- 唯一风险是你还想保留“和旧仓库逐项数值比对”的历史审计能力。
- 如果还需要这类审计，更合理的做法是把它们移到 `archive/` 或单独文档，不要放在日常 `tests/` 里。

#### 建议

- 这是整个报告里最推荐先做的动作。
- 删除后保留 1 个简短文档说明“旧 infer 等价回归已完成，旧树已移除，因此这批测试退役”。

### 4.2 修正或重写 `test_equivalence_source_coverage`

#### 证据

- 该测试当前是全套测试中唯一 failure：
  - `tests/test_equivalence_source_coverage.py:89-95`
- 它维护了一张手工枚举的 `EXPECTED_SRC_FILES`：
  - `tests/test_equivalence_source_coverage.py:67-72`
- 但新增的 `src/train/deterministic_gpu.py` 没有同步登记，导致基线非绿。

#### 收益

- 这不是“删代码”的大户，但它直接决定后续任何精简工作都能否有干净基线。
- 同时它还暴露出另一个问题：这张人工清单本身就是维护负担。

#### 建议

两种方向：

1. 如果你要继续保留“源树分类约束”，先把 `src/train/deterministic_gpu.py` 纳入清单。
2. 如果目标是减负，建议弱化这个测试：
   - 只保留“兼容 wrapper 必须是纯 re-export”的断言。
   - 不再维护全量 `EXPECTED_SRC_FILES` 名单。

第二种更符合“减少维护面”的目标。

## 5. 中低风险、高收益的源码精简点

### 5.1 `src/infer/voice_converter.py` 很像遗留 WebUI 适配层

#### 证据

- `README.md:3` 明确说仓库不是完整 WebUI 工程，训练流程以命令行和 YAML 为主。
- `src/infer/voice_converter.py` 里存在明显的 UI 返回结构：
  - `__type__ = "update"` 风格的返回值，见 `13-73`
  - 面向单文件/批量转换的 `vc_single` / `vc_multi`，见 `75-183`
- 静态引用扫描结果：
  - `src/infer/voice_converter.py` 没有任何 `src/` 内部引用
  - 没有测试引用
  - README 也没有提到它

#### 判断

- 这 173 行更像“给旧 UI 或交互层留的服务对象”，不是当前仓库 CLI 主链路的核心部分。
- 如果你已经不打算兼容 WebUI 风格调用，这个文件是很强的删除候选。

#### 风险

- 可能有仓库外部脚本直接 `from src.infer.voice_converter import VC`。
- 仓库内静态扫描无法证明外部没有依赖。

#### 建议

- 如果你愿意明确宣布“不再提供旧 VC 类接口”，可直接删。
- 如果还想保留兼容，但又想减小维护成本，可以只保留一个更薄的 shim：
  - `VC = VoiceConversionService`
  - 或把 UI 专有逻辑移出仓库

### 5.2 `src/index/build_v1.py` 与 `src/index/build_v2.py` 明显重复

#### 证据

- 两个文件都是“解析参数 -> 读 config -> 校验 feature_dim -> 调 `build_faiss_index(...)`”：
  - `src/index/build_v1.py:8-56`
  - `src/index/build_v2.py:9-62`
- 基于行级相似度的粗测，两个文件的 line similarity 约 `0.766`。
- 真正核心逻辑已经统一在：
  - `src/index/builder.py:12-76`
  - `src/index/common.py:5-35`
- 当前 README 仍然把两个版本入口作为用户可调用命令写出来：
  - `README.md:297-298`

#### 判断

- 这两个文件不是必要的两套实现，而是同一个入口被版本拆成两份脚本。

#### 建议

优先级很高，且实现方式有两种：

1. 保守方案：
   - 保留 `build_v1.py` / `build_v2.py`
   - 但把它们缩成薄 wrapper
   - 真正的参数解析统一到 `src/index/__main__.py` 或 `src/index/builder.py`
2. 激进方案：
   - 删除 `build_v1.py` / `build_v2.py`
   - 统一只保留 `python -m src.index`
   - 用 `feature_dim` 或 `selectors.version` 驱动行为

如果只考虑仓库内部，第二种更干净；如果考虑用户命令习惯，第一种更稳妥。

### 5.3 `src/train/process_ckpt.py` 是纯兼容 wrapper

#### 证据

- 整个文件只是在重导出：
  - `src/train/process_ckpt.py:1-14`
- 仓库内部没有任何引用。
- 测试只把它当兼容 wrapper 分类：
  - `tests/test_equivalence_source_coverage.py:22-29`

#### 判断

- 这是低收益、低风险、但也低优先级的删除点。
- 只有 14 行，真正价值不在“减几行”，而在减少一个旧名字。

#### 建议

- 如果你准备系统性降低旧 API 兼容面，可以一并删除。
- 如果还想维持旧导入路径，这个文件已经足够薄，不值得专门动它。

### 5.4 `src/utils/infer_pack/*` 已经不是重复实现，只是兼容导入层

#### 证据

- `src/utils/infer_pack/attentions.py:1`
- `src/utils/infer_pack/commons.py:1`
- `src/utils/infer_pack/models.py:1`
- `src/utils/infer_pack/modules.py:1`
- `src/utils/infer_pack/transforms.py:1`

这 5 个文件都只有一行 `from src.models... import *`。

- 当前真正实现都在 `src/models/*`：
  - `src/infer/service.py:12-17`
  - `src/train/runner.py:30-56`
  - `src/train/deterministic_gpu.py:38-45`
  都直接依赖 `src.models`

#### 判断

- 这里没有“两套模型实现”可合并，合并早就做完了。
- 现在剩下的只是兼容层。

#### 建议

- 如果你接受 breaking change，可以整体删除。
- 但它总共只有 6 行，收益非常有限。
- 相比之下，外部导入 breakage 风险更高，所以我不建议把这组 wrapper 放在第一波处理。

## 6. 中风险、可观收益的重构去重

### 6.1 提升 HuBERT 共享边界，别再在 `preprocess` 和 `infer` 各写一遍

#### 证据

- `src/features/hubert.py` 目前只共享了：
  - `read_wave_16k`，见 `7-19`
  - `load_hubert_model`，见 `22-65`
- 但真正的 HuBERT 前向准备逻辑在两边都写了一遍：
  - `src/preprocess/features.py:88-100`
  - `src/infer/pipeline.py:138-160`
- 两边都在做：
  - 单声道处理
  - `layer_norm`
  - `view(1, -1)`
  - `padding_mask`
  - `output_layer` 选择
  - `model.extract_features`
  - `v1` 场景下 `final_proj`

#### 判断

- 共享层粒度太低。
- 这不是算法重复，而是推理前后处理胶水重复。

#### 建议

把以下 helper 下沉到 `src/features/hubert.py`：

- `prepare_hubert_input(...)`
- `extract_hubert_features(model, audio, version, normalize=...)`

这样：

- `src/preprocess/features.py` 只负责遍历文件和保存 `.npy`
- `src/infer/pipeline.py` 只负责切块、索引混合和后处理

### 6.2 统一 F0 method dispatch

#### 证据

- `src/features/f0.py` 已经提供 primitive：
  - `compute_pm_f0`，`8-25`
  - `compute_world_f0`，`28-51`
  - `compute_crepe_f0`，`54-73`
  - `load_rmvpe_model`，`76-81`
  - `f0_to_coarse`，`84-104`
- 但 method 分派、RMVPE 懒加载和 coarse 化在两边重写：
  - `src/preprocess/f0.py:85-108`
  - `src/infer/pipeline.py:76-121`

#### 判断

- 共享 primitive 已经存在，但共享接口还不够高层。

#### 建议

把 `src/features/f0.py` 升级为统一入口，比如：

- `compute_f0_by_method(...)`
- 或 `F0Extractor(method, device, is_half, pretrain_root)`

再让：

- `preprocess/f0.py` 只负责批量遍历和 `.npy` 落盘
- `infer/pipeline.py` 只负责升降调、`f0_file` 覆写和切段

### 6.3 抽出统一日志 helper

#### 证据

相似的 `print + append log file` helper 至少有四份：

- `src/preprocess/audio.py:23-31` `println`
- `src/preprocess/f0.py:24-29` `log_message`
- `src/preprocess/features.py:15-20` `log_message`
- `src/preprocess/pipeline.py:41-47` `_log_message`

#### 建议

抽一个共享函数，比如：

- `append_log(log_path, message, echo=True)`

同时评估 `src/preprocess/audio.py:13-21` 那套 `LOG_HANDLE` 是否还值得保留。  
如果不再追求跨进程长时间持有文件句柄，这套全局状态也可以删掉。

### 6.4 统一预处理 stage 的多进程调度骨架

#### 证据

- `src/preprocess/pipeline.py:501-549` 手写了一套 audio stage 的 `Process/start/join/exitcode`
- `src/preprocess/audio.py:108-137` 里面又有一套非常接近的逻辑
- `src/preprocess/pipeline.py:552-613` 对 F0 stage 又写了一遍类似模式

#### 判断

- 真正重复的是 orchestration，不是音频算法本身。

#### 建议

把“分片 + 启进程 + join + exit code 检查”抽成一个小 runner。  
`pipeline.py` 只决定“跑哪个 stage，传什么 payload”，不要再直接管每个 stage 的内部 worker 细节。

### 6.5 训练侧工具函数还有几组可收口重复

#### 证据

`src/train/utils.py` 里有几组明显重复：

- checkpoint 读写双份：
  - `load_checkpoint_d`，`24-69`
  - `load_checkpoint`，`71-118`
  - `save_checkpoint`，`121-140`
  - `save_checkpoint_d`，`143-166`
- matplotlib 初始化三次重复：
  - `plot_spectrogram_to_numpy`，`208-229`
  - `plot_validation_mels_to_numpy`，`232-283`
  - `plot_alignment_to_numpy`，`286-312`
- 训练字段 alias 平铺逻辑：
  - `_sync_train_aliases`，`353-363`
  - `_apply_training_cli_overrides`，`366-430`

`src/train/checkpoint_export.py` 内部也有类似的“剔除 `enc_q`、拼 export payload”的重复：

- `savee`，`39-74`
- `extract_small_model`，`90-118`
- `merge`，`147-189`

另外，`src/train` 里还有一批“仓库内零引用”的 helper，适合在你明确功能边界后直接删掉：

- `src/train/utils.py:24-69` `load_checkpoint_d`
- `src/train/utils.py:143-166` `save_checkpoint_d`
- `src/train/utils.py:538-543` `get_hparams_from_dir`
- `src/train/utils.py:546-550` `get_hparams_from_file`
- `src/train/utils.py:553-576` `check_git_hash`

`src/train/checkpoint_export.py` 中，训练主链路内部实际只用：

- `src/train/checkpoint_export.py:39-74` `savee`

其余函数：

- `show_info`
- `extract_small_model`
- `change_info`
- `merge`

当前仓库内部没有调用。如果你不再想提供“checkpoint 读信息 / 改 metadata / 合并模型”的仓库内工具面，这一整段都可视为强候选可删面。

#### 建议

- checkpoint 读写可抽“state dict 装配/过滤器”。
- matplotlib setup 可抽 `_ensure_matplotlib_agg()`。
- checkpoint export 可抽 `_export_weight_payload(...)`。

这些都不是第一优先级，但性价比不错。

### 6.6 `deterministic_gpu` 和 `mel_processing` / `features.mel` 有一部分同源实现

#### 证据

- `src/features/mel.py:4-58` 定义了 mel 频率转换和 `build_mel_basis`
- `src/train/mel_processing.py:37-127` 定义了常规 `spectrogram_torch` / `spec_to_mel_torch` / `mel_spectrogram_torch`
- `src/train/deterministic_gpu.py:48-105` 又实现了一套 `_hz_to_mel` / `_mel_to_hz` / `_mel_frequencies` / `_build_mel_basis`
- `src/train/deterministic_gpu.py:173-232` 又实现了一套 deterministic 版 `spectrogram_torch` / `spec_to_mel_torch` / `mel_spectrogram_torch`

#### 判断

- 这里不是简单文件级相似，而是“共享数学定义 + 不同执行后端”。
- 不应该粗暴合并成一个函数，但可以复用公共数学 helper。

#### 建议

- 至少让 deterministic 路径复用 `src/features/mel.py` 的 mel basis 构造逻辑。
- 保留 deterministic 版 STFT 实现本身，因为它确实承担不同数值保证。

### 6.7 `deterministic_gpu` 是一整块可独立裁掉的可选功能

#### 证据

- `src/train/deterministic_gpu.py:17-302` 是一个完整的独立功能面。
- `src/train/runner.py` 为它额外承担了多处分支：
  - `173-178`
  - `320-326`
  - `430-450`
  - `611-623`
  - `919-943`
- `tests/test_train_strict_runtime.py:112-163` 只覆盖了其中一小部分 helper。
- README 没有把这一功能当成面向普通用户的公开主路径单独介绍。

#### 判断

- 如果你接受功能收缩，而不只是内部去重，`deterministic_gpu` 是“删一块、连带删一圈分支”的典型目标。
- 它的收益会明显大于删除 1 行 wrapper。

#### 风险

- 这不是纯内部整理，而是功能面裁剪。
- 任何依赖严格数值复现、单 GPU deterministic patch 的用法都会受影响。

#### 建议

- 若要保留功能：做局部去重即可，不要第一波重构。
- 若允许缩功能：可以把它列入第二或第三波清理，和 `runner.py` 的关联分支一起下掉。

## 7. 高风险、但仍然存在的进一步瘦身点

### 7.1 `configs/project_config.py` 职责过重

#### 证据

这个文件 889 行，承担了太多职责：

- 配置读取与继承链解析：
  - `_read_yaml_file`，`206-212`
  - `_load_config_chain`，`284-295`
- `--hparams` 解析：
  - `parse_hparams_overrides`，`566-592`
- 运行时环境探测：
  - `_detect_runtime_environment`，`636-690`
- runtime autofill：
  - `_resolve_runtime_profile`，`694-805`
- 路径推导：
  - `_resolve_paths`，`809-847`
- alias flatten：
  - `_flatten_aliases`，`861-887`
- snapshot 构建与落盘：
  - `_build_snapshot_config`，`891-926`
  - `save_project_config_snapshot`，`1045-1060`

此外还保留了大量旧字段兼容逻辑：

- `REMOVED_TOP_LEVEL_FIELD_HINTS`，`143-164`
- `REMOVED_TRAIN_FIELD_HINTS`，`166-173`
- `train.mel_loss_device` 兼容映射到 `numeric_backend`：
  - `721-725`
  - `782-804`

#### 判断

- 这是一个典型的“单文件中枢”，后续任何精简都容易在这里碰到旧兼容包袱。
- 但这个文件也是整个仓库路径与 runtime 默认值的稳定器，不能贸然大删。

#### 建议

先分两步：

1. 结构拆分，不先追求立即减行：
   - `config_loading`
   - `runtime_resolution`
   - `path_resolution`
   - `snapshot_io`
2. 再决定是否删兼容逻辑：
   - 顶层旧字段提示
   - `mel_loss_device` alias
   - 扁平 alias 导出字段

如果你愿意明确仓库只服务“新 YAML + 新入口”，这个文件后续还有较大减量空间。  
但这属于后期工程，不建议第一波处理。

### 7.2 模型类家族仍有可抽象重复，但不建议最先动

#### 证据

- `src/models/models.py:636-782` `SynthesizerTrnMs256NSFsid`
- `src/models/models.py:785-829` `SynthesizerTrnMs768NSFsid`
- `src/models/models.py:832-937` `SynthesizerTrnMs256NSFsid_nono`
- `src/models/models.py:940-...` `SynthesizerTrnMs768NSFsid_nono`

当前设计已经把一部分差异参数化了：

- `feature_dim`
- `use_f0`

但 `*_nono` 变体仍然复制了 `forward` / `forward_val` / `reconstruct_full` / `infer` 的大量结构，只是把 `pitch` 分支去掉。

#### 判断

- 这里理论上还能再缩。
- 但这是模型核心路径，风险远高于脚本层、测试层和兼容层。

#### 建议

- 只有在前面几轮清理做完之后，再考虑把 `use_f0` 差异继续内收。
- 不建议把第一轮精简预算用在这里。

## 8. 不建议优先动的点

### 8.1 不要把 `src/models/` 当成“可删的重复实现”

`src/utils/infer_pack/` 已经只是 wrapper。  
真正实现只在 `src/models/`。删错方向会直接伤到训练和推理主链路。

### 8.2 不要为了几行代码优先删除 `src/utils/infer_pack/*`

虽然它们是兼容残留，但总共只有 6 行。  
如果没有明确 breaking-change 意愿，这一组收益太小，不值得先动。

### 8.3 不要在测试基线非绿时直接做大重构

当前基线已经有 1 个 failure。  
先把测试面收口，不然后续很难分辨“精简引入的新问题”和“原仓库就存在的问题”。

## 9. 推荐执行顺序

### Phase 1：先清死代码和死测试

目标：快速减量，不碰核心数值逻辑。

建议动作：

1. 删掉 5 个依赖旧 `infer/` 树的跳过测试。
2. 删掉 `tests/cuda_numeric_probe.py`，或移到 `archive/`。
3. 修掉 `tests/test_equivalence_source_coverage.py` 的失配，或者把它缩成只检查兼容 wrapper。
4. 评估删除 `src/infer/voice_converter.py`。

### Phase 2：合并明显重复的入口脚本

建议动作：

1. 合并 `src/index/build_v1.py` / `src/index/build_v2.py`。
2. 如果愿意缩兼容面，再评估 `src/train/process_ckpt.py`。
3. 审视 README 中仍然鼓励用户直接调用的旧版本入口命令。

### Phase 3：压缩 `preprocess` / `infer` 胶水代码

建议动作：

1. HuBERT 前向 helper 下沉。
2. F0 dispatch 下沉。
3. 日志 helper 统一。
4. stage multiprocessing runner 统一。
5. 若明确只保留 config-first 流程，删掉旧 positional CLI 兼容。

### Phase 4：再考虑配置与核心模型

建议动作：

1. 拆 `configs/project_config.py`。
2. 减训练工具重复。
3. 最后才考虑模型类内部抽象收口。

## 10. 粗略减量预估

以下是比较保守的粗略区间，不是精确承诺：

| 动作 | 粗略可减少行数 |
| --- | ---: |
| 删除旧 `infer` 跳过测试 + `cuda_numeric_probe.py` | 450-470 |
| 删除 `src/infer/voice_converter.py` | 170 左右 |
| 合并 `src/index/build_v1.py` / `build_v2.py` | 60-100 |
| 删除 `src/train/process_ckpt.py` | 10-15 |
| 统一 preprocess 日志与多进程骨架 | 40-90 |
| HuBERT / F0 胶水下沉共享 | 60-140 |
| 训练工具和 checkpoint export 去重 | 50-100 |
| 配置兼容逻辑缩减 | 150+，但风险高 |

如果只做前两阶段，不碰高风险核心代码，理论上就有机会减掉约 700 行上下的仓库代码与测试代码。

如果进一步接受“功能面收缩”而不是只做内部瘦身：

- 裁掉 `deterministic_gpu` 及其配套分支，减量会继续上升。
- 但这一步的风险显著高于前两阶段，不应和“纯删兼容层”混为一谈。

## 11. 验证建议

每一阶段都建议用 `RVC` 环境跑串行验证：

```powershell
F:\Anaconda3\envs\RVC\python.exe -m unittest discover -s tests
```

如果进入 Phase 3 以后改动到预处理/训练入口，建议至少补跑：

```powershell
F:\Anaconda3\envs\RVC\python.exe -m unittest tests.test_preprocess_pipeline tests.test_train_strict_runtime tests.test_train_validation_config
```

如果删掉兼容 wrapper 或旧入口，要同步做三件事：

1. 改 README。
2. 改 `tests/test_equivalence_source_coverage.py`。
3. 明确在变更说明里写出“这是一次兼容面收缩，不是单纯内部重构”。

当前测试覆盖还有一个现实问题：  
`run()`、`train_and_evaluate()`、`validate()`、`build_faiss_index()`、`load_retrieval_index()`、`blend_search_features()` 这些真正的大块流程，并没有被现有活跃测试直接兜住。  
所以越往后期走，越应该把“先补最小 smoke tests，再删代码”作为节奏，而不是直接大裁主流程。

## 12. 最终建议

如果你的目标是“尽量减少代码量，但不要把核心算法重构成高风险工程”，我建议：

1. 第一波只删死测试、死脚本、纯 UI 兼容壳子、重复入口。
2. 第二波专门收 `preprocess` / `infer` 的共享 helper。
3. 先不要把主要精力放在 `src/models/` 主体和 `src/train/runner.py` 上。

最重要的一点是：  
这个仓库现在最大的冗余，不在模型数学实现本身，而在“迁移后还没有完全退出历史模式”的兼容层和脚本层。
