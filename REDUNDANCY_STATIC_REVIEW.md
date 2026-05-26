# 冗余与任务边界静态审查报告

日期：2026-05-26

范围：`src/`、`configs/`、相关迁移边界测试。审查重点是冗余代码、死分支、无效配置、过期兼容层风险、导入边界和能用静态扫描确认的坏味道。

## 结论

本轮修复集中在三类问题：

1. 配置声明和实际行为不一致。
2. 训练、预处理、推理拆分后的残留冗余和过期参数链。
3. 旧迁移测试仍依赖已删除的 `infer/` 源树，导致当前项目边界验证失效。

保留项主要是兼容 API：`src/train/process_ckpt.py`、`src/utils/infer_pack/*` 和部分公开训练工具函数虽然生产路径不直接调用，但仍可能被外部脚本导入，不在本轮删除。

## 已修复问题

| 严重级别 | 位置 | 问题 | 修复 |
|---|---|---|---|
| 高 | `configs/project_config.py` | `train_dir/export_dir/index_dir` 可以配置但被 `work_dir` 派生值静默覆盖 | `_resolve_paths()` 改为显式解析这三个 path override |
| 中 | `configs/project_config.py` | `_normalize_overrides()` 先校验 top-level key，导致 `train.batch_size` 形式的 dotted key 分支不可达 | 先归一化 dotted key，再校验归一化后的 mapping |
| 低 | `configs/project_config.py` | `_flatten_aliases()` 已全量摊平 `paths`，后续逐项赋值重复 | 删除重复逐项赋值 |
| 中 | `src/train/runner.py` | `writer_eval`、`eval_loader`、`schedulers` 参数链创建/传递/解包但未使用 | 移除整条未使用传参链 |
| 中 | `src/train/runner.py` | 训练入口仍通过兼容 shim 导入 `savee` | 改为直接从 `src.train.checkpoint_export` 导入，保留 shim 给外部旧入口 |
| 低 | `configs/base.yaml` | `init_lr_ratio`、`warmup_epochs`、`data.max_wav_value` 当前无生产引用 | 删除无效默认项 |
| 低 | `src/train/data_utils.py` | `max_wav_value`、`min_text_len`、`max_text_len` 只赋值不参与逻辑 | 删除死属性 |
| 低 | `src/train/mel_processing.py` | 无用 import、logger、`MAX_WAV_VALUE` 常量 | 删除 |
| 低 | `src/train/utils.py` | `latest_checkpoint_path()` 的 debug 日志和模块导入时 root DEBUG 配置造成噪音 | 删除 debug 日志，避免导入时配置 root logging |
| 低 | `src/train/utils.py` | `np.fromstring(..., sep="")` 触发弃用警告 | 改为 `np.frombuffer()` |
| 中 | `src/preprocess/features.py` | config 模式文案允许 device/model option，但 `--device/--is-half` 会被项目配置覆盖 | `--device` 默认跟随项目配置，显式传入时保留；`--is-half` 改为可区分“未传入”和“显式启用” |
| 中 | `src/preprocess/f0.py` | `--is-half` 在 config/legacy/manual 模式下无法区分默认值和显式传参 | 改为 `default=None`，在最终阶段落到 bool |
| 低 | `src/infer/pipeline.py`、`src/models/models.py` | 残留低价值调试注释/空注释 | 删除 |
| 中 | `tests/test_equivalence_source_coverage.py` | 测试仍假定旧 `infer/` 源树存在 | 改为当前边界测试：旧树不存在、`src` 文件全分类、兼容 wrapper 只 re-export、canonical wildcard 目标定义 `__all__` |
| 中 | `tests/test_equivalence_visual.py` | 测试仍 import `infer.lib.train.utils` | 改为验证当前 `src.train.utils` 绘图输出的 RGB 形状和确定性 |

## 已确认但保留

| 位置 | 原因 |
|---|---|
| `src/train/process_ckpt.py` | 兼容旧入口；删除会破坏外部脚本导入 |
| `src/utils/infer_pack/*` | 兼容旧模型模块路径；canonical 模块已用 `__all__` 限制 wildcard 暴露 |
| `src/train/utils.py` 的 `load_checkpoint_d/save_checkpoint_d/check_git_hash` | 当前生产路径不调用，但属于公开训练工具函数，先不删 |
| `src/train/mel_processing.py` 的 decompression helper | 当前生产路径不调用，但属于公开 mel helper，删除有外部 API 风险 |
| `tests/cuda_numeric_probe.py` 的 `infer.lib...` import | 这是外部旧仓库数值对比探针，不属于 `src` 运行边界；未纳入当前 smoke |

## 验证记录

所有 Python 验证都使用 PowerShell job 外层监控：设置无输出超时和总超时；子进程异常会退出非 0。

| 验证 | 结果 |
|---|---|
| AST/BOM/近似未使用 import 扫描，范围 `src`、`configs`、`tests` | 通过：`parsed 60 files`，无 BOM，无语法错误，无未使用 import 候选；30s idle / 60s total |
| 配置解析 smoke | 通过：`train.batch_size` dotted override 生效，`train_dir/export_dir/index_dir` override 生效；60s idle / 120s total |
| CLI help smoke | 通过：`python -m src.preprocess --help`、`src.preprocess.f0 --help`、`src.preprocess.features --help` 均退出 0；60s idle / 120s total |
| broad smoke | 通过：infer/train/preprocess import、`spec_to_mel_torch` 两种 FFT 形状、F0 coarse、retrieval blend、legacy wrapper identity、model init、RMVPE mel；90s idle / 240s total |
| `unittest tests.test_equivalence_source_coverage tests.test_equivalence_visual` | 通过：7 tests OK；120s idle / 300s total |
| pytest | RVC 环境缺 `pytest` 模块；base Anaconda 的 pytest 可运行，但先前失败原因是过期旧 `infer/` 边界测试，已改用 RVC 环境的 `unittest` 覆盖同一文件 |

