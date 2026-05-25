# 项目文件说明

本文档描述当前仓库的目录职责，以及 SOAP 风格配置链路的落点。

## 根目录

- `README.md`：项目总览与使用方式
- `PROJECT_FILE_GUIDE.md`：本说明文件
- `requirements.txt`：运行依赖
- `pyproject.toml`：项目元数据
- `tools/migrate_layout.py`：旧布局迁移脚本

## 配置

- `configs/project_config.py`：唯一配置解析器。负责解析 `base_config` 继承链、加载 `work_dir/config.yaml`、应用 `--hparams` 标量覆盖、选择 `variants`，并完成 runtime auto 补全。
- `configs/base.yaml`：共享默认配置。
- `configs/<task>.yaml`：任务配置入口。顶层包含 `base_config`、可选根路径字段、`selectors`、`preprocess`、`runtime`、`infer`、`train`、`data`、`model`、`variants`。
- `ckpt/<name>/config.yaml`：训练入口生成的可重放任务快照。只在训练入口写入；preprocess/index/纯解析模式只读不写。

推荐写法：

- 常规任务显式保留 `work_dir`、`data_root`、`ckpt_root`、`pretrain_root`
- `dataset_dir/preprocess_dir/train_dir/export_dir/index_dir` 通常由 resolver 自动推导
- `final_model_name/final_index_name` 默认由 `name` 自动推导
- 只有偏离默认布局时才覆盖这些派生字段

旧字段已移除：

- `version/sample_rate/if_f0`：改为 `selectors.*`
- `experiment_dir/ckpt_dir`：改为 `work_dir`
- `preprocess_per/noparallel`：改为 `preprocess.per`、`preprocess.noparallel`
- `train_common/data_common/model_common`：分别并入 `train/data/model`
- `paths`：改为顶层路径字段

## 预训练资源

- `pretrain/hubert/`：HuBERT 权重
- `pretrain/rmvpe/`：RMVPE 权重
- `pretrain/pretrained/`：v1 训练相关预训练权重
- `pretrain/pretrained_v2/`：v2 训练相关预训练权重

## 数据与产物

- `data/<name>/dataset/`：原始训练音频
- `data/<name>/preprocess_data/`：预处理输出，包含 `0_gt_wavs`、`1_16k_wavs`、`2a_f0`、`2b-f0nsf`、`3_feature256|768`、`filelist.txt` 和日志
- `ckpt/<name>/train/`：训练日志、TensorBoard、`G_*.pth`、`D_*.pth`
- `ckpt/<name>/export/`：导出的模型文件
- `ckpt/<name>/index/`：FAISS 索引和相关中间产物

## 核心代码

- `infer/modules/train/preprocess.py`：预处理入口。配置模式只接受 `--config`、`--hparams`、`--reset`；同时保留手工路径模式。
- `infer/modules/train/train.py`：训练主入口。
- `infer/lib/train/utils.py`：训练参数解析、快照写入、日志和 checkpoint 辅助函数。
- `tools/infer/train-index.py`：构建 v1 / 256-dim 特征索引。
- `tools/infer/train-index-v2.py`：构建 v2 / 768-dim 特征索引。

## 当前配置心智模型

- 用户主要编辑 `configs/base.yaml` 和 `configs/<task>.yaml`
- `work_dir` 是唯一实验根，`train_dir/export_dir/index_dir` 都由它派生
- `selectors` 负责选择 `version/sample_rate/if_f0`
- `runtime.is_half` 和 `train.fp16_run` 支持 `auto|true|false`
- preprocess/train/index 都通过同一个 resolver 获取最终配置
- 恢复训练时优先读取 `ckpt/<name>/config.yaml`
