# 项目文件说明

本文档按清理后的实际保留文件整理，用于说明仓库中每个文件的作用。

## 根目录

- `.env`：定义核心运行时默认环境变量，提供权重、索引、rmvpe 资源目录和线程设置。
- `.gitignore`：定义仓库忽略规则，避免训练产物、缓存和本地资源污染版本库。
- `CONTRIBUTING.md`：保留的贡献说明文档，记录项目合并和提交的基本约定。
- `LICENSE`：项目主许可证文件。
- `MIT协议暨相关引用库协议`：免责声明与第三方库许可引用清单。
- `PROJECT_FILE_GUIDE.md`：当前这份清理后文件总表。
- `README.md`：面向使用者的核心仓库说明，概述保留能力、安装方式和模型资源。
- `pyproject.toml`：Python 项目元数据和依赖声明，供 Poetry 类工具使用。
- `requirements.txt`：主环境依赖清单。
- `requirements-py311.txt`：面向 Python 3.11 的依赖清单。

## GitHub

- `.github/PULL_REQUEST_TEMPLATE.md`：拉取请求模板。
- `.github/workflows/pull_format.yml`：针对拉取请求执行格式检查的工作流。
- `.github/workflows/push_format.yml`：针对推送执行格式检查的工作流。
- `.github/workflows/sync_dev.yml`：同步开发分支的自动化工作流。
- `.github/workflows/unitest.yml`：核心训练链路的基础自动化测试工作流。

## assets

- `assets/hubert/.gitignore`：保留 `hubert` 资源目录结构的占位文件。
- `assets/indices/.gitignore`：保留索引资源目录结构的占位文件。
- `assets/pretrained/.gitignore`：保留主预训练模型目录结构的占位文件。
- `assets/pretrained_v2/.gitignore`：保留第二套预训练模型目录结构的占位文件。
- `assets/rmvpe/.gitignore`：保留 rmvpe 资源目录结构的占位文件。
- `assets/weights/.gitignore`：保留导出模型权重目录结构的占位文件。

## configs

- `configs/config.json`：历史遗留的本地配置样例，当前核心链路未直接使用。
- `configs/config.py`：运行时配置入口，负责设备选择、配置复制和推理切片参数初始化。
- `configs/inuse/.gitignore`：保留运行中配置目录结构的占位文件。
- `configs/inuse/v1/.gitignore`：保留 v1 运行配置子目录结构的占位文件。
- `configs/inuse/v2/.gitignore`：保留 v2 运行配置子目录结构的占位文件。
- `configs/v1/32k.json`：v1 32k 训练与推理配置模板。
- `configs/v1/40k.json`：v1 40k 训练与推理配置模板。
- `configs/v1/48k.json`：v1 48k 训练与推理配置模板。
- `configs/v2/32k.json`：v2 32k 训练与推理配置模板。
- `configs/v2/48k.json`：v2 48k 训练与推理配置模板。

## infer/lib

- `infer/lib/audio.py`：音频加载、重采样和格式转换辅助函数。
- `infer/lib/rmvpe.py`：rmvpe 音高提取核心实现，使用 eager PyTorch 模型推理。
- `infer/lib/slicer2.py`：音频切片工具。

## infer/lib/infer_pack

- `infer/lib/infer_pack/attentions.py`：推理模型用注意力模块实现。
- `infer/lib/infer_pack/commons.py`：模型公共数学工具和张量辅助函数。
- `infer/lib/infer_pack/models.py`：核心声学模型、解码器和判别器定义。
- `infer/lib/infer_pack/modules.py`：模型内部通用网络模块实现。
- `infer/lib/infer_pack/transforms.py`：谱域与特征变换辅助实现。

## infer/lib/infer_pack/modules/F0Predictor

- `infer/lib/infer_pack/modules/F0Predictor/__init__.py`：音高预测器子模块入口。
- `infer/lib/infer_pack/modules/F0Predictor/DioF0Predictor.py`：基于 `dio` 的音高预测实现。
- `infer/lib/infer_pack/modules/F0Predictor/F0Predictor.py`：音高预测器抽象基类。
- `infer/lib/infer_pack/modules/F0Predictor/HarvestF0Predictor.py`：基于 `harvest` 的音高预测实现。
- `infer/lib/infer_pack/modules/F0Predictor/PMF0Predictor.py`：基于 `parselmouth` 的音高预测实现。

## infer/lib/train

- `infer/lib/train/data_utils.py`：训练数据集、采样器和批处理拼装逻辑。
- `infer/lib/train/losses.py`：训练损失函数集合。
- `infer/lib/train/mel_processing.py`：梅尔谱和声学特征处理函数。
- `infer/lib/train/process_ckpt.py`：模型权重导出、合并、信息修改和轻量化处理工具。
- `infer/lib/train/utils.py`：训练配置读取、日志、检查点和可视化辅助函数。

## infer/modules/train

- `infer/modules/train/extract_feature_print.py`：提取 hubert 内容特征的脚本入口。
- `infer/modules/train/preprocess.py`：训练前的数据切片、重采样和清单生成脚本。
- `infer/modules/train/train.py`：主训练脚本。

## infer/modules/train/extract

- `infer/modules/train/extract/extract_f0_print.py`：基础音高提取脚本入口。
- `infer/modules/train/extract/extract_f0_rmvpe.py`：基于 rmvpe 的音高提取脚本入口。

## infer/modules/vc

- `infer/modules/vc/__init__.py`：变声模块包入口。
- `infer/modules/vc/modules.py`：高层变声封装，负责加载模型、单条推理和批量推理。
- `infer/modules/vc/pipeline.py`：变声主流水线，负责特征提取、索引融合、音高处理和音频拼接。
- `infer/modules/vc/utils.py`：变声相关辅助函数，包含索引路径查找和 hubert 加载逻辑。

## logs

- `logs/mute/0_gt_wavs/mute32k.wav`：测试夹具中的 32k 静音样本。
- `logs/mute/0_gt_wavs/mute40k.wav`：测试夹具中的 40k 静音样本。
- `logs/mute/0_gt_wavs/mute48k.wav`：测试夹具中的 48k 静音样本。
- `logs/mute/1_16k_wavs/mute.wav`：测试夹具中的 16k 静音样本。
- `logs/mute/2a_f0/mute.wav.npy`：测试夹具中的基础音高提取结果。
- `logs/mute/2b-f0nsf/mute.wav.npy`：测试夹具中的 rmvpe 音高提取结果。
- `logs/mute/3_feature256/mute.npy`：测试夹具中的 v1 特征提取结果。
- `logs/mute/3_feature768/mute.npy`：测试夹具中的 v2 特征提取结果。

## tools

- `tools/calc_rvc_model_similarity.py`：计算模型相似度的维护脚本。
- `tools/dlmodels.bat`：在 Windows 环境下载核心模型资源的批处理脚本。
- `tools/dlmodels.sh`：在类 Unix 环境下载核心模型资源的脚本。
- `tools/download_models.py`：通过 Python 下载核心模型资源的脚本。

## tools/infer

- `tools/infer/train-index-v2.py`：构建第二套索引格式的脚本。
- `tools/infer/train-index.py`：构建索引文件的脚本。
- `tools/infer/trans_weights.py`：转换模型权重结构的维护脚本。
