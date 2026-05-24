# RVC Core

这是一个仅保留 `CUDA/CPU` 核心链路的 `RVC` 仓库，提供训练、离线推理、模型处理和索引维护能力。

当前仓库统一使用 eager PyTorch，不再包含 TorchScript/JIT 导出、加载或兼容分支。

## 当前保留内容

- 训练链路：`infer/modules/train/*`
- 变声核心库：`infer/modules/vc/*`、`infer/lib/infer_pack/*`
- 模型处理：`infer/lib/train/process_ckpt.py`
- 维护脚本：`tools/download_models.py`、`tools/calc_rvc_model_similarity.py`、`tools/infer/train-index.py`、`tools/infer/train-index-v2.py`、`tools/infer/trans_weights.py`

## 环境准备

先安装 PyTorch，再安装项目依赖：

```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

## 模型资源

项目核心仍依赖以下资源：

- `assets/hubert/hubert_base.pt`
- `assets/rmvpe/rmvpe.pt`
- `assets/pretrained/`
- `assets/pretrained_v2/`

可以使用：

```bash
python tools/download_models.py
```

也可以使用：

```bash
tools/dlmodels.bat
tools/dlmodels.sh
```

## 参考项目

- [VITS](https://github.com/jaywalnut310/vits)
- [HIFIGAN](https://github.com/jik876/hifi-gan)
- [FFmpeg](https://github.com/FFmpeg/FFmpeg)
- [audio-slicer](https://github.com/openvpi/audio-slicer)
- [RMVPE](https://github.com/Dream-High/RMVPE)
