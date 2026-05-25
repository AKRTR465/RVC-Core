# `infer/` 目录结构与每个文件作用详解

本文只讲 `F:\deeplearning\SVC\RVC_rebuild\infer` 这一层，目标是把这个目录里每个源码文件在整条 RVC 流程中的位置、输入输出、关键类函数、以及和别的文件的关系讲清楚。

说明两点：

1. `infer/` 这个名字有历史包袱，它并不只是“推理代码”，而是当前仓库里训练、预处理、特征提取、索引构建、离线变声共用的主代码目录。
2. 下面默认覆盖 `rg --files infer` 能看到的源码/说明文件；`__pycache__/*.pyc` 这类 Python 运行时缓存文件不是手写业务代码，所以不逐个展开。

---

## 先看整体分层

从职责上看，`infer/` 大概分成 5 层：

- `infer/modules/train/`
  训练前和训练中的“入口脚本层”。包括预处理、F0 提取、HuBERT 特征提取、正式训练。
- `infer/modules/vc/`
  离线变声/推理运行时。核心是加载导出的 `.pth` 小模型，然后走 HuBERT 特征提取、可选索引检索、F0 提取、声码器生成。
- `infer/index/`
  FAISS 检索索引构建脚本，以及一个历史遗留的权重转换脚本。
- `infer/lib/train/`
  训练链路的底层库。负责 dataset、loss、mel 频谱、checkpoint、训练参数解析等。
- `infer/lib/infer_pack/` 与 `infer/lib/*.py`
  模型本体和通用音频/切片/RMVPE 模块。训练和推理两边都会用到。

可以把整个主链路理解成：

1. `preprocess.py` 把原始音频切片并生成训练采样率 wav + 16k wav
2. `extract_f0_*.py` 提取 F0，`extract_feature_print.py` 提取 HuBERT 特征
3. `train.py` 读取 `filelist.txt`、训练生成器/判别器、周期性导出小模型
4. `train-index.py` / `train-index-v2.py` 用特征构建 FAISS 索引
5. `modules/vc/modules.py` + `modules/vc/pipeline.py` 在推理时加载小模型和索引，完成离线变声

---

## 代码依赖速查

如果你想快速建立“谁调谁”的心智模型，可以先记这几条：

- `modules/train/preprocess.py`
  依赖 `infer.lib.audio.load_audio` 和 `infer.lib.slicer2.Slicer`
- `modules/train/extract_f0_print.py`
  依赖 `infer.lib.audio.load_audio`；统一封装 `pm`、`harvest`、`dio`、`rmvpe`，其中 RMVPE 默认自动选择 CUDA/CPU，并支持 GPU 分片模式
- `modules/train/extract_feature_print.py`
  直接加载 `pretrain/hubert/hubert_base.pt`，不走仓库内的封装
- `modules/train/train.py`
  依赖 `lib.train.utils/data_utils/losses/mel_processing/process_ckpt`，并从 `lib.infer_pack.models` 取核心模型
- `modules/vc/modules.py`
  负责组装推理：调用 `modules.vc.utils` 找模型/索引，调用 `modules.vc.pipeline.Pipeline` 真正做变声
- `modules/vc/pipeline.py`
  依赖 `faiss`、`torchcrepe`、`pyworld`、`parselmouth`、`infer.lib.rmvpe`
- `lib.train.process_ckpt.py`
  是训练大 checkpoint 转推理小模型 `.pth` 的关键桥梁
- `lib.infer_pack.models.py`
  定义训练和推理真正使用的 RVC 模型主体

---

## 每个文件逐个说明

## 1. `infer/` 根目录

### `infer/structure.md`

这是当前这份说明文档。

它的作用不是被程序调用，而是给人读的“目录地图”。最有价值的地方是把下面这些信息集中到一个地方：

- 目录分层怎么理解
- 每个入口脚本到底在什么阶段用
- 某个 `.py` 是“直接运行的脚本”还是“只被 import 的库”
- 训练产物、导出产物、索引产物分别从哪来

---

## 2. `infer/index/`

### `infer/index/train-index.py`

这是 v1 模型对应的索引构建脚本，面向 `256` 维 HuBERT 特征。

它在主链路中的位置是：

- 先有 `extract_feature_print.py` 产出的 `3_feature256/*.npy`
- 再由这个脚本把所有特征拼起来，训练 FAISS 索引
- 最后输出到 `ckpt/<name>/index/*.index`

核心职责：

- 解析两种运行方式
  - `--config` 模式：从项目配置里自动拿 `feature_dir`、`index_dir`、`final_index_path`
  - 手工模式：直接传 `--inp_root` 和 `--output`
- 校验项目是不是 v1/256 维
  - 如果 `project["feature_dim"] != 256`，直接报错，提示改用 `train-index-v2.py`
- 读取特征
  - 遍历 `inp_root` 下所有 `.npy`
  - 用 `np.concatenate` 拼成一个大矩阵 `big_npy`
- 保存中间产物
  - 把大矩阵写成 `big_src_feature.npy`
- 构建 FAISS 索引
  - 使用 `faiss.index_factory(256, f"IVF{n_ivf},Flat")`
  - `n_ivf` 规则是 `max(1, min(512, big_npy.shape[0]))`
  - `nprobe` 固定为 `9`
- 输出两个索引文件
  - `trained_<name>.index`：训练完成但还没 `add` 全量向量
  - 最终 `<name>.index`：已经 `add(big_npy)`，可直接用于检索

它的特点是实现很直接，没有做大规模数据压缩，也没有分批 `add()`；因为目标就是 v1 的 256 维特征场景。

### `infer/index/train-index-v2.py`

这是 v2 模型对应的索引构建脚本，面向 `768` 维特征。

和 `train-index.py` 相比，它更像“面向大数据量的增强版”。

主要职责：

- 同样支持 `--config` 模式和手工模式
- 校验项目必须是 `768` 维，否则报错
- 允许额外传 `--n_cpu`
  - 配置模式下默认取 `project["n_cpu"]`
  - 手工模式下默认取 `cpu_count()`

核心逻辑上的差异：

- 读取全部 `3_feature768/*.npy` 并拼接
- 对特征行做一次随机打乱，避免聚类和训练索引时过于偏序
- 当总帧数超过 `2e5` 时，尝试先做 `MiniBatchKMeans`
  - 目标压成 `10000` 个中心
  - 这是为了降低后续 FAISS 训练成本
  - 如果聚类失败，只会记 warning，不会中断整个流程
- `n_ivf` 的设定更激进
  - `min(int(16 * sqrt(N)), N // 39)`
- `nprobe` 设成 `1`
- `index.add()` 时按 `8192` 一批分批写入

这个文件可以理解成：

- v2 的特征维度更高、常见数据量更大
- 所以它在索引构建前多了一层“可选聚类压缩”

### `infer/index/trans_weights.py`

这是一个明显的历史/实验脚本，不属于当前主流程。

它做的事非常简单：

- 从硬编码的本地路径加载某个 `G_1000.pth`
- 取出其中的 `"model"` 权重字典
- 把所有张量 `.half()`
- 再保存成一个新的 `.pt`

为什么说它不是主流程的一部分：

- 路径直接写死成了开发者本机 `E:\...`
- 没有 CLI 参数，没有配置接入，没有项目目录推导
- 也不依赖仓库里的配置系统

它更像“当年为了临时做半精度权重导出写的单次脚本”，阅读价值主要在于提醒你：这个文件不要当成正式入口。

---

## 3. `infer/modules/train/`

### `infer/modules/train/preprocess.py`

这是训练数据预处理入口，负责把原始音频变成训练前的切片数据。

它的输出目录固定是：

- `0_gt_wavs/`
  训练采样率下的切片 wav
- `1_16k_wavs/`
  16k 重采样后的切片 wav，供 HuBERT 和 F0 提取使用
- `preprocess.log`
  预处理日志

核心组件：

- `init_log(preprocess_dir)`
  初始化日志文件句柄
- `println(message)`
  同时打印到终端和 `preprocess.log`
- `PreProcess`
  真正的预处理执行器

`PreProcess.__init__()` 里做了几件关键事：

- 创建 `Slicer`
  - `threshold=-42`
  - `min_length=1500ms`
  - `min_interval=400ms`
  - `hop_size=15ms`
  - `max_sil_kept=500ms`
- 设计一个 `48Hz` 的高通 Butterworth 滤波器
- 设置切片长度相关参数
  - `per`：每段目标长度，默认 `3.7s`
  - `overlap=0.3s`
  - `tail = per + overlap`
- 准备输出目录

`norm_write()` 是这个文件最值得注意的小函数：

- 先看当前片段绝对值峰值 `tmp_max`
- 如果峰值大于 `2.5`，直接记日志并过滤掉
- 否则做一层归一化混合
  - 不是单纯除以最大值
  - 而是按 `self.max=0.9` 和 `self.alpha=0.75` 做加权
- 先写训练采样率 wav，再重采样到 16k 写一份副本

`pipeline(path, idx0)` 的流程：

1. `load_audio()` 读音频到目标采样率
2. 高通滤波
3. 用 `Slicer.slice()` 按静音切段
4. 对每个大段再按 `per-overlap` 滑窗切成多个训练片段
5. 每片调用 `norm_write()`

`pipeline_mp_inp_dir()` 负责并行：

- 收集输入目录中所有文件
- 如果 `noparallel=True`，就在当前进程按分片顺序执行
- 否则按 `n_p` 启动多个 `multiprocessing.Process`

这个文件还保留了 3 套 CLI 兼容入口：

1. 新配置模式：`--config --hparams --reset`
2. 手工模式：`--inp_root --preprocess_dir --sample-rate --n_p --per`
3. 旧版位置参数模式：`inp_root sr n_p preprocess_dir noparallel per`

所以它既是当前配置系统入口，也是老脚本兼容层。

### `infer/modules/train/train.py`

这是正式训练入口，是整个 `infer/` 目录里最核心的脚本之一。

可以把它理解成“训练 orchestrator”：

- 启动多进程/分布式
- 选择正确的数据集类和模型类
- 恢复 checkpoint 或加载预训练
- 跑完整个对抗训练循环
- 周期性写 `G_*.pth` / `D_*.pth`
- 再把推理用的小模型导出到 `export/`

这个文件有一个重要特征：

- `hps = utils.get_hparams()` 在模块 import 阶段就执行了
- 也就是说，运行 `python train.py ...` 时，CLI 参数解析发生得非常早

主要部分：

#### `EpochRecorder`

一个很轻量的计时器，作用只是给 epoch 日志打上：

- 当前时间
- 上一个 epoch 到现在的耗时

#### `main()`

负责训练进程的总启动：

- 读取 GPU 数
- 如果没有 GPU，会打印一个 CPU fallback 提示，并把 `n_gpus` 强行置为 `1`
- 设置 `MASTER_ADDR` / `MASTER_PORT`
- 为每张卡启动一个 `mp.Process(target=run, args=(rank, ...))`

这里值得注意的点：

- 使用的是多进程 + `torch.distributed`
- backend 选的是 `gloo`，不是常见的 `nccl`
- 文件末尾强制 `torch.multiprocessing.set_start_method("spawn")`

#### `run(rank, n_gpus, hps, logger)`

这是每个 rank 进程里的训练初始化逻辑。

它做的事情很多：

- `rank==0` 时初始化 logger 和两个 `SummaryWriter`
  - 一个主训练日志
  - 一个 `eval/` 子目录 writer
- `dist.init_process_group(...)`
- 设随机种子
- 如果有 CUDA，绑定当前 rank 对应的 GPU
- 根据 `hps.if_f0` 选择 dataset / collate
  - 有 F0：`TextAudioLoaderMultiNSFsid` + `TextAudioCollateMultiNSFsid`
  - 无 F0：`TextAudioLoader` + `TextAudioCollate`
- 创建 `DistributedBucketSampler`
  - 目的是让同一个 batch 里的长度更接近，减小 padding 浪费
- 构造 `DataLoader`
  - `num_workers=4`
  - `persistent_workers=True`
  - `prefetch_factor=8`

模型选择逻辑：

- `version == "v1"`
  - 有 F0：`SynthesizerTrnMs256NSFsid`
  - 无 F0：`SynthesizerTrnMs256NSFsid_nono`
  - 判别器：`MultiPeriodDiscriminator`
- `version == "v2"`
  - 有 F0：`SynthesizerTrnMs768NSFsid`
  - 无 F0：`SynthesizerTrnMs768NSFsid_nono`
  - 判别器：`MultiPeriodDiscriminatorV2`

优化器和包装：

- G / D 都用 `AdamW`
- 再包上 `DistributedDataParallel`
- CUDA 下传 `device_ids=[rank]`
- CPU fallback 时就不指定 `device_ids`

恢复逻辑：

1. 优先尝试从 `train_dir` 下最新的 `D_*.pth` / `G_*.pth` 自动 resume
2. 如果失败，回退到 `hps.pretrainG` / `hps.pretrainD`
3. 成功 resume 后用 `epoch_str` 继续训练，并恢复 `global_step`

学习率与 AMP：

- G / D 都用 `ExponentialLR`
- 混合精度用 `GradScaler(enabled=hps.train.fp16_run)`

#### `train_and_evaluate(...)`

名字里有 evaluate，但当前实现本质上是“训练循环本体”，没有单独 eval dataloader。

它的职责：

- 设置 sampler epoch
- 切到 train mode
- 根据 `if_cache_data_in_gpu` 决定是否把 dataset cache 到 GPU
  - cache 模式第一次会把所有 batch 先搬到 GPU 再存到列表里
  - 后面 epoch 直接 shuffle 这个列表
- 遍历 batch

每个 step 的关键计算：

1. 解包 batch
2. 需要的话把 phone / pitch / spec / wave / sid 搬到 GPU
3. 跑生成器前向
   - 有 F0 时给 `phone + pitch + pitchf + spec + sid`
   - 无 F0 时不传 pitch 分支
4. 把真实 spec 转成 mel
5. 从真实 mel 和真实 wav 中切出和当前 segment 对应的片段
6. 判别器前向，算 `discriminator_loss`
7. 反向更新 D
8. 再跑一次判别器，算 G 侧损失
   - `loss_mel`
   - `loss_kl`
   - `feature_loss`
   - `generator_loss`
9. 反向更新 G

日志部分：

- 每 `log_interval` 打一次
- 记录学习率、grad norm、G/D 各项损失
- TensorBoard 里还会写 mel 图

checkpoint / 导出部分：

- 每 `save_every_epoch` 个 epoch 保存一次大 checkpoint
- `if_latest==1` 时会反复覆盖 `G_2333333.pth` / `D_2333333.pth`
- 如果 `save_every_weights=="1"`，还会调用 `savee()` 额外导出一个小模型到 `export/`

训练结束时：

- 如果到达 `hps.total_epoch`
- 记录 “Training is done”
- 再导出最终小模型
- `os._exit(2333333)` 强制退出

所以这个文件同时承担了：

- 训练调度
- checkpoint 管理
- TensorBoard 可视化
- 小模型导出触发

### `infer/modules/train/extract_feature_print.py`

这是 HuBERT 特征提取脚本。

它的输入是：

- `1_16k_wavs/*.wav`

输出是：

- v1 项目 -> `3_feature256/*.npy`
- v2 项目 -> `3_feature768/*.npy`

它是一个偏“worker”风格的脚本，不像 `preprocess.py` 那样接入了完整配置系统。

命令行参数形态比较历史化，有两种：

1. 新一点的形式
   - `device n_part i_part exp_dir version is_half`
2. 更老的形式
   - 多带一个 `i_gpu`
   - 并通过 `CUDA_VISIBLE_DEVICES` 绑定显卡

几个关键细节：

- 文件最开始虽然读了 `device = sys.argv[1]`
- 但后面又直接用 `device = "cuda" if torch.cuda.is_available() else "cpu"`
- 所以前面的 `device` 参数更多是历史兼容而不是现在真正的设备控制

主要流程：

1. 打开 `extract_f0_feature.log`
2. 从 `pretrain/hubert/hubert_base.pt` 加载 fairseq HuBERT 模型
3. 按分片规则 `sorted(...)[i_part::n_part]` 只处理自己负责的文件子集
4. `readwave()` 读取 16k wav
   - 可选做 `layer_norm`
5. 根据版本决定抽哪一层
   - `v1` -> `output_layer=9`，并走 `model.final_proj`
   - `v2` -> `output_layer=12`
6. 把输出特征保存成 `.npy`

它的作用可以概括成：

- 把原始 16k 音频变成训练/检索都要用的 HuBERT 帧级特征

### `infer/modules/train/extract_f0_print.py`

这是统一后的 F0 提取脚本，也是当前训练侧唯一保留的 F0 提取入口。

输入：

- `1_16k_wavs/*.wav`

输出：

- `2a_f0/*.npy`
  粗量化后的离散 F0，给带 F0 的模型做 pitch embedding
- `2b-f0nsf/*.npy`
  连续 F0 曲线，给 NSF 侧使用

关键对象是 `FeatureInput`。

#### `FeatureInput.__init__()`

定义了：

- 采样率 `16000`
- hop `160`
- F0 上下界
- 用于把 Hz 映射到 mel-pitch bin 的参数

#### `compute_f0(path, f0_method)`

根据 `f0_method` 切不同分支：

- `pm`
  - `parselmouth.Sound(...).to_pitch_ac(...)`
- `harvest`
  - `pyworld.harvest` + `pyworld.stonemask`
- `dio`
  - `pyworld.dio` + `pyworld.stonemask`
- `rmvpe`
  - 惰性加载 `infer.lib.rmvpe.RMVPE`
  - 默认自动选择 CUDA/CPU，并按 GPU 能力自动决定是否启用 half

#### `coarse_f0(f0)`

把连续 F0 映射到 `1..255` 的离散桶。

这一步对训练很重要，因为：

- 连续 F0 给 NSF 声源分支
- 离散 F0 给文本/特征编码器里的 pitch embedding

#### `go(paths, f0_method, log_path)`

单个进程负责自己的那部分文件：

- 已经存在的 `2a/2b` 文件会跳过
- 否则先存连续 F0，再存粗量化 F0

脚本现在统一支持两类调度方式：

1. 旧兼容入口
   - `python extract_f0_print.py <exp_dir> <workers> <f0method>`
   - 等价于 `--exp-dir ... --workers ... --f0method ...`
2. 新规范化入口
   - `--exp-dir`
   - `--f0method {pm,harvest,dio,rmvpe}`
   - `--workers`
   - `--n-part --i-part`
   - `--i-gpu`
   - `--is-half`

统一脚本内部有两种运行模式：

- 普通 worker 模式
  - 不提供 `--n-part/--i-part`
  - 按 `workers` 在脚本内部 fan-out 多进程
- 外部分片模式
  - 提供 `--n-part` 和 `--i-part`
  - 只处理 `paths[i_part::n_part]`
  - 更适合 RMVPE 在多卡/多进程调度器下分片运行

RMVPE 分支的行为也被统一进来了：

- 没提供 `--i-gpu`
  - 默认按仓库 runtime 风格自动选择设备
  - 有 CUDA 时优先用 `cuda:0`
  - 是否启用 half 会按 GPU 能力自动决定
- 提供了 `--i-gpu`
  - 会设置 `CUDA_VISIBLE_DEVICES`
  - 仍然优先走 CUDA
  - half 也按可见 GPU 的能力自动决定；显式 `--is-half` 只在 GPU 支持时才会生效

非 RMVPE 方法会接受这些参数，但只记录“已忽略”，不会改变 `pm`、`harvest`、`dio` 的原始行为。

所以这个文件现在既覆盖了原来通用版 F0 提取脚本的多进程逻辑，也吸收了原来 RMVPE 专用入口的 GPU 分片运行方式。

---

## 4. `infer/modules/vc/`

### `infer/modules/vc/__init__.py`

这是空的包标记文件。

它本身没有业务逻辑，作用只是让 `infer.modules.vc` 作为 Python package 被导入。

### `infer/modules/vc/utils.py`

这个文件是推理阶段的辅助工具集合，主要做三件事：

1. 找模型文件
2. 找索引文件
3. 加载 HuBERT

#### `_iter_paths(root, suffix)`

- 递归遍历目录下的指定后缀文件
- 返回排序后的 `Path` 列表

#### `get_model_path_from_sid(sid, ckpt_root)`

这是“根据 sid 找模型”的核心逻辑。

它支持几种情况：

- `sid` 本身就是一个存在的绝对路径
- `ckpt_root / sid` 这个直连路径存在
- 否则递归扫描 `ckpt_root` 下所有 `.pth`
  - 排除 `G_*.pth` / `D_*.pth` 这类训练大 checkpoint
  - 优先匹配文件名或 stem
  - 如果匹配结果里有 `export/` 目录下的文件，优先选 `export/`

这说明推理阶段默认更希望加载“导出后的小模型”，不是训练中的大 checkpoint。

#### `get_index_path_from_model(sid, ckpt_root)`

根据模型反推索引路径：

- 如果模型在 `.../export/xxx.pth`
- 就优先去同项目的 `.../index/` 里找
- 跳过 `trained_*.index`
- 更偏向返回最终可用索引

#### `load_hubert(config)`

加载 HuBERT 推理模型：

- 默认从 `pretrain/hubert/hubert_base.pt`
- 也允许 `config.hubert_path` 覆盖
- 加载后放到 `config.device`
- 根据 `config.is_half` 决定 half/float

这个函数在推理里非常关键，因为 HuBERT 是把输入音频转成语音内容特征的第一步。

### `infer/modules/vc/modules.py`

这是推理侧的总控文件，核心类是 `VC`。

它的职责不是直接做声学推理，而是：

- 管理当前加载的模型状态
- 对接外部 UI/调用层
- 为单文件和批量推理提供统一入口

#### `VC.__init__(config)`

初始化一堆运行时状态：

- 当前 speaker 数
- 目标采样率
- 已加载的生成器
- `Pipeline`
- checkpoint 元信息
- `version`
- `if_f0`
- `hubert_model`

#### `get_vc(sid, *to_return_protect)`

这是“切换模型/加载模型”的入口。

它分两种大分支：

##### 1. `sid` 为空

表示要清空当前模型缓存。

它会：

- 删除 `net_g`、`hubert_model` 等对象
- 清空 CUDA cache
- 临时重新构造一次模型类后再删掉
  - 这是为了尽量让显存释放得更彻底

返回值还是明显的“UI 风格”：

- `{"visible": ..., "__type__": "update"}`

说明这个文件原本就是为某个图形界面/网页界面交互层服务的。

##### 2. `sid` 非空

表示要加载某个说话人/模型。

流程：

1. `get_model_path_from_sid()` 找到模型路径
2. `torch.load()` 读小模型 `.pth`
3. 从 checkpoint 元信息里解析：
   - 目标采样率 `tgt_sr`
   - 是否带 F0 `if_f0`
   - 版本 `version`
   - speaker embedding 数量 `n_spk`
4. 按 `(version, if_f0)` 选择正确的 `Synthesizer...` 类
5. `del self.net_g.enc_q`
   - 推理不需要 posterior encoder，所以删掉省内存
6. 加载权重、切到 eval、搬到目标 device、根据 `is_half` 切 half/float
7. 创建 `Pipeline(self.tgt_sr, self.config)`
8. `get_index_path_from_model()` 找默认索引

所以 `get_vc()` 本质是：

- 从“模型名/路径” -> “已经准备好可推理的生成器 + 推理管线”

#### `vc_single(...)`

单文件推理入口。

主要流程：

1. 用 `load_audio()` 把输入音频读成 16k 单声道
2. 如果峰值过高先整体缩放
3. 第一次推理时懒加载 HuBERT
4. 规范化 `file_index`
   - 会把历史字符串里的 `trained` 自动替换成 `added`
   - 即使当前项目实际常用的是最终 `.index`，这里仍保留了历史兼容逻辑
5. 调 `self.pipeline.pipeline(...)`
6. 根据是否重采样决定最终输出采样率
7. 组装状态字符串，返回 `(采样率, 音频数组)`

这里的 `times = [0,0,0]` 用来统计：

- `times[0]`：HuBERT / 特征相关耗时
- `times[1]`：F0 提取耗时
- `times[2]`：生成器推理耗时

#### `vc_multi(...)`

批量推理入口。

它做的事情是：

- 解析目录模式或上传文件列表模式
- 逐个调用 `vc_single()`
- 成功后把输出保存成指定格式
  - `wav` / `flac` 直接 `soundfile.write`
  - 其他格式先写到 `BytesIO` 中转，再交给 `wav2()`
- 每处理完一个文件，就 `yield` 一次累计状态

因此它不是一次性返回结果，而是“边跑边汇报”的生成器接口。

### `infer/modules/vc/pipeline.py`

这是离线变声真正的核心流水线文件。

如果说 `modules.py` 负责“装配和入口”，那这个文件负责“具体怎么把一段输入音频变成目标音频”。

文件顶层的两个辅助函数：

#### `cache_harvest_f0(...)`

- 对 `harvest` F0 结果做 `lru_cache`
- key 里包含 `input_audio_path`
- 底层实际音频数据来自全局字典 `input_audio_path2wav`

这说明作者希望：

- 同一输入文件反复跑 harvest 时，尽量复用结果

#### `change_rms(data1, sr1, data2, sr2, rate)`

- 计算输入音频和输出音频的 RMS 包络
- 通过插值对齐时间轴
- 再按 `rate` 混合响度

这个函数的用途是：

- 让变声后的音量起伏尽量贴近原始输入

#### `Pipeline.__init__(tgt_sr, config)`

初始化推理时各种窗口、padding 和切片参数：

- `x_pad`
- `x_query`
- `x_center`
- `x_max`
- `is_half`

还会派生出：

- `self.sr = 16000`
  - 因为 HuBERT 输入固定 16k
- `self.window = 160`
  - 对应每帧 hop
- `self.t_pad` / `self.t_pad_tgt`
- `self.t_query`
- `self.t_center`
- `self.t_max`

并记录：

- `device`
- `rmvpe_path`

这些参数共同决定了：

- 长音频如何分段
- 每段前后保留多少上下文
- 推理时如何避免切段边界爆音/接缝明显

#### `get_f0(...)`

这是推理阶段的 F0 入口。

支持 4 条分支：

- `pm`
- `harvest`
- `crepe`
- `rmvpe`

细节上：

- `harvest` 分支会把音频先放进全局缓存字典，配合 `cache_harvest_f0()`
- `filter_radius > 2` 时会对 harvest 结果做中值滤波
- `crepe` 分支直接调用 `torchcrepe.predict(...)`
- `rmvpe` 分支会懒加载 `infer.lib.rmvpe.RMVPE`

后处理部分：

- 统一做 `f0_up_key` 半音移调
- 如果外部传了 `inp_f0`，会把用户给的 F0 文件插值后覆写到一段区域
- 最后同时返回：
  - `f0_coarse`：离散桶
  - `f0bak`：连续 F0

这和训练阶段一样，是为了同时服务两个分支：

- 文本/特征编码器的离散 pitch embedding
- NSF 声源的连续 F0

#### `vc(...)`

这是单个音频片段的实际声学转换核心。

它做的事情可以分成 6 步：

1. 把 `audio0` 变成 torch tensor
2. 调 HuBERT 提取内容特征
   - `v1` 用 `output_layer=9` 且走 `final_proj`
   - `v2` 用 `output_layer=12`
3. 可选走索引检索特征融合
   - `faiss.index.search(k=8)`
   - 用距离倒数平方做权重
   - 从 `big_npy[ix]` 里取近邻后加权平均
   - 再与原始 HuBERT 特征按 `index_rate` 混合
4. 特征时间轴插值放大 2 倍
5. 如果 `protect < 0.5` 且带 pitch，则用 `protect` 混合一份“未改写特征”
   - 这是为了保护清辅音/无声段，减少音色塌陷
6. 调 `net_g.infer(...)` 生成目标音频片段

最后它会：

- 更新计时统计
- 清理临时 tensor 和 CUDA cache

#### `pipeline(...)`

这是处理整段输入音频的总流程。

主要步骤：

1. 如有有效 `file_index` 且 `index_rate != 0`
   - `faiss.read_index(file_index)`
   - 直接 `index.reconstruct_n(0, index.ntotal)` 把全部向量重建到内存
   - 所以后续不依赖单独读 `big_src_feature.npy`
2. 输入音频先过高通滤波
3. 前后做 reflect padding
4. 如果音频很长，就按局部能量最低点找切点 `opt_ts`
5. 读取外部 F0 文件（若有）
6. 如果模型带 F0，先整段提一遍 pitch / pitchf
7. 按切点逐段调用 `vc(...)`
8. 每段裁掉 pad 后拼接回整段
9. 如果 `rms_mix_rate != 1`，做响度包络混合
10. 如果 `resample_sr` 合法且不同于模型采样率，再重采样
11. 归一化到 int16

可以说，这个函数把推理时真正复杂的部分都包进去了：

- 长音频切块
- F0 提取
- 索引检索
- 片段拼接
- 响度对齐
- 重采样

---

## 5. `infer/lib/`

### `infer/lib/audio.py`

这是训练和推理都会用到的音频 I/O 工具。

它主要有 3 个函数：

#### `wav2(i, o, format)`

使用 `PyAV` 做格式转换：

- 输入可以是文件对象/字节流
- 输出写到目标文件对象
- 对 `m4a`、`ogg`、`mp4` 做了 codec 兼容映射

这个函数在 `vc_multi()` 里很有用，因为推理输出不一定只要 wav。

#### `load_audio(file, sr)`

这是整个项目里最常被调用的音频加载函数之一。

特点：

- 先走 `clean_path()` 清路径
- 检查文件是否存在
- 再通过 `ffmpeg-python` 调 `ffmpeg` CLI
- 直接解码成：
  - 单声道
  - `float32`
  - 目标采样率 `sr`

返回值是一个 `np.float32` 一维数组。

它的好处是：

- 对各种格式都比较稳
- 把重采样和单声道化统一了

#### `clean_path(path_str)`

一个很实用的“小白输入容错”函数：

- Windows 下把 `/` 替成 `\`
- 去掉 Unicode 控制字符
- 去掉首尾空格、引号、换行

### `infer/lib/slicer2.py`

这是静音切片器，`preprocess.py` 的核心依赖之一。

文件分成三部分：

#### `get_rms(...)`

- 从 `librosa` 改来的 RMS 计算逻辑
- 按 frame/hop 取音量包络

#### `Slicer`

这是真正的静音切片器。

构造函数接受的是“更偏人类可读”的毫秒参数，然后统一换算到样本/帧尺度：

- `threshold`
- `min_length`
- `min_interval`
- `hop_size`
- `max_sil_kept`

`slice(waveform)` 的算法要点：

- 先转单声道做 RMS 分析
- 扫描每一帧，记录静音开始/结束位置
- 结合：
  - 片段最短长度
  - 静音最短长度
  - 可保留的最大静音长度
- 决定是切掉前导静音、切中间静音，还是保留一定静音缓冲

这不是“简单按阈值直接断开”，而是相对细致地处理了：

- 前导静音
- 中间长静音
- 尾部静音

#### `main()`

这是一个独立 CLI，可单独拿某个音频文件做静音切片测试。

所以 `slicer2.py` 既能被 `preprocess.py` 调，也能单独命令行调试。

### `infer/lib/rmvpe.py`

这是 RMVPE 的完整实现文件，也是 `infer/lib/` 里最重的模型文件之一。

它既包含神经网络定义，也包含实际推理封装。

可以把它理解成 4 层：

#### 1. 频谱层：`STFT` 与 `MelSpectrogram`

`STFT`：

- 用卷积/反卷积思路封装 STFT 与 iSTFT
- 提供 `transform()` / `inverse()` / `forward()`

`MelSpectrogram`：

- 基于 `torch.stft`
- 缓存 mel basis 和 hann window
- 支持 `keyshift` 和 `speed`

虽然当前项目里 RMVPE 调用时基本走默认参数，但这个 mel 提取器本身设计得比“只做普通 mel”更通用。

#### 2. 编码器-解码器骨架

这些类共同组成 RMVPE 的 U-Net 风格主干：

- `BiGRU`
- `ConvBlockRes`
- `Encoder`
- `ResEncoderBlock`
- `Intermediate`
- `ResDecoderBlock`
- `Decoder`
- `DeepUnet`
- `E2E`

结构含义：

- 先把 mel 送入 2D 卷积 U-Net
- 再经过 CNN + GRU + Linear
- 输出每帧 360 个 pitch salience logits

#### 3. 推理封装：`RMVPE`

这是外部真正会实例化的类。

`__init__()`：

- 处理 device
- 创建 `MelSpectrogram`
- 加载 `rmvpe.pt`
- 构建 `cents_mapping`

`mel2hidden()`：

- 把 mel pad 到 32 的倍数长度
- 跑神经网络

`decode()`：

- 调 `to_local_average_cents()`
- 再把 cents 映射回 Hz

`infer_from_audio()`：

- 输入原始音频波形
- 提 mel
- 跑模型
- salience -> F0
- 返回连续 F0 曲线

#### 4. 后处理：`to_local_average_cents()`

作用是：

- 在 360 个 pitch bins 中找到峰值附近的局部窗口
- 做加权平均，得到更平滑的 cents 结果
- 再按阈值把低置信度帧置零

这个文件会被两类地方使用：

- 训练前 F0 提取脚本
- 推理时 `f0_method="rmvpe"` 分支

---

## 6. `infer/lib/train/`

### `infer/lib/train/data_utils.py`

这是训练数据读取层，核心是“把 filelist 里的路径和特征拼成训练 batch”。

它实际上维护了两套 dataset / collate：

- 带 F0
- 不带 F0

#### `TextAudioLoaderMultiNSFsid`

用于带 F0 的训练。

它假定 `training_files` 里的每一行格式是：

- `audiopath|phone|pitch|pitchf|sid`

主要行为：

- `load_filepaths_and_text()` 读文件清单
- `_filter()`
  - 根据 `min_text_len/max_text_len` 过滤
  - 估计每条样本长度，供 bucket sampler 分桶
- `get_labels(phone, pitch, pitchf)`
  - 分别 `np.load()` 三个特征文件
  - `phone` 在时间维 `repeat(2, axis=0)`
    - 这是为了让 HuBERT 特征长度和目标训练帧率更匹配
  - 最多截到 `900` 帧
- `get_audio(filename)`
  - 读取 wav
  - 计算 spectrogram
  - 并把结果缓存到旁边的 `.spec.pt`
- `get_audio_text_pair(...)`
  - 保证 `phone`、`pitch`、`pitchf`、`spec`、`wav` 长度对齐

#### `TextAudioCollateMultiNSFsid`

负责把带 F0 的单条样本整理成 batch：

- 按 spec 长度降序排序
- pad 出：
  - `phone_padded`
  - `pitch_padded`
  - `pitchf_padded`
  - `spec_padded`
  - `wave_padded`
- 同时返回各自长度和 `sid`

#### `TextAudioLoader`

用于不带 F0 的训练。

它假定 `training_files` 每一行格式是：

- `audiopath|phone|sid`

和上面的区别主要是：

- 不读 `pitch` 和 `pitchf`
- `get_labels()` 只读 `phone`
- `get_audio_text_pair()` 只对齐 `phone/spec/wav`

#### `TextAudioCollate`

不带 F0 的 batch 整理版本。

只 pad：

- `phone`
- `spec`
- `wave`
- `sid`

#### `DistributedBucketSampler`

这个类很重要，它决定了 batch 是怎么按长度组织的。

作用：

- 按样本长度分桶
- 每个 batch 尽量由长度相近的样本组成
- 多卡下保证每个 replica 拿到可均匀切分的样本数

实现特点：

- 落不到 boundaries 区间内的样本会被丢弃
- 每个桶会补齐到 `num_replicas * batch_size` 的整数倍
- 每个 epoch 会按种子重新 shuffle

### `infer/lib/train/losses.py`

这个文件很小，但训练时每步都会进来。

定义了 4 个损失：

#### `feature_loss(fmap_r, fmap_g)`

- 判别器各层 feature map 的 L1 差
- 常见于 HiFi-GAN / GAN 声码器训练

#### `discriminator_loss(disc_real_outputs, disc_generated_outputs)`

- LSGAN 风格
- 真实样本希望靠近 `1`
- 生成样本希望靠近 `0`

#### `generator_loss(disc_outputs)`

- 生成器希望判别器输出靠近 `1`

#### `kl_loss(z_p, logs_q, m_p, logs_p, z_mask)`

- 训练 latent flow / posterior-prior 对齐时用的 KL 项

所以这个文件本质上是：

- 对抗损失
- feature matching
- 变分/flow 相关 KL

### `infer/lib/train/mel_processing.py`

这是训练侧的频谱工具库。

主要职责：

- 生成线性谱
- 生成 mel 谱
- 做 log 压缩/反压缩
- 缓存 mel basis 和 hann window

关键函数：

#### `dynamic_range_compression_torch` / `dynamic_range_decompression_torch`

- 对谱幅度做对数压缩与还原

#### `spectral_normalize_torch` / `spectral_de_normalize_torch`

- 语义上更贴近“频谱归一化”的包装

#### `spectrogram_torch(...)`

- 输入波形
- 输出线性幅度谱
- 使用缓存的 `hann_window`

#### `spec_to_mel_torch(...)`

- 线性谱 -> mel log 谱
- 使用缓存的 `mel_basis`

#### `mel_spectrogram_torch(...)`

- 直接从波形得到 mel log 谱

训练里它主要被 `train.py` 用来：

- 算真实 mel
- 算生成 mel
- 再做 `L1` 的 mel loss

### `infer/lib/train/process_ckpt.py`

这是训练产物后处理文件，非常关键，因为推理侧加载的不是训练大 checkpoint，而是这里导出的“小模型”格式。

#### `_guess_project_name(name)`

- 如果文件名形如 `xxx_e10_s1000`
- 会把项目名猜成 `xxx`

#### `_export_path(name, suffix=".pth", hps=None)`

统一决定导出路径：

- 如果传了 `hps.export_dir`，优先用它
- 否则退回到 `ckpt_root/<project>/export/`

#### `savee(ckpt, sr, if_f0, name, epoch, version, hps)`

这是训练结束/中途导出小模型的核心函数。

它做的事情：

- 新建 `OrderedDict`
- 把所有权重拷到 `opt["weight"]`
- 跳过 `enc_q` 相关参数
  - 因为推理不需要 posterior encoder
- 把权重转成 half
- 构建 `opt["config"]`
  - 这是一个位置参数列表
  - 顺序必须和推理时 `Synthesizer...(*config)` 构造函数一致
- 写入元信息：
  - `info`
  - `sr`
  - `f0`
  - `version`
- 最后保存到 export 目录

可以说，`VC.get_vc()` 之所以能直接从一个小 `.pth` 重建模型，靠的就是这里打包进去的 `config + weight + meta`。

#### `show_info(path)`

- 读取小模型中的元信息
- 返回可读字符串

#### `extract_small_model(path, name, sr, if_f0, info, version)`

作用也是导小模型，但它不是从训练时内存里的 `state_dict` 出发，而是从现成 checkpoint 文件出发。

特点：

- 如果 `ckpt` 里有 `"model"`，就先解包
- 同样会剔除 `enc_q`
- 同样转 half
- 但模型结构 `config` 不是动态推导，而是按 `sr + version` 用硬编码模板拼出来

所以这个函数更像：

- 对已有 checkpoint 进行“离线补导出”

#### `change_info(path, info, name)`

- 修改小模型里的 `info` 字段
- 再另存一份

#### `_normalize_f0_flag(f0)`

- 把各种 bool/int/str 风格的 F0 开关统一成 `0/1`

#### `merge(path1, path2, alpha1, sr, f0, info, name, version)`

模型合并工具。

流程：

- 读取两个模型
- 如果是训练 checkpoint，就抽出真正权重
- 对同名参数做线性插值
- `emb_g.weight` 如果 speaker 数不一致，只取较小公共部分
- 再按小模型格式写回

这个文件的意义是：

- 它把“训练态大权重”和“推理态小权重”连接起来了
- 也是模型信息查看、修改、合并的工具层

### `infer/lib/train/utils.py`

这是训练链路最杂、也最重要的工具箱文件之一。

它可以分成 5 类职责。

#### 1. checkpoint 读写

- `load_checkpoint_d(...)`
  给某些组合判别器结构用的 checkpoint 恢复器
- `load_checkpoint(...)`
  当前训练主流程实际使用的恢复器
- `save_checkpoint(...)`
- `save_checkpoint_d(...)`
- `latest_checkpoint_path(...)`
  取目录里数字最大的 checkpoint 文件

这些函数都有一个共同特点：

- 对 shape 不一致或缺失 key 采取“尽量兼容加载”的策略
- 不会像严格 `load_state_dict(strict=True)` 那样直接炸掉

#### 2. TensorBoard 和可视化辅助

- `summarize(...)`
  往 `SummaryWriter` 写 scalars / histograms / images / audios
- `plot_spectrogram_to_numpy(...)`
- `plot_alignment_to_numpy(...)`

其中 `plot_*` 是为了把频谱图直接变成 numpy 图像，再喂给 TensorBoard。

#### 3. 基础 I/O

- `load_wav_to_torch(full_path)`
- `load_filepaths_and_text(filename, split="|")`

`load_filepaths_and_text()` 还处理了 utf-8 解码失败的回退逻辑。

#### 4. 配置系统与训练 CLI 粘合层

这部分是这个文件在当前仓库里的真正核心价值。

##### `_first_not_empty()` / `_required_value()`

- 负责把 CLI 和配置值合并时的优先级梳理清楚

##### `_normalize_save_every_weights()`

- 把 `true/1/yes/on` 之类统一成 `"1"` 或 `"0"`

##### `_sync_train_aliases(config)`

- 把新配置系统里的 `train.*` 字段同步成老训练代码习惯读取的顶层别名
- 比如：
  - `config["save_every_epoch"]`
  - `config["total_epoch"]`
  - `config["pretrainG"]`
  - `config["pretrainD"]`

也就是说，这个函数其实是“新配置系统对老训练脚本的适配层”。

##### `_apply_training_cli_overrides(config, args)`

- 把 CLI 里的 `batch_size / total_epoch / pretrainG / if_latest / ...`
- 覆盖到当前项目配置中
- 同时还会更新 `replayable_config["train"]`

##### `_snapshot_path_for_project(config)`

- 统一决定配置快照保存到 `work_dir/config.yaml`

##### `_build_hparams(config)`

- 把 dict 包装成 `HParams`
- 并补上训练代码真正要读的别名
  - `model_dir = train_dir`
  - `experiment_dir = work_dir`
  - `data.training_files = training_files`

##### `get_hparams(init=True)`

这是训练入口最重要的入口函数之一。

作用：

- 解析训练 CLI
- 强制要求 `--config`
- 读取项目配置
- 应用 `--hparams`
- 应用训练 CLI override
- 必要时写 `work_dir/config.yaml` 快照
- 最后返回 `HParams`

所以 `train.py` 虽然看起来没自己处理很多配置，其实大量工作都在这里做完了。

##### `get_hparams_from_dir(model_dir)` / `get_hparams_from_file(config_path)`

- 从已存在实验目录或配置文件反构 `HParams`

#### 5. 日志与元信息

- `check_git_hash(model_dir)`
  尝试把当前 git hash 写到实验目录
- `get_logger(model_dir, filename="train.log")`
  返回训练日志 logger
- `HParams`
  一个递归 dict-to-attribute 包装类
  让训练代码可以写 `hps.train.batch_size` 这种形式

这个文件在整个训练链路中的定位可以概括为：

- “训练脚本和配置系统之间的桥”
- “checkpoint/tensorboard/日志的工具箱”

---

## 7. `infer/lib/infer_pack/`

这一层是模型本体。很多文件不是直接运行脚本，而是被训练与推理共同 import。

### `infer/lib/infer_pack/commons.py`

这是纯函数工具库，主要服务模型内部张量操作。

关键函数可按类型理解：

#### 初始化和卷积辅助

- `init_weights(m, mean, std)`
- `get_padding(kernel_size, dilation)`

#### 概率/采样辅助

- `kl_divergence(...)`
- `rand_gumbel(shape)`
- `rand_gumbel_like(x)`

#### segment 切片

- `slice_segments(x, ids_str, segment_size)`
- `slice_segments2(x, ids_str, segment_size)`
- `rand_slice_segments(x, x_lengths, segment_size)`

这些函数在训练里非常关键，因为 `train.py` 每个 step 都会从长序列里随机切出一个 segment 给生成器/判别器。

#### 时序位置编码

- `get_timing_signal_1d(...)`
- `add_timing_signal_1d(...)`
- `cat_timing_signal_1d(...)`

#### mask 和路径

- `subsequent_mask(length)`
- `sequence_mask(length, max_length)`
- `generate_path(duration, mask)`

#### gating / padding / shift

- `fused_add_tanh_sigmoid_multiply(...)`
- `convert_pad_shape(...)`
- `shift_1d(x)`

#### 梯度处理

- `clip_grad_value_(parameters, clip_value, norm_type)`

这个文件的特点是：

- 没有任何“业务入口”
- 但很多模型层和训练逻辑都会反复调这里的小工具

### `infer/lib/infer_pack/transforms.py`

这个文件实现的是 normalizing flow 里常见的 rational quadratic spline 变换。

主要函数：

- `piecewise_rational_quadratic_transform(...)`
- `searchsorted(...)`
- `unconstrained_rational_quadratic_spline(...)`
- `rational_quadratic_spline(...)`

它在当前项目里的位置是：

- 为 `modules.py` 里的 `ConvFlow` 提供可逆单调变换

虽然当前主模型 `models.py` 主要用的是 `ResidualCouplingBlock`
而不是显式使用 `ConvFlow`，但这个文件依然是完整 flow 模块库的一部分。

### `infer/lib/infer_pack/attentions.py`

这是 Transformer/Attention 相关模块文件。

当前项目里最直接用到的是 `TextEncoder` 内部依赖的 `attentions.Encoder`，但这个文件本身提供了比当前主链路更完整的一套部件。

#### `Encoder`

- 多层堆叠的自注意力编码器
- 每层包含：
  - `MultiHeadAttention`
  - `LayerNorm`
  - `FFN`
  - dropout

当前 RVC 模型里的内容特征编码器 `TextEncoder` 主要就靠它来处理 HuBERT 特征和 pitch embedding。

#### `Decoder`

- 标准风格的 decoder 结构
- 含：
  - masked self-attention
  - encoder-decoder attention
  - causal FFN

在当前这套 RVC 主模型里没有像 `Encoder` 那样直接成为训练主路径核心，但它保留了完整的 seq2seq 组件。

#### `MultiHeadAttention`

这是整个文件最核心的底层层。

支持的能力包括：

- 常规 Q/K/V 多头注意力
- 相对位置窗口 `window_size`
- `heads_share`
- `block_length`
- `proximal_bias`
- `proximal_init`

也就是说，它不仅能做普通 self-attention，还带了一些更适合语音序列局部相关性的增强机制。

#### `FFN`

- 1D 卷积版前馈网络
- 支持 `causal=True`
- 支持自带 padding 逻辑

### `infer/lib/infer_pack/modules.py`

这是更底层的神经网络积木文件，很多模型类都会直接 import 它。

可以按模块类型理解：

#### 规范化和浅层卷积块

- `LayerNorm`
  通道维 layer norm
- `ConvReluNorm`
  多层 conv + relu + dropout，再接一个投影残差
- `DDSConv`
  Dilated Depth-Separable Convolution

#### WaveNet 风格编码块

- `WN`
  带条件输入的 gated dilated conv 堆叠
  当前 `PosteriorEncoder` 和 `ResidualCouplingLayer` 都会用到

#### 声码器残差块

- `ResBlock1`
- `ResBlock2`

这两个类是 `Generator` / `GeneratorNSF` 上采样后做细化的关键模块。

#### 可逆流变换基础块

- `Log`
  对数域变换
- `Flip`
  通道翻转
- `ElementwiseAffine`
  仿射变换
- `ResidualCouplingLayer`
  当前主模型 flow 的核心单元
- `ConvFlow`
  基于 spline 的另一种 flow 实现

其中当前主模型最直接重度使用的是：

- `WN`
- `ResBlock1/2`
- `Flip`
- `ResidualCouplingLayer`

而 `ConvFlow` 更像保留在模块库里的可选 flow 构件。

### `infer/lib/infer_pack/models.py`

这是整个仓库最核心的模型定义文件。

训练和推理用到的“真正模型主体”基本都在这里。

可以按从输入到输出的路径来理解。

#### `TextEncoder`

作用：

- 读取 HuBERT 特征 `phone`
- 如果带 F0，就再加一份 `pitch` embedding
- 经 `attentions.Encoder` 编码
- 最后投影成潜变量先验分布的 `m, logs`

要点：

- `v1` 模型的输入特征维度是 `256`
- `v2` 模型的输入特征维度是 `768`
- `skip_head` 参数用于分段推理时裁前缀

#### `ResidualCouplingBlock`

作用：

- 由多层 `ResidualCouplingLayer + Flip` 组成
- 构成模型里的 normalizing flow 主体

在训练里：

- 把 posterior latent 往 prior 空间推

在推理里：

- 反向 `reverse=True` 从 prior 采样生成可解码 latent

#### `PosteriorEncoder`

作用：

- 把真实目标 spectrogram 编码成 posterior latent `z`
- 同时输出 posterior 的 `m_q, logs_q`

这只在训练阶段需要，推理时会被删掉。

#### `Generator`

这是不带 F0 条件的基础声码器。

结构上：

- `conv_pre`
- 多级反卷积上采样
- 每级后接多个 `ResBlock`
- `conv_post`
- `tanh` 输出波形

主要给无 F0 (`nono`) 版本用。

#### `SineGen`

作用：

- 根据连续 F0 合成正弦激励
- 还能区分 voiced/unvoiced，并加噪声

这是 NSF 声源的基础。

#### `SourceModuleHnNSF`

作用：

- 把 `SineGen` 产生的谐波/噪声源进一步合成成单通道 excitation

#### `GeneratorNSF`

这是带 F0 条件的声码器版本。

相比普通 `Generator`，它多了：

- `m_source`：从 F0 生成谐波源
- `noise_convs`：把 excitation 注入到不同上采样层

所以：

- 带 F0 模型靠它生成波形
- 不带 F0 模型才走普通 `Generator`

#### `sr2sr`

一个小映射表：

- `"32k" -> 32000`
- `"40k" -> 40000`
- `"48k" -> 48000`

#### `SynthesizerTrnMs256NSFsid`

这是 v1、有 F0 的主模型类。

组成部分：

- `enc_p = TextEncoder(256, ...)`
- `dec = GeneratorNSF(...)`
- `enc_q = PosteriorEncoder(...)`
- `flow = ResidualCouplingBlock(...)`
- `emb_g = nn.Embedding(spk_embed_dim, gin_channels)`

训练时 `forward(...)`：

1. 说话人 id -> `emb_g`
2. `enc_p` 从 `phone + pitch` 得到先验分布
3. `enc_q` 从真实谱得到 posterior latent
4. `flow` 把 posterior latent 映射到 prior 空间
5. 随机切一个 latent segment
6. 切对应的 `pitchf`
7. `dec` 生成音频段

推理时 `infer(...)`：

1. 只走 `enc_p`
2. 从先验采样 `z_p`
3. `flow(reverse=True)` 反推得到解码 latent
4. `dec(z, nsff0, g=...)` 生成波形

#### `SynthesizerTrnMs768NSFsid`

这是 v2、有 F0 的主模型类。

它和上面几乎完全一样，最大的区别是：

- 把 `enc_p` 替换成 `TextEncoder(768, ...)`

所以 v1/v2 的根本差异主要是：

- 输入 HuBERT 特征维度不同

#### `SynthesizerTrnMs256NSFsid_nono`

这是 v1、无 F0 的主模型类。

核心差异：

- `enc_p` 的 `f0=False`
- `dec` 换成普通 `Generator`
- `forward/infer` 不再处理 `pitch` / `pitchf`

#### `SynthesizerTrnMs768NSFsid_nono`

这是 v2、无 F0 的版本。

和 `SynthesizerTrnMs256NSFsid_nono` 的区别同样只是输入维度变 `768`。

#### `MultiPeriodDiscriminator`

训练用判别器集合。

组成：

- 一个 `DiscriminatorS`
- 多个 `DiscriminatorP(period in [2,3,5,7,11,17])`

输出：

- 真实/生成音频的判别结果
- 各层 feature map

#### `MultiPeriodDiscriminatorV2`

v2 训练对应的判别器增强版。

区别是 period 更多：

- `[2,3,5,7,11,17,23,37]`

#### `DiscriminatorS`

- 直接在 1D 波形上做卷积判别

#### `DiscriminatorP`

- 先按 period 把 1D 波形重排成 2D
- 再做 2D 卷积判别

这两个判别器一起组成 HiFi-GAN 风格的多尺度/多周期对抗器。

总结这个文件最重要的一句话：

- `models.py` 决定了“训练时到底学什么模型”和“推理时到底恢复什么模型”

### `infer/lib/infer_pack/modules/F0Predictor/__init__.py`

这是空的包标记文件。

作用只是让 `F0Predictor` 这一组实现能作为子包被 import。

### `infer/lib/infer_pack/modules/F0Predictor/F0Predictor.py`

这是 F0 predictor 的抽象接口定义。

它只声明了两个方法：

- `compute_f0(wav, p_len)`
- `compute_f0_uv(wav, p_len)`

本身没有实现，主要用于给 `PM` / `Harvest` / `Dio` 版本统一接口。

### `infer/lib/infer_pack/modules/F0Predictor/PMF0Predictor.py`

这是基于 `parselmouth` 的 pitch predictor 封装。

主要逻辑：

- `compute_f0()`
  用 `to_pitch_ac()` 提取 F0
- `interpolate_f0()`
  把无声段/缺失段插值补齐
  同时可导出 `uv`（voiced/unvoiced）向量
- `compute_f0_uv()`
  同时返回连续 F0 和 uv

这个文件的价值主要在“把 PM 算法包装成统一 predictor 接口”。

### `infer/lib/infer_pack/modules/F0Predictor/HarvestF0Predictor.py`

这是基于 `pyworld.harvest` 的 predictor 封装。

主要逻辑：

- `compute_f0()`
- `compute_f0_uv()`
- `resize_f0()`
  把 world 输出重采样到目标帧数
- `interpolate_f0()`
  填补无声/缺失段

它和 `PMF0Predictor` 的区别主要在底层算法不同，但接口保持一致。

### `infer/lib/infer_pack/modules/F0Predictor/DioF0Predictor.py`

这是基于 `pyworld.dio` 的 predictor 封装。

结构和 `HarvestF0Predictor` 非常接近：

- `dio + stonemask`
- `resize_f0()`
- `interpolate_f0()`
- `compute_f0_uv()`

和 Harvest 版相比，它还会把 pitch round 到 0.1 精度。

### 关于 `F0Predictor/` 这一组文件的定位

这一组文件需要特别说明一下：

- 它们提供了规范的 F0 predictor 类封装
- 但当前仓库主链路里，`pipeline.py` 和 `extract_f0_*.py` 并没有直接实例化这些类
- 主链路更像是把 PM / Harvest / DIO / RMVPE 逻辑直接内联到了脚本里

所以 `F0Predictor/` 更像：

- 底层可复用实现
- 历史兼容代码
- 或为其他调用场景保留的封装接口

---

## 最后给一个“看代码顺序”建议

如果你现在是要真正啃懂 `infer/`，建议按这个顺序读，理解成本最低：

1. `modules/train/preprocess.py`
2. `modules/train/extract_f0_print.py`
3. `modules/train/extract_feature_print.py`
4. `lib/train/data_utils.py`
5. `modules/train/train.py`
6. `lib/train/process_ckpt.py`
7. `modules/vc/utils.py`
8. `modules/vc/modules.py`
9. `modules/vc/pipeline.py`
10. `lib/infer_pack/models.py`
11. `lib/infer_pack/modules.py`
12. `lib/infer_pack/attentions.py`

如果只想抓主流程，不想一开始就陷进模型细节里，那最少先抓住这条线：

- `preprocess.py -> extract_f0_print.py / extract_feature_print.py -> train.py -> process_ckpt.py -> train-index*.py -> vc/modules.py -> vc/pipeline.py`

这条线看懂以后，再回头啃 `infer_pack/`，会顺很多。
