# 第 1 周 W0 版本锁定（VERSION_LOCK）

- 修订（2026-11-26，Asia/Shanghai）：V2.8 官方验收机改为**官方 GeForce RTX 4090 24GB**（nvidia-smi 显存接受范围 24000–25000 MiB，典型约 24564 MiB，以新主机实测为准）。NVIDIA 官方无 4090 48GB。此前晨涧云 `ubuntu-30041` 改装 4090 48G（49140 MiB）的完整记录已备份为 `VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126`，不得再 SSH 连接该主机，其结果不能作为 V2.8 官方 24GB 验收。冻结软件版本（Python 3.12.0、PyTorch 2.8.0+cu128、CUDA 12.8、TE 2.21.0、huggingface_hub 2.4.0、cuDNN 安装目标 9.28.1.4 等）不变。
- 文档用途：冻结第 1 周环境的软件版本，并选定可安装的 Transformer Engine、CPU Faiss 与 Gate 0 公开工具版本。当前验收目标是官方 GeForce RTX 4090 24GB 新租主机，不是昨天的改装 48G 机。
- 撰写日期 / 查阅日期：2026-11-25（Asia/Shanghai）。下文凡未另标时间的「查阅日期」均为此日。
- 范围：只锁定与书写可重复安装脚本。本次**不登录云主机**、不索要密码或令牌、不下载模型权重、不在用户本机操作，也不开始完整 CUDA 解码内核或 KV 卸载。
- 原则：数字与版本号均来自官方页面、PyPI 元数据、任务给定的目标主机信息，或开发总管转述的主机实测；未在目标主机上测到的项一律标为「待实测」，不编造兼容性或 4090 基线数字。
- 开发总管已于 2026-11-25 审过 W0，三项决策见第 9 节；本文件已按决策修订，不再把这些事项列为待决策。

## 1. 目标主机与硬件

| 项目 | 锁定值 | 来源 | 备注 |
| --- | --- | --- | --- |
| 主机 | 待用户提供的新租官方 GeForce RTX 4090 24GB 主机 | 开发总管 2026-11-26：已确认换机，不是昨天测试的改装 48G | 不得使用历史地址（含 175.155.64.171）尝试登录。旧主机 `ubuntu-30041` 记录见 `VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126` |
| 操作系统 | 待实测（连接后写入快照） | 新主机尚未连接 | 旧 48G 主机曾为 Ubuntu 24.04.4 LTS，不假定新机相同 |
| 内核 | 待实测（连接后写入快照） | 新主机尚未连接 | 旧 48G 主机曾为 6.17.0-23-generic |
| GPU | 官方 GeForce RTX 4090 24GB GDDR6X，计算能力 sm_89（Ada Lovelace）；nvidia-smi 显存须在 24000–25000 MiB（典型约 24564 MiB，以新主机实测为准） | V2.8 计划与开发总管 2026-11-26 决定；计算能力对照见 Transformer Engine 2.21.0 支持矩阵 | 名称须含 RTX 4090。若显存约 49140 MiB 或属于 ≥40000 MiB 档，立即失败。软件显存上限不能作为官方 24GB 验收。旧记录「RTX 4090 48G，49140 MiB」见备份 |
| 驱动 | 待实测（连接后写入快照，随后禁止升级） | 新主机尚未连接 | 旧 48G 主机曾锁定 580.159.03，不得作为新机硬门槛 |
| CUDA 工具包 | 12.8，路径 `/usr/local/cuda-12.8` | 任务给定 | 禁止升级 |
| 数据盘 | `/data` | 任务给定 | — |
| 模型路径 | `/data/models/Meta-Llama-3-8B` | 任务给定 | 仅使用 Hugging Face safetensors；排除 `original/*`。权重由开发总管另行下载，本脚本不得执行 `huggingface-cli download` |
| Conda 环境 | `torch2.8_cuda12.8` | 任务给定 | **禁用** `torch2.14_cuda13.0`。非交互 shell 须先 `source ~/anaconda3/etc/profile.d/conda.sh` |
| Hugging Face 访问 | `https://hf-mirror.com` | 任务给定 | 目标主机访问不了 huggingface.co |

出处（硬件支持矩阵）：<https://docs.nvidia.com/deeplearning/transformer-engine/support_matrix.html>（Transformer Engine 2.21.0 文档，查阅于 2026-11-25）。该页写明：计算能力 8.9 对应 Ada Lovelace，代表 GPU 含 RTX 40 系列；该架构支持 BF16 与逐张量 FP8（`DelayedScaling` / `Float8CurrentScaling`），默认 recipe 为 `DelayedScaling`；Ada 上使用 FP8 还需要 cuBLASLt 12.1.3.x 或更高。FP8 block scaling 额外要求 CUDA 12.9 或更高，且支持矩阵未将 block scaling 列为 8.9 的能力。本方案 Gate 0 的 TE FP8 基线应走逐张量 FP8，不依赖 block scaling。

## 2. 已冻结、禁止升级的软件

下列版本已在目标主机给出，脚本必须检查并拒绝偏离，安装过程不得升级它们。`nvidia-cudnn-cu12` **不在**本表中：它已按决策 1 从主机实测的 9.10.2.21 单独升级并钉死为第 3 节的 9.28.1.4，同时仍禁止升级 PyTorch / CUDA 工具包 / 驱动。

| 名称 | 版本 | 来源 | 出处 URL | 查阅日期 | 备注 |
| --- | --- | --- | --- | --- | --- |
| Python | 3.12.0 | 任务给定（环境 `torch2.8_cuda12.8`） | — | 2026-11-25 | 禁止升级 |
| PyTorch | 2.8.0+cu128 | 任务给定 | PyPI 元数据 <https://pypi.org/project/torch/2.8.0/> | 2026-11-25 | 禁止升级。官方 `requires_dist` 声明 Linux x86_64 依赖于 `nvidia-cudnn-cu12==9.10.2.21`、`nvidia-cublas-cu12==12.8.4.1`、`nvidia-cuda-runtime-cu12==12.8.90` 等。其中 cuDNN 按第 4.1 节单独替换；cuBLAS 等其余 NVIDIA pip 包须用约束钉在当前已装版本，不得随 cuDNN 一起升级 |
| CUDA 工具包 | 12.8（`/usr/local/cuda-12.8`） | 任务给定 | — | 2026-11-25 | 禁止升级 |
| NVIDIA 驱动 | 待实测（连接后记录并禁止升级） | 新主机尚未连接；旧 48G 主机记录为 580.159.03 | — | 2026-11-26 修订 | 不得把已停用 48G 主机的 580.159.03 当作新机硬门槛。软件栈其余冻结项不变 |
| huggingface_hub | 2.4.0 | 任务给定「已装、不要升级」；PyPI 该版本上传时间为 2026-10-29T15:30:10Z | <https://pypi.org/project/huggingface-hub/2.4.0/> | 2026-11-25 | 禁止升级。该版本声明 `Python >=3.10.0`，未被 yank |

## 3. 本次选定并锁定的安装版本

| 名称 | 版本 | 来源 | 出处 URL | 查阅日期 | 备注 |
| --- | --- | --- | --- | --- | --- |
| nvidia-cudnn-cu12 | **9.28.1.4** | 2026-11-25 当日 PyPI 当前稳定版（面向 CUDA 12 的 `cu12` 包） | 项目页 <https://pypi.org/project/nvidia-cudnn-cu12/9.28.1.4/>；JSON <https://pypi.org/pypi/nvidia-cudnn-cu12/9.28.1.4/json>；索引 <https://pypi.org/pypi/nvidia-cudnn-cu12/json> | 2026-11-25 | 选版理由见第 4.1 节。提供 `py3-none-manylinux_2_27_x86_64` 轮子（x86_64 轮子上传时间为 2026-11-06T21:51:22.240285Z），未被 yank。声明依赖 `nvidia-cublas-cu12`（无版本下限），安装时必须用约束钉死已有 cublas，禁止把它升级掉 |
| Transformer Engine（元包 `transformer-engine`，核心 `transformer-engine-cu12`，PyTorch 扩展 `transformer-engine-torch`） | **2.21.0** | NVIDIA 当前稳定版文档、PyPI、GitHub Release | 安装页 <https://docs.nvidia.com/deeplearning/transformer-engine/installation.html>；发行说明 <https://docs.nvidia.com/deeplearning/transformer-engine/release_notes.html>；支持矩阵 <https://docs.nvidia.com/deeplearning/transformer-engine/support_matrix.html>；PyPI <https://pypi.org/project/transformer-engine/2.21.0/>、<https://pypi.org/project/transformer-engine-cu12/2.21.0/>、<https://pypi.org/project/transformer-engine-torch/2.21.0/>；GitHub Release <https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.21.0> | 2026-11-25 | 选版依据见第 4 节。PyPI 上 `transformer-engine-torch==2.21.0` 仅提供 sdist，声明 `torch>=2.1`；须用 `--no-build-isolation` 针对已冻结的 PyTorch 2.8.0+cu128 源码编译。核心库提供 `manylinux_2_28_x86_64` 轮子。默认安装 CUDA 12 核心（`core` / `core_cu12`），不要装 `core_cu13`。失败时不得自动降到 2.19.0 |
| faiss-cpu | **1.16.0** | PyPI 当前稳定版；v1.16.0 源码头文件 | <https://pypi.org/project/faiss-cpu/1.16.0/>；ProductQuantizer API <https://raw.githubusercontent.com/facebookresearch/faiss/v1.16.0/faiss/impl/ProductQuantizer.h> | 2026-11-25 | **只装 CPU 包**。提供 `cp310-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64` 轮子，分类器含 Python 3.12；`Requires-Python: >=3.10`，依赖 `numpy>=1.25`。v1.16.0 头文件含 `train(...)` 与 `compute_codes(...)`。禁止安装 `faiss-gpu` / `faiss-gpu-cuvs`，本周不使用 `search` / `IndexIVFPQ` / `GpuIndex` |
| transformers | **5.24.1** | PyPI 当前稳定版 | <https://pypi.org/project/transformers/5.24.1/> | 2026-11-25 | 声明 `Python >=3.11.0`；主依赖含 `huggingface-hub>=1.31.0,<3.0`（与已冻结的 2.4.0 相容）、`tokenizers>=0.23.1,<0.24.0`、`safetensors>=0.8.0`；可选 `torch` extra 声明 `torch>=2.6`。不选用 4.57.6：该线声明 `huggingface-hub>=0.34.0,<1.0`，与已冻结的 huggingface_hub 2.4.0 冲突 |
| accelerate | **1.16.0** | PyPI 当前稳定版 | <https://pypi.org/project/accelerate/1.16.0/> | 2026-11-25 | 声明 `Python >=3.11.0`、`torch>=2.6.0`、`huggingface_hub>=0.21.0`、`safetensors>=0.4.3`，与已冻结的 PyTorch / huggingface_hub 相容 |
| datasets | **5.2.0** | PyPI 当前稳定版（Gate 0 评测数据加载） | <https://pypi.org/project/datasets/5.2.0/> | 2026-11-25 | 主依赖含 `huggingface-hub>=1.31.0,<3.0`，与已冻结的 2.4.0 相容。不要安装其 `audio` / `dev` 等会拉高或钉死 torch 的 extras |
| lm-eval（`lm-evaluation-harness`） | **0.4.13** | PyPI 当前稳定版（Gate 0 的 MMLU 5-shot、WikiText-2 / C4 等评测入口） | <https://pypi.org/project/lm-eval/0.4.13/> | 2026-11-25 | 声明 `Python >=3.10`。使用 extras `hf`，其声明为 `transformers>=4.1`、`torch>=1.8`、`accelerate>=0.26.0`、`datasets>=2.16.0`。与 transformers 5.24.1 的运行时相容性为待实测。安装时必须约束 torch / huggingface_hub / transformers，避免被传递依赖升级 |
| tokenizers | **0.23.3** | 满足 transformers 5.24.1 的 `>=0.23.1,<0.24.0` 范围内的 PyPI 最新版 | <https://pypi.org/project/tokenizers/0.23.3/> | 2026-11-25 | 提供 `cp310-abi3` manylinux x86_64 轮子，可用于 Python 3.12 |
| safetensors | **0.8.0** | PyPI 当前稳定版；满足 transformers `>=0.8.0` | <https://pypi.org/project/safetensors/0.8.0/> | 2026-11-25 | 提供 `cp310-abi3` manylinux x86_64 轮子 |
| sentencepiece | **0.2.3** | PyPI 当前稳定版（tokenizer 备用依赖） | <https://pypi.org/project/sentencepiece/0.2.3/> | 2026-11-25 | 提供 `cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64` 轮子 |
| einops | **0.8.2** | PyPI 当前稳定版（部分量化 / 模型代码常见依赖） | <https://pypi.org/project/einops/0.8.2/> | 2026-11-25 | 纯 Python 轮子 |

### 3.1 Gate 0 量化对照库（已决策：本轮默认不装）

开发总管已于 2026-11-25 决定：**维持默认不装** AWQ / GPTQ / AQLM。这些包放到第二轮，待确认与 transformers 5.24.1 及已冻结栈相容后再写**单独脚本**。`setup_env.sh` 不得安装下表任何包。禁止安装 **gptqmodel 7.12.2** 以及任何要求 `torch>=2.14` 的版本。此项不再待决策。

| 名称 | 记录用候选版本 | 来源 | 出处 URL | 查阅日期 | 备注 |
| --- | --- | --- | --- | --- | --- |
| aqlm | 1.1.7（PyPI 当前稳定版，仅作第二轮候选记录） | PyPI | <https://pypi.org/project/aqlm/1.1.7/> | 2026-11-25 | 声明 `torch>=2.2.0`、`transformers>=4.38.0`、`accelerate>=0.27.0`。上传时间为 2025-04-16T12:40:03Z。与 transformers 5.24.1 的运行时相容性仍为待实测，本轮不装 |
| autoawq | 0.2.9；kernels 0.0.9（仅作第二轮候选记录） | PyPI | <https://pypi.org/project/autoawq/0.2.9/>、<https://pypi.org/project/autoawq-kernels/0.0.9/> | 2026-11-25 | `autoawq` 声明 `transformers>=4.45.0`、`huggingface_hub>=0.26.5`，仅有 sdist；`autoawq-kernels 0.0.9` 有 `cp312` manylinux 轮子且声明 `torch>=2.5.1`。与 transformers 5.x 的相容性未在官方依赖中明确，本轮不装 |
| gptqmodel | 本轮不装；**禁止** 7.12.2 及要求 `torch>=2.14` 的版本 | PyPI | <https://pypi.org/project/gptqmodel/7.6.0/>、<https://pypi.org/project/gptqmodel/7.12.2/> | 2026-11-25 | **7.12.2** 在 `python_version >= "3.11"` 时声明 `torch>=2.15.0`，会破坏已冻结的 PyTorch 2.8.0，禁止安装。7.7.0–7.10.0 声明 `torch>=2.14.1`，同样禁止。**7.6.0** 声明 `torch>=2.8.0`、`transformers>=5.14.0`，不强制升级 PyTorch，但钉死 `numpy==2.2.6`（Python < 3.14）并依赖 `torchao>=0.18.0`，是否列入第二轮单独脚本需另验。`auto-gptq 0.7.1` 无 cp312 轮子，不选用 |
| bitsandbytes / llama-cpp（GGUF） | 本轮不装 | — | — | 2026-11-25 | 若第二轮要做 GGUF 对照，需另选包并验证其不会升级 PyTorch |

## 4. Transformer Engine 选版依据

目录检查（只查不装）：2026-11-25 在本箱子上执行 `npx skills add nvidia/skills --list`（源：<https://github.com/nvidia/skills.git>），目录显示 488 个技能。检索结果中**没有**独立的 Transformer Engine 安装 / 版本锁定技能，因此未安装任何 NVIDIA skill，改以 docs.nvidia.com、GitHub 发行说明与 PyPI / NGC 安装信息为准。

选定 **Transformer Engine 2.21.0** 的依据如下。

1. **它是查阅当日的官方当前稳定版。** `docs.nvidia.com` 上 Transformer Engine 文档标题为 2.21.0，文档库也将 2.21 列为当前发行：<https://docs.nvidia.com/deeplearning/transformer-engine/documentation-archive.html>（查阅于 2026-11-25）。PyPI `transformer-engine` 最新稳定版同样是 2.21.0，轮子上传时间为 2026-11-10T21:19:18Z。GitHub Release `v2.21.0` 发布于 2026-11-10T20:56:58Z。
2. **官方安装页与本机 CUDA 12.8、Linux x86_64 对齐。** 2.21.0 安装页写明先决条件为：Linux x86_64；CUDA 12.1+（Blackwell 需要 12.8+）；支持 CUDA 12.1 或更新的驱动；**cuDNN 9.12 或更新**。官方安装要求为 Linux x86_64、CUDA 12.1+、支持该 CUDA 的驱动，以及 cuDNN 9.12 或更新。本锁定仍要求 CUDA 工具包 12.8；新官方 24GB 主机的发行版与精确驱动版本待连接后写入快照，不得沿用已停用 48G 主机的 580.159.03 作为硬门槛。RTX 4090 为 Ada（非 Blackwell），官方对 Ada 的 CUDA 下限是 12.1+，12.8 覆盖该要求。安装方式采用官方推荐的 PyPI 路径：`pip3 install --no-build-isolation transformer_engine[pytorch]`，并钉死 2.21.0；核心库默认走 CUDA 12（`transformer_engine_cu12`），与本机 CUDA 12.8 一致，禁止安装 `core_cu13`。
3. **有面向 PyTorch 2.8 的官方构建证据，且 PyTorch 扩展可对已装 2.8.0 源码编译。** GitHub `v2.21.0` 资产包含 `transformer_engine_torch-2.21.0+cu12torch2.8.0+cu129cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`，说明该发行正式针对 PyTorch 2.8.0、Python 3.12、Linux x86_64 出过 cu12 构建。PyPI 上的 `transformer-engine-torch 2.21.0` 则只提供 sdist，并声明 `torch>=2.1`；官方安装页说明「Source distributions are shipped for the JAX and PyTorch extensions」，要求 `--no-build-isolation`，以便对着**当前环境里已有的** PyTorch 编译。本锁定采用后一种官方安装路径，直接编译到已冻结的 `2.8.0+cu128`，而不去使用文件名带 `cu129` 的预编译轮子（本机是 cu128，不能假定该轮子可直接替换）。v2.21 发行说明还写明「Fixed source builds for Blackwell targets with CUDA 12.8.（#3602）」，说明 2.21 源码构建明确覆盖 CUDA 12.8。
4. **官方硬件支持矩阵覆盖本机 sm_89 的逐张量 FP8。** 见第 1 节。这是 Gate 0 TE FP8 基线的硬件前提。
5. **不采用 NGC 容器里的 TE 版本作为本环境的安装版本。** NGC 在查阅日没有「官方稳定版 PyTorch 2.8.0 + CUDA 工具包 12.8 + 某一 TE」的现成配对：PyTorch 容器 25.03 是 CUDA 12.8.1.012 + PyTorch `2.7.0a0+...`（<https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-03.html>）；25.06 是 CUDA 12.9.1 + PyTorch `2.8.0a0+...` + TransformerEngine 2.4（<https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-06.html>）；25.08 起已是 CUDA 13.0 + PyTorch `2.8.0a0+...`。这些容器版本或 CUDA、或 PyTorch 小版本与本机已冻结栈不一致，不能作为本机升级目标，只作对照说明。

### 4.1 cuDNN：主机实测与已选定的方案 (a)

Transformer Engine **2.21.0 安装页要求 cuDNN 9.12 或更新**。GitHub Release `v2.20.2`（2026-10-05，<https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.20.2>）的 Breaking Changes 写明：自 2.20.2 起将最低支持的 cuDNN 从 9.3 提高到 9.12.0，并在 configure / build / import / runtime 各阶段强制检查。

**主机已实测（由开发总管于 2026-11-25 转述，本次仍不登录该主机）：** 已装 `nvidia-cudnn-cu12 9.10.2.21`，`torch.backends.cudnn.version() = 91002`，低于 TE 2.21.0 的官方门槛。该 9.10.2.21 也与 PyTorch 2.8.0 在 PyPI 上官方钉死的 Linux x86_64 依赖一致（<https://pypi.org/project/torch/2.8.0/>，查阅于 2026-11-25）。

**已选方案 (a)：** 在不升级 PyTorch、CUDA 工具包、驱动的前提下，单独安装并钉死一个面向 CUDA 12、且 ≥ 9.12 的 `nvidia-cudnn-cu12`。选定确切版本为 **9.28.1.4**。脚本不得因此把 Transformer Engine 自动降到 2.19.0；锁定主版本仍是 TE 2.21.0。

**为何选 9.28.1.4，而不是更高或更低的符合版本（查阅于 2026-11-25）：**

- PyPI 项目页与 `https://pypi.org/pypi/nvidia-cudnn-cu12/json` 当日给出的当前发行是 **9.28.1.4**，包名 `nvidia-cudnn-cu12` 即面向 CUDA 12 的运行时库；未被 yank；有 Linux x86_64 的 `manylinux_2_27` 轮子，适合 Ubuntu 24.04。
- **不选更高的 9.29.0.35.dev84893002**：同日索引里唯一高于 9.28.1.4 的版本带 `.dev` 前发行后缀，且只有 sdist（上传时间为 2026-11-24T21:54:07.776284Z），不是稳定轮子发行。
- **不选更低的 9.12.0.46：** 它刚好达到 TE 2.21.0 的 9.12.0 下限（上传于 2025-08-07），但不是查阅当日的当前稳定发行。
- **不选 9.28.0.42：** 同属 9.28 线但不是该线最新补丁。9.25.0.15 已被 yank，也不选用该旧线。
- 9.28.1.4 对 `nvidia-cublas-cu12` 没有版本下限，便于用约束把已有 cublas（PyTorch 2.8.0 官方钉死值为 12.8.4.1）留在原地。

`setup_env.sh` 将：先按约束安装 `nvidia-cudnn-cu12==9.28.1.4`；装完后重检 `torch==2.8.0+cu128` 与 `huggingface_hub==2.4.0`，并通过后再编译 TE 2.21.0；若 cuDNN 安装、TE 编译或 FP8 冒烟失败，立即停止，把日志写到脚本同目录 `logs/`，**不改锁定主版本**。

**主机实测（2026-11-26，Asia/Shanghai）：** 第一次执行已把 `nvidia-cudnn-cu12` 从 9.10.2.21 换成 **9.28.1.4**，新进程中 `torch.backends.cudnn.version()` 为 **92801**（≥ 91200），`torch` 仍为 `2.8.0+cu128`，`huggingface_hub` 仍为 2.4.0，`nvidia-cublas-cu12` 仍为 12.8.4.1。随后用同一份同时钉死 `torch==2.8.0` 与 `nvidia-cudnn-cu12==9.28.1.4` 的约束安装 `transformer-engine[pytorch]==2.21.0` 时，pip 报 `ResolutionImpossible`：`torch 2.8.0+cu128` / `torch 2.8.0` 的元数据要求 `nvidia-cudnn-cu12==9.10.2.21`。脚本按策略于 01:31 停止，没有回退到 TE 2.19.0，也没有改锁定主版本。因此安装方式改为：先用**不含** `nvidia-cudnn-cu12` 引脚的求解约束安装 `transformer-engine-cu12==2.21.0` 与运行时依赖，再用 `--no-build-isolation --no-deps` 编译 `transformer-engine-torch==2.21.0` 并安装 `transformer-engine==2.21.0`；硬依赖 torch 的 `accelerate`、`peft` 同样 `--no-deps`。禁止安装 `transformer-engine-cu13`。说明：PyPI 上 `transformer-engine-torch 2.21.0` 的静态 Requires-Dist 写的是 `transformer_engine_cu13==2.21.0`（sdist 在 CUDA 13 环境打包，该字段为 Dynamic），但源码 `setup.py` 会按 `torch.version.cuda` 主版本选择 cu12 或 cu13；本环境是 `2.8.0+cu128`，应装 cu12。TE 编译与 DelayedScaling FP8 冒烟仍待这次重跑验证。

## 5. Faiss 选版依据

- 本周只使用 `faiss.ProductQuantizer` 的 `train` / `compute_codes` 做离线码本训练与编码，不调用 `search()`，不建 `IndexIVFPQ`，不安装也不使用 GPU Faiss / `GpuIndex`。
- 锁定 PyPI 当前稳定版 **faiss-cpu 1.16.0**（上传 2026-11-10T16:40:58Z）。该版本分类器明确包含 Python 3.12，Linux 轮子为 manylinux_2_27 / manylinux_2_28 x86_64 的 abi3 构建，可在 Ubuntu 24.04（glibc 高于 2.28 的常规发行，目标主机上的准确 glibc 仍为待实测）与 Python 3.12.0 上安装。
- v1.16.0 的 `ProductQuantizer.h` 含 `train` 与 `compute_codes` 接口，满足本周用法。
- 安装来源只用 PyPI `faiss-cpu==1.16.0`，脚本在安装后拒绝出现 `faiss-gpu` 或 `faiss-gpu-cuvs`。

## 6. 安装约束

1. 只在 conda 环境 `torch2.8_cuda12.8` 中安装；非交互 shell 先 `source ~/anaconda3/etc/profile.d/conda.sh`。检测到环境名 `torch2.14_cuda13.0` 或 PyTorch / CUDA 工具包 / Python / huggingface_hub 与锁定不符时，立即退出且不做任何安装。新主机驱动版本待实测记录，不把旧 48G 主机的 580.159.03 作为硬门槛。GPU 必须通过官方 24GB 硬件断言（24000–25000 MiB，名称含 RTX 4090，sm_89）；改装 48G 立即失败。
2. 不升级已冻结的 PyTorch、CUDA 工具包、驱动、Python、huggingface_hub。完整约束文件钉死 `torch==2.8.0`、`huggingface_hub==2.4.0`（及 `huggingface-hub==2.4.0`）和 `nvidia-cudnn-cu12==9.28.1.4`，并把环境中除 `nvidia-cudnn-cu12` 以外的已装 `nvidia-*` 包（尤其是 `nvidia-cublas-cu12`）钉在当前版本。只允许把 `nvidia-cudnn-cu12` 安装为 **9.28.1.4**。该完整约束只用于单独安装 cuDNN。安装 TE 或硬依赖 torch 的包时，改用**去掉** `nvidia-cudnn-cu12` 行的求解约束，并对 TE torch 插件 / `nvdlfw-inspect` / `accelerate` / `peft` 使用 `--no-deps`，装完后必须复检 cuDNN 仍为 9.28.1.4、`version()` ≥ 91200，且未安装 `transformer-engine-cu13`。不把官方 GitHub 上文件名含 `cu129` 的 TE 预编译轮子当作本机安装物。
3. 安装顺序：先按完整约束安装 cuDNN 9.28.1.4 → 重检 torch 仍为 `2.8.0+cu128`、huggingface_hub 仍为 `2.4.0`，且 `torch.backends.cudnn.version()` ≥ 91200、包版本为 9.28.1.4 → 再安装 `transformer-engine-cu12` 与运行时依赖，并以 `--no-build-isolation --no-deps` 编译 TE 2.21.0 → 做逐张量 DelayedScaling 的 FP8 冒烟 → 再装 Faiss 与 Gate 0 基础工具（`accelerate` / `peft` 用 `--no-deps`）。任一步失败即停止，不回退、不改 TE / PyTorch / 驱动 / CUDA 工具包的锁定主版本。`peft==0.22.0` 是 `lm-eval[hf]` 的辅助依赖（查阅于 2026-11-26，要求 `torch>=1.13.0`，未被 yank），不是新的主版本决策。
4. 失败日志写在**脚本同目录**的 `logs/` 下：完整记录 `setup_env-YYYYMMDD-HHMMSS.log`（时间戳为 Asia/Shanghai），另有分步文件 `cudnn-install-*.log`、`te-install-*.log`、`fp8-smoke-*.log`。脚本失败时会在终端打印这些路径。
5. 安装与后续可能的 Hub 访问走 `HF_ENDPOINT=https://hf-mirror.com`。本脚本不下载权重；可检查 `/data/models/Meta-Llama-3-8B` 是否存在以及是否有 safetensors，若路径缺失只告警、不失败退出（权重由开发总管另行下载）。
6. Transformer Engine 必须 `--no-build-isolation`，设置 `CUDA_HOME=/usr/local/cuda-12.8`、`NVTE_FRAMEWORK=pytorch`。按官方环境变量说明，可将 `NVTE_CUDA_ARCHS` 设为 `89`，只为这张 4090 编译（官方默认在 CUDA 12.8+ 下是 `70;80;89;90;100;120`，见 <https://docs.nvidia.com/deeplearning/transformer-engine/envvars.html>，查阅于 2026-11-25）。若构建资源不足，可按官方安装页使用 `MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1`。
7. 源码编译 TE 需要官方 README / 安装页提到的构建依赖：CMake 3.18+、Ninja、可编译 C++17 的 GCC 9+ 或 Clang 10+、pybind11。这些是否已在目标主机存在为待实测；脚本仅在缺失时安装构建工具，不把某一新的 cmake 主版本写成强制锁定。
8. 已决策：不安装 GPU Faiss；不安装 gptqmodel / autoawq / aqlm / auto-gptq（见第 3.1 节）；禁止安装要求 `torch>=2.14` 的 gptqmodel（含 7.12.2）。不安装会把 PyTorch 拉到 2.14 / 2.15 或 CUDA 13 的包。
9. 可重复执行：已是锁定版本则保持不变；对 `VERSION_LOCK.md` 的主机快照使用标记块整体替换，禁止无限追加。

## 7. 停线说明（口径已定，阈值待 4090 基线写入）

开发计划中的 S2 停线曾按 **H100 口径** 写成 1.2 ms/token。开发总管已于 2026-11-25 决定：**口径先定、阈值后写**。不得把该 1.2 ms 填进本机阈值，也不得编造任何 4090 基线或停线数字。

**已定测量口径：**

- 机器：官方 GeForce RTX 4090 24GB（sm_89；显存 24000–25000 MiB），环境 `torch2.8_cuda12.8`。改装 48G 不能作为官方验收
- 模型形状：Llama 3 8B 官方形状
- TE FP8：逐张量 `DelayedScaling`（与官方支持矩阵在 Ada 上的默认 recipe 一致，见第 1 节；官方入门示例用法见 <https://docs.nvidia.com/deeplearning/transformer-engine/getting_started/index.html>，查阅于 2026-11-25）
- decode 先 `batch=1`
- prefill 与 decode 分列报告
- 先 warmup，再取重复运行的中位数
- FFN gather 的 ms/token 对照**同卡** TE FP8 基线（不是 H100 文档数字）

**阈值位（仍留空，待上述口径在 4090 上测完基线后再写入）：**

| 校准项 | 状态 | 数值 |
| --- | --- | --- |
| 4090 上 TE 逐张量 DelayedScaling 的 decode / FFN 基线延迟（中位数；prefill 与 decode 分列） | 待实测 | （留空） |
| 由该同卡基线校准后的 S2 停线（FFN gather ms/token） | 待实测 | （留空，禁止填 1.2 或任何虚构值） |
| 是否真正走到 TE FP8 kernel（而非静默回退） | 待实测 | （留空） |
| 峰值显存、prefill / decode 吞吐 | 待实测 | （留空；按已定口径：warmup 后取重复运行中位数） |

## 8. 不确定项 / 待实测

- **已测（2026-11-26）：** 换成 9.28.1.4 后，新进程中 `torch.backends.cudnn.version()` 为 92801（不再是 91002）；`torch` 仍为 `2.8.0+cu128`，`nvidia-cublas-cu12` 仍为 12.8.4.1。cuDNN 安装步骤本身可以通过。同一次安装里，用完整约束编译 TE 会在 pip 求解阶段失败，见第 4.1 节。
- 目标主机实际加载的 cuBLAS / cuBLASLt 是否 ≥ 12.1.3.x（Ada FP8 的官方附加要求）。目前包版本仍是 PyTorch 2.8.0 官方钉死的 `nvidia-cublas-cu12==12.8.4.1`，数字上高于 12.1.3，但仍须在 TE 编译与冒烟后再确认没有被改掉。
- GCC / Clang、CMake、Ninja、cuDNN 头文件路径是否满足 TE 2.21.0 源码编译；ABI 是否与已装 PyTorch 一致（官方安装页将 ABI 不匹配列为常见 `ImportError` 原因）。主机已有 g++ 13.3.0；TE 源码编译仍待这次 `--no-deps` 重跑验证。
- `transformer_engine.pytorch` 在本机 2.8.0+cu128 + CUDA 12.8 + cuDNN 9.28.1.4 + sm_89 上导入，以及逐张量 DelayedScaling 的 FP8 冒烟与是否真正走 FP8 kernel，仍待重跑验证。
- glibc 准确版本；faiss-cpu 1.16.0 与 transformers / lm-eval / datasets 在该环境的导入与基本功能。
- `lm-eval 0.4.13` 与 `transformers 5.24.1` 的运行时相容性。aqlm / autoawq / gptqmodel 与冻结栈的相容性留待第二轮单独验证，不在本轮安装中解决。
- 模型目录 `/data/models/Meta-Llama-3-8B` 是否已就绪、是否仅含 HF safetensors（排除 `original/*` 后的可用检查点）。
- 第 7 节阈值位：4090 TE FP8 基线与由此校准的停线数字。

## 9. 已决策事项（2026-11-25，不再待决策）

1. **cuDNN：** 安装前主机实测为 `nvidia-cudnn-cu12 9.10.2.21`、`torch.backends.cudnn.version()=91002`。已选方案 (a)：不升级 PyTorch / CUDA 工具包 / 驱动，单独安装并钉死 `nvidia-cudnn-cu12==9.28.1.4`，然后继续编译 TE **2.21.0**。不得自动降到 TE 2.19.0。失败则停止并保留 `logs/` 中的日志，不改锁定主版本。2026-11-26 第一次执行已确认 9.28.1.4 可装上且 `version()`=92801；随后的 TE pip 求解冲突及 `--no-deps` 安装方式见第 4.1 节，不改变本决策。
2. **AWQ / GPTQ / AQLM：** 维持默认不装。量化包放到第二轮，确认与 transformers 5.24.1 及冻结栈相容后再写单独脚本。禁止安装 gptqmodel 7.12.2 及任何要求 `torch>=2.14` 的版本。
3. **停线：** 口径已按第 7 节写明；阈值位留空，待 4090 基线测完再写。禁止填入 H100 的 1.2 ms。

本轮 W0 锁定不再等待新的总管决策。余下工作是在目标主机执行脚本、查看 `logs/`，以及在 W1 按第 7 节口径实测后再回填阈值。

## 10. 脚本跑完后的主机快照占位

`setup_env.sh` 在安装与校验结束后，会把当时的 `nvidia-smi` 与 `conda list` 完整输出写入本节下方的标记块。

- 标记：`<!-- BEGIN HOST_SNAPSHOT -->` … `<!-- END HOST_SNAPSHOT -->`
- 重复运行时**整块替换**，不得在文件末尾无限追加。
- 在目标主机成功跑完脚本之前，下面只有占位，没有实测内容。失败日志不写进本标记块，而写在脚本同目录 `logs/`。

<!-- BEGIN HOST_SNAPSHOT -->
## 主机实测快照（由 setup_env.sh 写入，可重入替换）

以下历史段落记录的是已停用改装 48G 主机 `ubuntu-30041` 上的第一次安装尝试，**不能**作为 V2.8 官方 24GB 验收快照。新官方 24GB 主机成功跑完 `setup_env.sh` 后，由脚本整块替换本标记。第一次执行（2026-11-25 20:21 启动，2026-11-26 01:31 停止，Asia/Shanghai）曾在 `ubuntu-30041` 上跑过：cuDNN 9.28.1.4 安装成功，`version()`=92801，冻结栈未被改动；TE 2.21.0 因 pip 约束冲突未装上。完整安装快照仍等脚本成功结束后写入下面的标记之间。失败日志在 `/data/te-faiss-week1/logs/setup_env-20261125-202100.log` 与 `te-install-20261125-202100.log`。

<!-- END HOST_SNAPSHOT -->
