# 官方 GeForce RTX 4090 24GB 第一天操作清单（V2.8）

- 撰写：2026-11-26（Asia/Shanghai）
- 规划基线：`/workspace/te-faiss-v2.8-plan.md`（2026-11-26，评审与原型验证计划，不是工程定稿）
- 开发总管当日范围：在**一天**租期内，于**官方** RTX 4090 24GB 新主机上完成硬件断言、W0 冻结安装与快照、Llama 3 8B BF16 + TE FP8、带宽与可用显存实测、Gate 1 gather（64 KB 与 32 KB）、以及本地语料已就绪时的 1–2 层离线 PQ。
- 不得把第 2 / 7 / 10 节的体积、Roofline tok/s、商业落点估算写成已测结果。目标任务仍为【待补】。
- 不得把改装 48G（历史约 49140 MiB）或软件显存上限当作 V2.8 官方 24GB 验收。不要再连接昨天的主机（含 175.155.64.171）。

本包复用已验收的 `/workspace/te-faiss-week1/` 路径（`setup_env.sh`、`VERSION_LOCK.md`、W1 / W2 / W3 脚本），不是另起一套非官方树。

## 0. 验收边界

- **通过条件（硬件）：** 整机只有一张 GPU；名称含 **RTX 4090**（拒绝 4090 D / Tesla P4 / 专业 48GB 卡）；计算能力 **sm_89**；`nvidia-smi` 显存 **24000–25000 MiB**（典型约 **24564 MiB**，以新主机实测为准）。
- **立即失败：** 显存约 49140 MiB，或属于 **≥40000 MiB** 档；非约 24GB；多卡；名称不符。GeForce 4090 无 MIG，不要建议 MIG。软件显存限制只是补充估计，**绝不是**官方 24GB 验收，也不是默认路径。
- **冻结软件（不得改引脚）：** Python 3.12.0，conda 环境 `torch2.8_cuda12.8`（禁用 `torch2.14_cuda13.0`），PyTorch **2.8.0+cu128**，CUDA 工具包 **12.8**（期望 `/usr/local/cuda-12.8`），Transformer Engine **2.21.0**（官方 PyPI，`--no-build-isolation`，对着冻结 torch 编译；禁止自动回退 2.19.0；禁止 `transformer-engine-cu13`），faiss-cpu **1.16.0**（只做离线 `ProductQuantizer.train` / `compute_codes`），transformers 5.24.1，accelerate 1.16.0，datasets 5.2.0，lm-eval 0.4.13，tokenizers 0.23.3，safetensors 0.8.0，sentencepiece 0.2.3，peft 0.22.0，huggingface_hub **2.4.0**。`nvidia-cudnn-cu12` 装完必须是 **9.28.1.4**，且 `torch.backends.cudnn.version() ≥ 91200`；求解约束里**不要**钉死 cuDNN（因为 torch 2.8.0+cu128 元数据要求 `==9.10.2.21`）。
- **默认不安装** AWQ / GPTQ / AQLM / GGUF。禁止 gptqmodel 7.12.2 及任何要求 `torch>=2.14` 的包。Qwen2.5-32B 对 AWQ 的显存对照是 V2.8 核心差异化场景，只准备脚本与说明，**不得**作为默认 W0 安装，也不得在这一天用 32B / 量化下载挡住 8B 工作。
- Ada 4090 支持逐张量 DelayedScaling FP8；block scaling 需要 CUDA 12.9+，不在范围。
- 国内主机常访问不了 huggingface.co：默认 `HF_ENDPOINT=https://hf-mirror.com`，允许覆盖。
- 非交互 SSH 须先 `source ~/anaconda3/etc/profile.d/conda.sh`（或设置 `CONDA_SH`）。不要假定新主机布局与旧机相同。
- 8B FFN 查表停线**措辞**为 **> 2 ms/token**（V2.8 第 8 节）。这不是已测的 4090 阈值，须等 Gate 1 实测后再写。禁止填入 H100 的 1.2 ms。

## 1. 把工具包拷到新主机

将整个 `te-faiss-week1/`（含 `official-4090-24g/wheelhouse/`）拷到新主机，例如 `/data/te-faiss-week1/`。不要只拷脚本而丢掉断言脚本与约束文件。

## 2. 当天推荐顺序

1. **连接**（由用户提供新主机 SSH / IP / 端口；本包不保存、不使用历史地址）。
2. **硬件断言（必须先做，失败则全天停止官方验收）：**
   ```bash
   bash official-4090-24g/assert_official_4090_24g.sh
   ```
3. **检查驱动与 CUDA 12.8。** 记录 `nvidia-smi` 驱动版本（待实测，不把旧 48G 机的 580.159.03 当硬门槛）。确认存在 `/usr/local/cuda-12.8` 且 `nvcc` 为 12.8。若路径不同但版本确为 12.8，再决定是否设置 `CUDA_HOME`，不要升级工具包。
4. **创建或激活冻结 conda 环境并装上 torch 2.8.0+cu128：**
   ```bash
   # 非交互 shell 必须先 source conda.sh
   source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
   export WHEELHOUSE=/data/te-faiss-week1/official-4090-24g/wheelhouse
   export PIP_FIND_LINKS="$WHEELHOUSE"
   bash official-4090-24g/create_conda_env.sh
   ```
5. **运行 W0 安装：**
   ```bash
   export WHEELHOUSE=...   # 同上
   export PIP_FIND_LINKS="$WHEELHOUSE"
   export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
   bash setup_env.sh
   ```
   脚本会再次调用官方 24GB 断言；非 24GB 立即失败，不再警告后继续。
6. **安装后断言：** torch 仍为 `2.8.0+cu128`，hub 仍为 2.4.0，`nvidia-cudnn-cu12==9.28.1.4`，`version()≥91200`，TE 2.21.0 可导入，DelayedScaling 冒烟通过，未安装 `transformer-engine-cu13` / GPU Faiss / 量化包。
7. **把 nvidia-smi 与 conda list 快照写入 `VERSION_LOCK.md` 标记块**（`setup_env.sh` 成功结束时自动替换）。旧 48G 主机上的历史段落不能当作官方 24GB 快照。
8. **可选：下载官方 Meta-Llama-3-8B**（约 16 GB；排除 `original/*`）。若目录已有完整 HF safetensors 则跳过。令牌只放环境变量，不要写入文件。
   ```bash
   # 若 gated：export HF_TOKEN=...   # 不要写入磁盘
   bash official-4090-24g/download_llama3_8b.sh
   ```
9. **实测带宽（替换规划值 0.9 TB/s）与可用显存（替换约 23.5 GB 假设）：**
   ```bash
   python official-4090-24g/measure_bandwidth.py
   python official-4090-24g/measure_usable_vram.py
   ```
   报告中位数、范围；不要把规划值抄成实测。
10. **W1 Gate 0：** 同源 8B BF16 教师 + TE 逐张量 DelayedScaling FP8。性能主表：`batch=1`，prefill/上下文 `L=2048`，decode 256；先 warmup，再至少 3 次取中位数，报告范围与延迟 P50/P95；吞吐与延迟 `temperature=0`（greedy）、关闭 `top_p`；质量 `--seed 0`。
    ```bash
    source "${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
    conda activate torch2.8_cuda12.8
    cd w1_gate0
    ./run_gate0.sh --configs bf16,te_fp8
    ```
11. **W2 / Gate 1 gather：**
    - 优先：`w2_gather/lut_gather_bench.cu` 已包含 L2 直查、SMEM **64 KB**（`<8>`）与 SMEM **32 KB**（`<4>`）对照，断言失败则不得开跑。在目标机用 `nvcc -O3 -std=c++17 -arch=sm_89` 编译后运行。
    - 已验收的 `w2_gather` 默认合法瓦片仍是 64 KiB，gate/up `SPLIT_K=8`，down `SPLIT_K=28`，每块 8 packed groups，grid 448，block=256。`--opt-half2` / `--opt-bank-replicate` 默认关闭。该二进制目前只有 64 KB 瓦片；若 Gate 1 还要用它做 32 KB 对照，见 `NOTES_W2_32K_PATCH.md`（连接后的小补丁，不扩展成内核开发）。
    ```bash
    ./w2_gather/scripts/run_4090.sh
    ```
12. **W3 单层离线 PQ：** 仅当主机上已有本地校准 / 评测 jsonl（默认 `/data/te-faiss-week1/calib/w3_calib_seed42.jsonl` 与 `w3_eval_seed43.jsonl`）时才跑正式对照。不要编造，也不要从网上拉这些语料。
    ```bash
    ./w3_pq/run_w3_pq.sh --dry-run
    # 路径就绪后再：
    ./w3_pq/run_w3_pq.sh --groups all
    ```

## 3. 明确不在当天默认范围

- 完整 CUDA 解码内核、融合 Epilogue、CUDA Graph、70B、把 48G 当 24G、未批准的量化包、TE 2.19 回退、冻结栈升级。
- 默认不下载 Qwen2.5-32B / Gemma 2 27B，不在 `torch2.8_cuda12.8` 里安装 AWQ/GPTQ/AQLM/GGUF。32B 对 AWQ 的显存对照见 `QWEN32B_VS_AWQ_PREP.md`，须另开隔离环境并另批。
- 不要在共享盒的 Tesla P4 上测数。

非 24GB 或改装卡上的任何结果，都不能写入 V2.8 官方 24GB 验收。
