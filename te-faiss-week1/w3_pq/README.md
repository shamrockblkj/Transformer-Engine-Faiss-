# 第 1 周 W3：单层 FFN 离线 PQ 四组对照

撰写日期：2026-11-26（Asia/Shanghai）。
修订：2026-11-26（Asia/Shanghai）。按 V2.8 与开发总管分派，将验收口径同步为官方 RTX 4090 24GB。不改 `VERSION_LOCK.md`、`setup_env.sh` 或本目录硬检脚本。

依据：`/workspace/te-faiss-v2.8-plan.md`（V2.8，2026-11-26）Gate 1 单层 PQ 与 S3；评测口径以 [../eval_protocol.md](../eval_protocol.md) 与 [../templates/pq_single_layer_report.md](../templates/pq_single_layer_report.md) 为准。以更新后的官方 24GB 锁定表 [../VERSION_LOCK.md](../VERSION_LOCK.md)（2026-11-26 22:06）为准，旧 48G 记录见 [../VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126](../VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126)。软件钉死项保持不变。新机尚未连接，安装状态仍【待实测】。

本目录只提供**可复现脚本与报告模板**。脚本在共享盒上编写，**不要在本盒跑训练/编码**。激活若在 GPU 上抽取，必须在官方 RTX 4090 24GB 上执行；改装 48G 或非 24GB 卡的结果不得写成 V2.8 正式验收。不要登录主机、不要下载权重、不要推 GitHub、不要索要或写入密码 / SSH 密码 / HF token。不要改 `VERSION_LOCK.md` 或 `setup_env.sh`。

## 1. 实验范围

在官方 Meta-Llama-3-8B 的 **1–2 层真实 FFN**（默认第 0 层与第 16 层，含 `gate` / `up` / `down`，其中 `down` 为建议必做项）上，用 **CPU Faiss** `ProductQuantizer.train` / `compute_codes` 做 8-bit 乘积量化（`dsub=4`，`nbits=8`）：

| 角色 | 权重形状 `[out, in]` | `(d, M)` |
| --- | --- | --- |
| `gate` / `up` | `[14336, 4096]` | `(4096, 1024)` |
| `down` | `[4096, 14336]` | `(14336, 3584)` |

四组对照（有/无按行 RMS × 有/无子空间 4×4 激活二阶矩加权）：

| `--groups` id | 行 RMS（B2.3） | Σ 加权（B2.5） |
| --- | --- | --- |
| `plain` | 否 | 否 |
| `row_rms` | 是 | 否 |
| `sigma` | 否 | 是 |
| `row_rms_sigma` | 是 | 是 |

列缩放（方案 2.1）与 FWHT 旋转（方案 2.2）是**固定前置**，不列入本周消融。`gate` 与 `up` 共用同一旋转；`down` 默认 7 段独立 2048 维分块 FWHT。本周按**单层单角色单独训练码本**，不做「每 4 层共享」。质心预乘 `1/sqrt(d)` 再存 FP16 的部署打包、残差蒸馏、完整 CUDA 解码内核均不在范围。

**禁止：** 调用 `search()`、构建 `IndexIVFPQ` 或任何 `GpuIndex` / GPU Faiss；安装 AWQ / GPTQ / AQLM 或任何要求 `torch>=2.14` 的包；使用本机 Tesla P4 或把改装 48G 结果写成正式验收；把 H100 的 1.2 ms 写进官方 24GB 停线；编造 MSE / KL 数字。第 1 周官方验收模型是 Llama 3 8B；32B 扩展须等 8B 通过 Gate 3；70B 不进单卡测试。

**本实验不能单独用来判定 S3。** S3 要等同体积的全模型对照。本周也不做全模型蒸馏，不做三组精度路线的完整 A/B。出口只是：判断行归一化和加权 k-means 是否值得写进后续契约，给阶段 2 提供是否继续的依据。

## 2. 在目标主机上的跑法

软件钉死项与更新后的官方 24GB 锁定表 `VERSION_LOCK.md` 一致（硬件行不得沿用 48G 备份）。先

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8
```

不要用 `torch2.14_cuda13.0`。Faiss 锁定为 **faiss-cpu 1.16.0**，训练/编码走 CPU；激活抽取与单层 KL 若走 GPU，必须在官方 RTX 4090 24GB（`cuda:0`）上进行。权重只读 `/data/models/Meta-Llama-3-8B` 的 HF safetensors，排除 `original/*`。脚本设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`，并以 `local_files_only=True` 加载，不会去网上拉模型或数据集。

正式测量路径（`run_w3_pq.sh` 默认，不含下面两个例外）会硬检：

- conda 环境名必须是 `torch2.8_cuda12.8`，禁用 `torch2.14_cuda13.0`；
- PyTorch 必须是 `2.8.0+cu128`，`transformers` 必须是 `5.24.1`，`faiss-cpu` 必须是 `1.16.0`（软件钉死项与 `VERSION_LOCK.md` 一致，偶离即退）；
- 官方验收要求官方 RTX 4090 24GB，计算能力 **sm_89**，显存以 MiB 计（接受 24000–25000 MiB，典型约 24564 MiB，以新主机 `nvidia-smi` 为准）。Tesla P4、其它非 24GB 卡、改装 48G、或 `--device cpu` 不得作为正式验收。注意：本目录脚本里的硬件硬检查可能仍只核名称含 `4090` 与 sm_89，本文不改这些脚本；硬检查更新落地前，不要把仅通过「名称含 4090」的 48G 结果写成正式验收。除 `--self-check` / `--dry-run` 外，拒绝用 CPU 跑激活采集或评测。

`--dry-run` 只检查模型/语料路径与配置，不做 GPU 或冻结环境硬检。`--self-check` 只用合成小矩阵（可选极小 Faiss 冒烟），同样跳过 GPU/环境硬检。

### 2.1 准备本地校准 / 评测语料（若尚未放置）

默认路径写在 `configs/default.json`：

- 校准：`/data/te-faiss-week1/calib/w3_calib_seed42.jsonl`
- 评测：`/data/te-faiss-week1/calib/w3_eval_seed43.jsonl`

格式：每行一个 JSON 对象，字段 `text`。第 1 周默认从校准文件中按种子 42 抽 **32 条、截断 512 token**；评测按种子 43 抽 **16 条、截断 512 token**。完整方案口径是 512 条 × 2048 token，本地语料就绪后只改配置里的 `n_sequences` / `max_length`，不要改随机算法。校准与评测必须隔离；默认 `fallback_to_calib=false`，评测文件缺失时**不偷偷拿校准来算 KL**，而是把单层 KL 标为「待准备」。

若文件不在主机上，脚本会失败并说明原因，状态为**待准备**。不要从网上下载，不要提供 token。

### 2.2 建议命令

在本目录执行：

```bash
# 先检查路径与配置（不训练）。退出码 0=路径可用，2=模型目录缺失，3=校准语料待准备。
./run_w3_pq.sh --dry-run

# 数学可逆性 + 若已装 faiss-cpu 则做极小冒烟（不碰真实权重）
./run_w3_pq.sh --self-check

# 正式四组对照（默认 all）
./run_w3_pq.sh --groups all

# 只跑其中几组
./run_w3_pq.sh --groups plain,row_rms
./run_w3_pq.sh --groups sigma,row_rms_sigma
```

其它常用开关：`--skip-kl`（只算重建与输出 MSE）、`--device cuda:0`（正式路径禁止 `cpu`）、`--config /path/to.json`。

`--weights-only` 是**退化通路**：校准语料未就绪时，关闭列缩放与 Σ 加权，只对 `plain` / `row_rms` 报告旋转域重建 MSE。它**不是**正式四组对照，不能用来下 B2.3 / B2.5 的结论。

日志写到 `logs/w3_pq-<时间戳>.log`，结果表与元数据写到 `results/`（含 `run_meta_latest.json`、分角色 CSV、FFN MSE / KL 的 JSON，以及 `results/artifacts/` 下的码字、质心、行尺度）。把实测数字回填到 `pq_single_layer_report.md` 的空白格，**禁止手写估计值**。

Faiss 训练是本实验最耗时的部分：2 层 × 3 个角色 × 4 组 = 24 次 CPU k-means，主机上可能需要数十分钟到数小时。`pq.omp_num_threads` 默认 8，可在配置里改，但不要因此改锁定的软件版本。

## 3. 行 RMS 与 4×4 加权的实现要点

处理顺序（每个 `(layer, role)`）：

1. **列缩放（固定前置）**：用该线性层输入激活的通道 RMS（α=1），按中位数归一化后再钳位到 `[0.05, 20]`，得到 `s`。`W_s = W @ diag(s)`，`x_s = x / s`。本周不把 `s^{-1}` 折进 RMSNorm γ。
2. **FWHT 旋转（固定前置）**：`W_rot = W_s @ R`，`x_rot = R^T x_s`，其中 `R = H D / sqrt(d_block)`，`D` 为配置中种子生成的 ±1 对角。脚本在 self-check 里验证 `W x = W_rot x_rot`。
3. **B2.3 按行 RMS**（分组开关）：在**旋转域**对每一行除以其 RMS，行尺度以 FP16 存储；训练与重建都使用 cast 回 FP32 后的存储值，避免训练/解码不一致。重建后再乘回，对应方案里 Epilogue / down 归约乘回行尺度。
4. **B2.5 子空间加权**（分组开关）：Faiss 本身不支持加权 PQ。对旋转域激活按 4 维子空间累积二阶矩 `Σ_m = E[x_m x_m^T]`（不加中心化）。特征值设下限后取主平方根，把权重子向量右乘 `Σ_m^{1/2}` 再交给 Faiss；得到质心后右乘 `Σ_m^{-1/2}` 变回旋转（及可选行归一化）域。这等价于在子空间内按马氏距离做 k-means，以逼近输出误差 `‖(W-Ŵ)X‖` 而不是纯权重 L2。
5. **Faiss**：仅 `ProductQuantizer(d, M, 8).train` 与 `compute_codes`。重建用 NumPy 按码字取质心，不调用 `decode` / `search`。
6. **指标**：
   - 重建 MSE：主对照在旋转+列缩放域（四组同一参考域）比较 `W_rot` 与乘回行尺度、反白化后的 `Ŵ_rot`；同时报告还原到原始参数域的 MSE。
   - 该层线性输出 MSE：`mean((W x - Ŵ x)^2)`，以及当三个角色都量化时的整块 FFN（SwiGLU + down）输出 MSE。默认在校准激活上计算。**Σ 分组（`sigma` / `row_rms_sigma`）的这两项偏乐观，不得当契约数**；日志、`run_meta_latest.json` 与报告模板均会注明。
   - 单层 KL：教师为完整 BF16 模型；学生只替换目标层的所选 FFN 线性层为重建权重，其余层保持 BF16。单位为 **nats / token**（温度 1，因果移位，mask 掉 pad）。这不是蒸馏。

所有阈值列留空，等实测回填。

## 4. 缺项与待实测

下列项目在脚本或报告里标「待实测」或「待准备」，**不要填虚构数字**：

- 全部重建 MSE、线性 / FFN 输出 MSE、单层 KL：必须来自目标主机实测。
- 校准语料、评测语料、以及 `/data/models/Meta-Llama-3-8B` 是否已在主机就位：待主机确认。缺失则停止对应步骤，不下载。
- 第 1 周默认 32×512 与完整方案 512×2048 的指标差：待语料就绪后用同一脚本重跑，不得把小规模数字直接当成最终契约依据。
- Σ 分组（`sigma` / `row_rms_sigma`）的线性 / FFN 输出 MSE 用校准激活计算，**偏乐观，不得当契约数**；持出激活上的输出 MSE 尚未作为默认步骤（待追加，若需要）。
- 本周不做每 4 层共享码本，共享后的重建质量待后续。
- 列缩放折进 RMSNorm γ、以及 `|γ|≤64` 的契约检查：待后续数学契约，本周只显式应用 `s`。
- 质心 FP32→FP16（及预乘 `1/sqrt(d)`）引起的 KL 增量：待后续单测。
- `down` 在 7 段分块 FWHT 下若重建误差明显偏高，是否改用 `H_28 ⊗ H_512`：待实测后再决定，本周不自动切换。
- Faiss k-means 是权重空间 L2，与 GEMV 内积目标不完全同一，精度影响待实测（计划书审阅补充已指出）。
- 行归一化 / Σ 加权是否写入后续 `quantization_contract.yaml`：等四组表填齐后由开发总管判断，本实验不能单独判定 S3。
- W0 软件钉死项（conda `torch2.8_cuda12.8`、faiss-cpu 1.16.0、transformers 5.24.1 等）保持不变；以更新后的官方 24GB 锁定表为准，旧 48G 记录见备份。新机安装状态仍【待实测】，正式路径软件偶离即退。
- 不得把 H100 1.2 ms、48G 改装卡数字、或任何编造的 4090 停线写进本报告。

## 5. 交付文件

| 路径 | 说明 |
| --- | --- |
| `configs/default.json` | 锁定路径、种子、四组定义、小规模校准默认值 |
| `run_w3_pq.sh` | 主机入口：source conda、离线环境变量、记日志 |
| `run_pq_single_layer.py` | 主程序 |
| `pq_math.py` | 列缩放 / FWHT / 行 RMS / Σ 白化 / 重建 |
| `pq_faiss_cpu.py` | CPU ProductQuantizer 封装 |
| `pq_io.py` | 本地 safetensors 与环境检查 |
| `pq_calib.py` | 本地语料与激活抽取 |
| `pq_metrics.py` | MSE 与单层 KL |
| `pq_single_layer_report.md` | 本目录报告副本，已同步官方 24GB 口径（数字列留空） |
| `results/` `logs/` | 运行输出（本盒保持为空） |
