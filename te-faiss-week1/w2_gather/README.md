# W2 gather 吞吐微基准（L2 直查 vs SMEM 切片）

第 1 周 W2 / Gate 1 实验①：只测「查表 gather」吞吐，不写完整解码内核。
修订：2026-11-26（Asia/Shanghai）。按 V2.8 与开发总管分派，将验收口径同步为官方 RTX 4090 24GB。不改 `VERSION_LOCK.md`、`setup_env.sh`、`lut_gather_bench.cu`、`src/gather_microbench.cu` 或本目录脚本。

- 官方验收机：官方 RTX 4090 24GB（sm_89），CUDA 12.8（`/usr/local/cuda-12.8`）。改装 48G 或非 24GB 卡的结果不得写成 V2.8 正式验收。不要用本机 Tesla P4。
- 正式微基准：本目录 `lut_gather_bench.cu`（开发总管 2026-11-26 已接受为正式 Gate 1 实验①）。编译必须使用冻结 CUDA 12.8。sm_89 与官方 24GB 显存硬检查已写入该正式副本（mtime 2026-11-26 22:10，Asia/Shanghai；按 MiB，接受 24000–25000 MiB，典型约 24564 MiB；≥40000 MiB / 改装 48G 立即失败）。本文不改 `.cu`。
- 本目录另有 `src/gather_microbench.cu` 与 `scripts/run_4090.sh`。若采用其结果，也须在官方 24GB 机上取得，且不得写成与正式副本冲突的验收口径。
- 模型形状：官方 **Meta-Llama-3-8B**（不要改用 3.1 / 3.2）
- 输入：随机 `uint8` 码 + 随机 FP16 LUT
- 禁止把 H100 的 1.2 ms、V2.8 估算 0.5–0.7 ms、或旧 48G 主机数字写成官方 24GB 停线或已测结果。

依据：`/workspace/te-faiss-v2.8-plan.md`（V2.8，2026-11-26）第 4.1 节、S2 与 Gate 1；评测口径以 [../eval_protocol.md](../eval_protocol.md) 与 [../templates/gather_microbench_report.md](../templates/gather_microbench_report.md) 为准。以更新后的官方 24GB 锁定表 [../VERSION_LOCK.md](../VERSION_LOCK.md)（2026-11-26 22:06）为准，旧 48G 记录见 [../VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126](../VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126)。软件钉死项（含冻结 CUDA 12.8）保持不变。

## 1. 查找次数与形状（出处）

官方 Meta-Llama-3-8B `config.json`：`hidden_size=4096`，`intermediate_size=14336`，`num_hidden_layers=32`。

方案 PQ：`dsub=4`、256 中心。每层 FFN：

| 矩阵 | N（输出通道） | M = 输入维 / 4 | 每层查找次数 N×M | LUT 字节（FP16） |
| --- | ---: | ---: | ---: | ---: |
| gate / up | 14336 | 1024 | 14,680,064 | 524,288 |
| down | 4096 | 3584 | 14,680,064 | 1,835,008 |

每 token 查找次数（计划 B2.1）：

```
FFN 参数量 = 32 × 3 × 4096 × 14336 = 5,637,144,576
查找次数   = FFN 参数量 / 4     = 1,409,286,144  ≈ 1.41×10^9
```

假设：`batch=1` 解码、一层一次 GEMV、每个参数对应一次 4 维子空间查表。预填不在本微基准范围内。码布局与计划 4.1 一致：`[M/16, N]` 的 `uint4`（16 个 uint8），保证 warp 内连续 512 B 合并读码。

换算（脚本会打印并写入 JSON）：

```
ms/token = median_ms × (1,409,286,144 / 本次实际查找次数)
```

默认 `--layers 32` 且 gate+up+down 全开时，分母等于 1,409,286,144，中位数毫秒值就是 ms/token。若缩短层数，必须用上式换算，不得直接当全模型数字。

## 2. 4090 SMEM 切片如何从 H100 估算调整

计划 B2.1 / 4.1 按 **H100 228 KB SMEM** 写的默认切片：

| 矩阵 | H100 SPLIT_K | 每块 packed 组 | 切片 | 网格（block=256） |
| --- | ---: | ---: | ---: | ---: |
| gate / up | 4 | 16 | 16×16×256×2 B = **128 KiB** | 224 |
| down | 16 | 14 | 14×16×256×2 B = **112 KiB** | 256 |

这两档在 4090 上不合法。出处：

- CUDA Programming Guide「Compute Capabilities」Table 34（查阅于 2026-11-26）：计算能力 **8.9** 每 SM 共享内存上限 **100 KB**，每 block 上限 **99 KB**（表中 KB = 1024 B）。[文档](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)
- Ada Tuning Guide：Ada 统一 L1/共享内存 128 KB，carveout 可选 0 / 8 / 16 / 32 / 64 / 100 KB；CUDA 每 block 预留 1 KB，故单块最多 99 KB；超过 48 KB 必须用动态共享内存并显式 opt-in。[Ada Tuning Guide PDF](https://docs.nvidia.com/cuda/pdf/Ada_Tuning_Guide.pdf)

128 KiB 与 112 KiB 都大于 99 KB，不能沿用。

本微基准在 **保持 block=256（8 warp）**、**`M/16` 可被 SPLIT_K 整除**、**网格 ≥ 公开 RTX 4090 的 128 SM**（16384 CUDA 核 ÷ 每 SM 128 核；产品页 CUDA 核数，运行时仍查询 `multiProcessorCount`）的前提下，把每块 LUT 切片降到 **64 KiB**（Ada 合法 carveout）：

| 矩阵 | 4090 SPLIT_K | 每块 packed 组 | 切片 | 网格 |
| --- | ---: | ---: | ---: | ---: |
| gate / up | **8** | 8 | 8×16×256×2 B = **64 KiB** | 448 |
| down | **28** | 8 | 8×16×256×2 B = **64 KiB** | 448 |

部分和：gate `8 × 14336 × 4 B = 448 KiB`，down `28 × 4096 × 4 B = 448 KiB`，只作防止编译器消除计算的写回，不是融合 Epilogue。

启动时会核对设备 opt-in SMEM 上限；若切片放不进去或算力 < sm_89 / 名称像 Tesla P4，直接报错退出。非 4090 的其它 sm_89 卡必须显式 `--allow-non-target-gpu`，且不得把该卡数字写进 4090 的 S2 停线。

## 3. 在 4090 主机上编译与跑法

不要在本箱子或 Tesla P4 上编译、跑数。目标主机：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
# 若后续还要进 conda 环境，非交互 shell 先：
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate torch2.8_cuda12.8

cd /workspace/te-faiss-week1/w2_gather
make
# 或一键（编译 + 默认测量，写 JSON）：
./scripts/run_4090.sh
```

默认测量口径（warmup 后取重复中位数；decode、batch=1；软件钉死项与更新后的官方 24GB 锁定表一致，硬件行不得沿用 48G 备份）：

```bash
./out/gather_microbench \
  --path both \
  --warmup 5 \
  --iters 11 \
  --layers 32 \
  --seed 42 \
  --json ./out/gather_microbench.json
```

只看配置、不启动内核：

```bash
./out/gather_microbench --print-config-only
```

可选 Nsight（L2 流量、SMEM bank conflict）。没有 `ncu` 就保持「待实测」，不要用计划里的 22 GB 估算冒充实测：

```bash
./scripts/run_ncu.sh
```

失败时应看到明确错误（找不到 nvcc / 没有 GPU / 算力不足 / SMEM 放不下 / L2 与 SMEM 部分和不一致 / ncu 缺失或 metric 名不被接受），而不是静默写一个数字。

默认在正式 warmup / 计时**之前**会做一次 L2 直查与 SMEM 切片的 Split-K 部分和一致性核对（同一组随机 uint8 码与 FP16 LUT，覆盖本次所有矩阵与层）。通过则打印 `PASS` 再进入计时；不一致则打印最大绝对/相对误差及最差位置，以退出码 4 结束，**不记录耗时**。这不是 Nsight，也不是 4090 上已跑过的结果。

排障时可加 `--skip-partial-check` 关闭核对，但默认必须开启。默认命令行不用改。`scripts/run_ncu.sh` 会带上该开关，避免核对启动的另一条路径污染分路径的 L2 流量统计；正式计时仍以 `run_4090.sh` / 默认可执行文件为准（核对开启）。

通过判据（解析界限，**不是**主机实测误差）：

```
|L2 - SMEM| <= abs_tol + rel_tol * max(|L2|, |SMEM|)
abs_tol = 2^{-10}       ≈ 9.765625e-4   （binary16 在 1.0 处的 ulp）
rel_tol = 2 * 2^{-10}   ≈ 1.953125e-3   （两个 binary16 machine epsilon）
```

依据：两条路径都把 FP16 LUT 项经 `__half2float`（提升到 FP32 是精确的）按相同码顺序做 FP32 累加，每个部分和是 128 项（`packed_per_split=8 × 16`）。预加载正确时应逐元素一致；阈值只为良性代码生成差异留出解析余量。本发生器下单项 LUT 量级约 0.5–8（指数位 14–17），下标算错通常远超此带。不要把该阈值当成 4090 实测误差。

## 4. 记录什么

- 计时前部分和核对：默认开启；通过信息或失败时的最大绝对/相对误差（失败则无耗时）
- 耗时：CUDA Event，warmup 之后、重复运行的**中位数**，并给出 min / p25 / p75 / max / mean
- 换算后的 ms/token、ns/lookup
- L2 流量：默认留空为待实测；用 `scripts/run_ncu.sh` 或同等 CUPTI/Nsight 接口补。拿不到就保持待实测
- bank conflict 迹象：二进制会打印一层随机码的**静态地址模式估算**（Ada 32 bank，FP16 下标 `i` 落在 bank `(i/2)%32`），这只是迹象，不是 Nsight 实测。Nsight 列同样待补
- 设备查询：名称、sm、SM 数、SMEM 上限、L2 大小

## 5. 明确不做 / 默认关闭

- 完整解码 GEMV 产品化、融合 Epilogue、残差小核、CUDA Graph、L2 持久化窗口：本周不做
- `--opt-half2`、`--opt-bank-replicate`：开关保留，**默认关闭**。本周未批准，打开后程序拒绝运行并说明原因（64 KiB 再按 bank 复制会超过 99 KB，需要再加大 SPLIT_K 才能合法）。未批准可不做
- `--skip-partial-check`：仅供排障，**默认关闭（即默认做核对）**
- 不下载权重、不登录脚本里写死的密码或令牌、不推 GitHub
- 不安装 AWQ / GPTQ / AQLM，不依赖 Transformer Engine / Faiss / PyTorch（本程序只链 CUDA runtime）

一次 token 等价测量包含 32×3=96 次内核启动，启动开销计入 ms/token。用 CUDA Graph 消除启动开销是后续阶段，本周不启用。

## 6. 交付文件

| 路径 | 说明 |
| --- | --- |
| `lut_gather_bench.cu` | 正式 Gate 1 实验①微基准（开发总管已接受）。网格须对齐 V2.8 §4.1；`codes_only` 只在 8 groups/block 网格上测，与 SMEM 32 KB（4 groups/block）对照须注明网格不同。sm_89 + 官方 24GB（MiB，24000–25000）硬检查已落地（2026-11-26 22:10） |
| `src/llama3_8b_shapes.h` | 官方形状、查找次数、H100 对照切片与 4090 合法切片 |
| `src/gather_microbench.cu` | 目录内另一份可编译微基准；非正式验收默认副本 |
| `Makefile` | `nvcc -arch=sm_89`，默认 `CUDA_HOME=/usr/local/cuda-12.8` |
| `scripts/run_4090.sh` | 编译并跑 `src/gather_microbench.cu` 的默认脚本（非正式验收默认入口） |
| `scripts/run_ncu.sh` | 可选 Nsight；缺工具则失败并保持待实测 |
| `gather_microbench_report.md` | 本目录报告副本，已同步官方 24GB 口径 |
| `out/` | 构建产物与 JSON（运行后生成） |

## 7. 缺项与待实测

- 本目录在编写时**未在官方 24GB 机上编译或跑数**，所有耗时、L2 流量、Nsight bank conflict、部分和核对结果均为「待实测」
- S2 按 V2.8 原文：8B FFN 查表 > 2 ms/token 且无法缓解。2 ms 不是本卡已测停线，且只针对 gather 主体；通过了也不等于批准内核开工。禁止填 H100 的 1.2 ms，也禁止把 0.5–0.7 ms 估算写成已测
- `lut_gather_bench.cu` 硬检查已落地（2026-11-26 22:10）：显存按 `totalGlobalMem / (1024²)` 打印与判定整数 MiB；须单卡、名称含 RTX 4090（排除 4090D）、sm_89、显存 24000–25000 MiB，否则 FATAL 并退出码 2。≥40000 MiB（改装 48G）立即失败。本文不改源码
- `run_ncu.sh` 的 metric 名需在主机 ncu 版本上确认；被拒就改名重跑，不要编造字节数
- 官方 24GB 4090 的 SM 数、L2 容量以运行时 `cudaGetDeviceProperties` 为准（公开规格为 128 SM、L2 72 MB；若查询结果不同，以主机为准并写入报告）。显存验收以 MiB 计，不要用 bytes/1e9
- 每层使用独立随机 LUT/码，避免「同一张表反复打热 L2」的乐观偏差；这仍不是真实权重，只是 gather 模式
