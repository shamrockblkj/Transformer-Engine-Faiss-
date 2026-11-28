# 第 1 周 W3 报告模板：单层 FFN 离线 PQ 四组对照

- 模板撰写日期：2026-11-26（Asia/Shanghai）
- 修订：2026-11-26（Asia/Shanghai）。已同步官方 RTX 4090 24GB 口径；与 [../templates/pq_single_layer_report.md](../templates/pq_single_layer_report.md) 对齐验收机与环境记录写法。
- 实测填写日期：（待填，Asia/Shanghai）
- 运行日志：（待填，如 `logs/w3_pq-YYYYMMDD-HHMMSS.log`）
- 结果元数据：（待填，默认 `results/run_meta_latest.json`）
- 配置：`configs/default.json`（若有改动，写明实际文件与 diff）

**填写规则：** 所有 MSE / KL / 耗时必须来自官方验收路径的实测。激活若在 GPU 上抽取，必须在官方 RTX 4090 24GB 上进行；改装 48G 或非 24GB 卡的结果不得写成 V2.8 正式验收。不确定的格子保持空白或写「待实测」，禁止编造、禁止填入 H100 的 1.2 ms、禁止用本机 Tesla P4 或旧 48G 主机数字。本周不做全模型蒸馏，也不做三组精度路线的完整 A/B。第 1 周官方验收模型是 Llama 3 8B；32B 扩展须等 8B 通过 Gate 3；70B 不进单卡测试。

**关于 S3：** 本实验不能单独用来判定 S3。S3 要求等同体积的全模型对照（相对 AQLM 2-bit 等成熟方案）。本报告只为判断行归一化（B2.3）与子空间加权 k-means（B2.5）是否值得写入后续契约，给阶段 2 提供是否继续的依据。

## 1. 环境（回填，不改 VERSION_LOCK）

以更新后的官方 24GB 锁定表 `VERSION_LOCK.md`（2026-11-26 22:06）为准，旧 48G 记录见 `VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126`。软件钉死项保持不变。新机尚未连接，须抄录官方 24GB 机实际版本与显存 MiB。

| 项 | 软件钉死 / 官方验收期望 | 本次实测 |
| --- | --- | --- |
| 主机 | 官方 24GB 机（待抄录实际主机名；不得填写晨涧云 `ubuntu-30041` 作为验收机） | （待填） |
| GPU | 官方 RTX 4090 24GB，sm_89；显存以 MiB 计（接受 24000–25000 MiB，典型约 24564 MiB，以新主机 `nvidia-smi` 为准）。改装 48G 或其他卡不得作为 V2.8 正式验收 | （待填，含 `nvidia-smi` 名称与显存 MiB） |
| 驱动 / CUDA 工具包 | 驱动以新主机实测为准；CUDA 工具包须为冻结 12.8 | （待填） |
| conda 环境 | torch2.8_cuda12.8 | （待填） |
| Python | 3.12.0 | （待填） |
| PyTorch | 2.8.0+cu128 | （待填） |
| faiss-cpu | 1.16.0（仅 CPU） | （待填，发行包名与版本） |
| transformers | 5.24.1 | （待填） |
| 模型 | `/data/models/Meta-Llama-3-8B`，仅 HF safetensors，排除 `original/*` | （待填：是否就位） |
| 校准语料 | `/data/te-faiss-week1/calib/w3_calib_seed42.jsonl`，32 条 × 512 token，seed=42 | （待填：路径、实际条数、是否缺失） |
| 评测语料 | `/data/te-faiss-week1/calib/w3_eval_seed43.jsonl`，16 条 × 512 token，seed=43 | （待填；默认不回退到校准集） |

官方验收要求官方 RTX 4090 24GB（sm_89，显存以 MiB 计）。Tesla P4、其它非 24GB 卡、改装 48G 或 `--device cpu` 不得作为正式验收。注意：本目录脚本里的硬件硬检查可能仍只核名称含 4090 与 sm_89，本文不改这些脚本；硬检查更新落地前，不要把仅通过「名称含 4090」的 48G 结果写成正式验收。除 `--self-check` / `--dry-run` 外，不得用 CPU 跑激活采集或评测。conda 环境必须是 `torch2.8_cuda12.8`（禁用 `torch2.14_cuda13.0`），PyTorch / transformers / faiss-cpu 的软件钉死项与 `VERSION_LOCK.md` 偶离即退。PQ 训练与编码仍走 CPU Faiss。

## 2. 方法简述（与脚本一致，回填时不要改定义）

固定前置（四组共用，不消融）：

1. 列缩放：通道 RMS（α=1），中位数归一化后钳位到 `[0.05, 20]`，显式作用于权重列与激活，不折进 RMSNorm γ。
2. 正交旋转：`R = H D / sqrt(d_block)`。`gate`/`up` 共用 4096 维 FWHT；`down` 为 7×2048 分块 FWHT。符号种子见配置 `rotation.sign_seeds`。

四组因子：

- **B2.3 按行 RMS：** 在旋转域对每行除以 RMS，行尺度以 FP16 存储并按存储值乘回。
- **B2.5 子空间 4×4 加权：** 旋转域每个 4 维子空间估计二阶矩 `Σ_m = E[xx^T]`（不减均值），权重子向量先乘 `Σ_m^{1/2}` 再交给 Faiss `train` / `compute_codes`，质心再乘 `Σ_m^{-1/2}` 变回。Faiss 本身不支持加权 PQ。

Faiss：`ProductQuantizer(d, M, 8)`，`niter=20`，`nredo=1`，`seed=1234`。只用 `train` / `compute_codes`。本周单层单角色单独训练，不共享 4 层码本。重建在 FP32 数学域完成，不把质心打包成预乘 `1/sqrt(d)` 的 FP16。

指标定义：

| 指标 | 定义 | 数据 |
| --- | --- | --- |
| 重建 MSE（旋转域，主对照） | `mean((W_rot - Ŵ_rot)^2)`，`Ŵ_rot` 已乘回行尺度并反白化 | 权重本身 |
| 重建 MSE（原始参数域） | `mean((W - Ŵ)^2)` | 权重本身 |
| 线性输出 MSE | `mean((W x - Ŵ x)^2)`，FP32 | 默认：该层校准激活。**`sigma` / `row_rms_sigma` 两列偏乐观，不得当契约数** |
| FFN 输出 MSE | 同一分组替换该层 `gate`+`up`+`down` 后，SwiGLU FFN 输出相对 BF16 教师的 MSE | 默认：该层校准激活（FFN 输入）。**`sigma` / `row_rms_sigma` 两列偏乐观，不得当契约数** |
| 单层 KL | 只替换目标层所选 FFN 线性层，其余保持 BF16；next-token `KL(teacher ‖ student)`，单位 nats/token，温度 1，因果移位，mask pad | 评测语料 |

阈值一律留空，不把任何候选门禁（例如 KL 0.05 nat）写成已通过或已失败。填写第 3 节时：`sigma` / `row_rms_sigma` 的线性输出 MSE 与 FFN 输出 MSE 格子若回填了校准激活上的数字，须在格内或表下注明「偏乐观，不得当契约数」。

## 3. 四组总表（主结果，数字留空）

先按层填写。默认层号为 0 与 16，若实测改层，改表头并在第 7 节说明。下列线性输出 MSE 与整块 FFN 输出 MSE 均来自校准激活；其中 **`sigma` / `row_rms_sigma` 偏乐观，不得当契约数**。

### 3.1 第 0 层

**重建 MSE（旋转域）**

| 角色 | plain | row_rms | sigma | row_rms_sigma |
| --- | --- | --- | --- | --- |
| gate |  |  |  |  |
| up |  |  |  |  |
| down |  |  |  |  |

**线性输出 MSE**

| 角色 | plain | row_rms | sigma | row_rms_sigma |
| --- | --- | --- | --- | --- |
| gate |  |  |  |  |
| up |  |  |  |  |
| down |  |  |  |  |

**整块 FFN 输出 MSE**

| 分组 | FFN 输出 MSE | cosine | 相对 MSE |
| --- | --- | --- | --- |
| plain |  |  |  |
| row_rms |  |  |  |
| sigma |  |  |  |
| row_rms_sigma |  |  |  |

**单层 KL（nats/token；替换该层 gate+up+down）**

| 分组 | KL 均值 | KL 中位数 | 有效 token 数 |
| --- | --- | --- | --- |
| plain |  |  |  |
| row_rms |  |  |  |
| sigma |  |  |  |
| row_rms_sigma |  |  |  |

### 3.2 第 16 层

**重建 MSE（旋转域）**

| 角色 | plain | row_rms | sigma | row_rms_sigma |
| --- | --- | --- | --- | --- |
| gate |  |  |  |  |
| up |  |  |  |  |
| down |  |  |  |  |

**线性输出 MSE**

| 角色 | plain | row_rms | sigma | row_rms_sigma |
| --- | --- | --- | --- | --- |
| gate |  |  |  |  |
| up |  |  |  |  |
| down |  |  |  |  |

**整块 FFN 输出 MSE**

| 分组 | FFN 输出 MSE | cosine | 相对 MSE |
| --- | --- | --- | --- |
| plain |  |  |  |
| row_rms |  |  |  |
| sigma |  |  |  |
| row_rms_sigma |  |  |  |

**单层 KL（nats/token；替换该层 gate+up+down）**

| 分组 | KL 均值 | KL 中位数 | 有效 token 数 |
| --- | --- | --- | --- |
| plain |  |  |  |
| row_rms |  |  |  |
| sigma |  |  |  |
| row_rms_sigma |  |  |  |

## 4. 明细补充（可选回填）

原始参数域重建 MSE、逐行相对误差中位数、列尺度钳位命中数量、Σ 被特征值下限截断的个数、各次 Faiss 训练墙钟时间，从 `results/pq_single_layer_results_*.csv` 与 `results/artifacts/` 抄录。无实测则保持「待实测」。

| 层 | 角色 | 分组 | 原始域重建 MSE | 行相对误差中位数 | Σ 截断特征值数 | train 秒数 |
| --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |  |

## 5. 对照解读占位（有数字再写，禁止提前下结论）

- 行 RMS 相对 `plain` 是否稳定降低旋转域重建误差与输出 MSE / 单层 KL：（待实测）
- Σ 加权相对 `plain` 是否降低**输出**误差（即使权重 L2 不一定更好）：（待实测）
- 两者叠用是否明显优于单用：（待实测）
- `down` 是否明显差于 `gate`/`up`，是否需要在后续改用 `H_28 ⊗ H_512`：（待实测）
- 是否建议把行尺度、加权 k-means 写入第 2 周契约：（待实测后由开发总管判断）

不得根据本表宣布 S3 通过或失败。

## 6. 口径偏差与限制

- 本周默认校准 / 评测规模小于完整方案的 512×2048，小规模数字不能直接当作最终契约依据。
- 线性 / FFN 输出 MSE 默认用**校准激活**。Σ 分组（`sigma` / `row_rms_sigma`）的这两项**偏乐观，不得当契约数**（脚本日志与 `run_meta_latest.json` 的 `sigma_group_output_mse_caveat` 亦写明）。
- 评测语料若缺失且未授权回退，单层 KL 为「待准备」。若有人把 `fallback_to_calib` 改为 true，必须在此注明「评测与校准未隔离」。
- 未做每 4 层码本共享、未做质心 FP16 打包、未折 γ、未蒸馏。
- Faiss 优化目标仍是白化（或未白化）后的权重 L2，与在线 GEMV 内积不完全同一。

## 7. 缺项 / 异常 / 待准备

| 项 | 状态 | 说明 |
| --- | --- | --- |
| 模型权重就位 | 待准备 / 待实测 |  |
| 校准语料就位 | 待准备 / 待实测 | 缺失则不得下载 |
| 评测语料就位 | 待准备 / 待实测 | 默认不回退 |
| faiss-cpu 1.16.0 可导入 | 待实测 | 以主机锁定快照为准 |
| 四组数字回填 | 待实测 |  |
| 512×2048 重跑 | 待准备 |  |
| 持出激活上的输出 MSE | 待准备 |  |
| S3 全模型对照 | 本周不做 |  |

其它异常（失败日志路径、OOM、版本不符导致退出）写在这里。失败即停止，不要改锁定主版本，不要自动改用 TE / PyTorch / Faiss 的其它版本。
