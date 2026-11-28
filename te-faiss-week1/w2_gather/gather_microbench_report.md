# W2 gather 微基准报告（模板）

> 修订：2026-11-26（Asia/Shanghai）。已同步官方 RTX 4090 24GB 口径。本副本保留 `scripts/run_4090.sh` / `src/gather_microbench.cu` 的记录栏，正式 Gate 1 实验①以同目录 `lut_gather_bench.cu` 为准。
>
> 状态：模板已写，**尚未在官方 RTX 4090 24GB 上实测**。所有数值栏保持「待实测」或留空。禁止填入 H100 的 1.2 ms/token，也禁止把计划中 SMEM 路径 0.5–0.7 ms 的**估算**、或旧 48G 主机数字写成官方 24GB 已测结果或停线。
>
> 正式填写来源：官方 24GB 机上按冻结 CUDA 12.8 编译并运行 `lut_gather_bench.cu`。本目录 `scripts/run_4090.sh` 得到的 `out/gather_microbench.json` 与可选的 `scripts/run_ncu.sh` 若采用，也须来自同一官方 24GB 机，且不得写成与正式副本冲突的验收口径。
>
> 官方验收只认官方 RTX 4090 24GB（sm_89）。以更新后的官方 24GB 锁定表 `VERSION_LOCK.md`（2026-11-26 22:06）为准，旧 48G 记录见 `VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126`；软件钉死项（含冻结 CUDA 12.8）保持不变。形状为官方 Llama 3 8B，decode 先 batch=1，先 warmup 再取重复运行中位数。S2 按 V2.8 原文 2 ms/token（只针对 gather 主体），不是 H100 的 1.2 ms，也不是本卡已测停线。

## 0. 运行元数据（待填）

| 项 | 值 |
| --- | --- |
| 日期（Asia/Shanghai） | 待实测 |
| 主机 / GPU 名称 | 待实测（须为官方 RTX 4090 24GB，sm_89；改装 48G 或其他卡不得作为 V2.8 正式验收） |
| 计算能力、SM 数、显存 MiB | 待实测（显存抄录 MiB，不要用 bytes/1e9）。`lut_gather_bench.cu` 硬检查已落地（2026-11-26 22:10）：接受 24000–25000 MiB，典型约 24564 MiB；≥40000 MiB / 改装 48G 立即失败。须记录硬检查通过或未通过（未通过不得作为 V2.8 正式验收） |
| 驱动、CUDA 工具包 | 待实测（构建必须使用冻结 CUDA 12.8，`/usr/local/cuda-12.8`，`-arch=sm_89`；驱动以新主机实测为准） |
| 可执行文件与 git 提交 | 待实测（W0 脚本由开发总管推送，本目录当时可能尚未入仓） |
| `--warmup` / `--iters` / `--layers` / `--seed` | 建议 5 / 11 / 32 / 42 |
| 本次查找次数 / 全量 1,409,286,144 | 待实测 |
| 计时前 L2 vs SMEM 部分和核对 | 待实测（默认开启；通过则记录 max_abs / max_rel；失败则无耗时。不是 Nsight，也不是编写时已在 4090 上跑过） |

## 1. 工作负载定义

- 检查点形状：官方 Meta-Llama-3-8B（`hidden=4096`，`intermediate=14336`，`layers=32`），不是 3.1/3.2。
- 查找次数出处：计划 B2.1，`FFN 参数量 / 4 = 5,637,144,576 / 4 = 1,409,286,144 ≈ 1.41×10^9` 次/token。
- 输入：随机 uint8 码（`[M/16, N]` 的 `uint4` 交错布局）+ 随机有限 FP16 LUT；每层独立缓冲，避免把同一张表打热 L2。
- 正式路径（`lut_gather_bench.cu`）：`codes_only`、`lut_l2`、`lut_smem<8>`（64 KB，8 groups/block）、`lut_smem<4>`（32 KB，4 groups/block）；block=256。`codes_only` 只在 8 groups/block 网格上测，与 SMEM 32 KB（4 groups/block）对照须注明网格不同。
- 本目录另一份程序（`src/gather_microbench.cu`）路径：L2 直查 vs SMEM 切片。累加并写回 Split-K 部分和，仅防止计算被优化掉；**没有**融合 Epilogue、残差小核或 CUDA Graph。正式验收以 `lut_gather_bench.cu` 为准。
- 换算：`ms/token = median_ms × (1,409,286,144 / 本次实际查找次数)`。默认 32 层全矩阵时二者相等。

## 2. 4090 切片参数（与 H100 估算的差异）

详见 README 第 2 节。摘要：

| | H100 计划估算（B2.1 / 4.1） | 本微基准 4090 合法默认 |
| --- | --- | --- |
| 每 SM / 每块 SMEM | 228 KB / 227 KB（CC 9.0） | 100 KB / 99 KB（CC 8.9，CUDA Programming Guide Table 34；Ada Tuning Guide） |
| gate/up SPLIT_K、切片 | 4，128 KiB | **8，64 KiB** |
| down SPLIT_K、切片 | 16，112 KiB | **28，64 KiB** |
| 网格（block=256） | 224 / 256 | 448 / 448（≥ 公开 4090 的 128 SM；以主机查询为准） |

H100 的 128/112 KiB 切片超出 4090 单块 99 KB 上限，不能启动。64 KiB 是 Ada 允许的 carveout 档位，超过 48 KB 静态上限，启动前已做动态 SMEM opt-in。

## 3. 结果表（L2 vs SMEM，阈值位留空）

耗时单位：CUDA Event，warmup 之后的重复运行统计。未跑之前全部为「待实测」。

| 路径 | 中位数 ms（本跑） | 换算 ms/token | ns/lookup | min / p25 / p75 / max ms | L2 流量 | bank conflict |
| --- | --- | --- | --- | --- | --- | --- |
| L2 直查 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测（ncu / CUPTI；不要把 B2.1 的 ≈22 GB 估算填进本栏） | 不适用或待实测 |
| SMEM 切片（4090 64 KiB 瓦片） | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测（Nsight）；静态地址迹象见下 |

原始 samples 列表：待实测（写入 JSON `paths.*.samples_ms`）。

静态地址模式迹象（非 Nsight，程序可能打印 mean/max replay degree）：待实测。计划 B2.1 在 H100 上估算随机码约 3–4 路冲突，**不是** 4090 实测。

## 4. S2（V2.8 原文口径；2 ms 不是本卡已测停线）

S2 按 V2.8 原文：8B FFN 查表 > 2 ms/token 且无法缓解。该 2 ms 只针对 gather 主体（不含建表、FWHT、残差、epilogue、attention、输出头）。它不是本卡已测停线。**禁止**把 H100 的 1.2 ms 或 0.5–0.7 ms 估算填进本栏。未触发停机线不等于批准内核开工。同卡 TE FP8 基线仍须在官方 24GB 机实测，供日后 Gate 3 对照，本周只记录。

| 校准项 | 状态 | 数值 |
| --- | --- | --- |
| 本次 gather 主体中位数 ms/token（正式路径） | 待实测 | （留空） |
| 是否超过 V2.8 原文 2 ms/token 且无法缓解 | 待实测 | （只针对 gather 主体；2 ms 不是已测停线） |
| 同卡 TE 逐张量 DelayedScaling 的 decode / FFN 基线（中位数；prefill 与 decode 分列，来自 W1） | 待实测 | （只记录，不改写 S2） |
| 未触发停机线是否等于批准内核开工 | 否 | 否 |

## 5. 配置核对清单（跑完勾选）

- [ ] 设备为官方 RTX 4090 24GB，sm_89；显存以 MiB 抄录；未把改装 48G、Tesla P4 或其它非目标卡写成正式验收
- [ ] 编译命令使用冻结 CUDA 12.8（`/usr/local/cuda-12.8`）与 `-arch=sm_89`
- [ ] 正式验收使用 `lut_gather_bench.cu`；`codes_only` 在 8 groups/block 网格上测，与 SMEM 32 KB（4 groups/block）对照已注明网格不同
- [ ] 若另附 `src/gather_microbench.cu` 结果：使用的是 64 KiB / SPLIT_K=8（gate/up）与 28（down），没有强行加载 128/112 KiB，且来自同一官方 24GB 机
- [ ] 官方 Meta-Llama-3-8B 形状，查找次数按 1,409,286,144 换算
- [ ] 计时前部分和核对（若该程序默认开启）已通过（或附上失败日志；未使用 `--skip-partial-check` 来交差）
- [ ] 先 warmup 再取中位数；batch=1
- [ ] 未启用未批准优化开关
- [ ] 未把 H100 的 1.2 ms、0.5–0.7 ms 估算、或 48G 数字写入第 3、4 节的「已测」或「停线」栏
- [ ] 未把「未触发 2 ms」写成批准内核开工
- [ ] L2 / Nsight 栏：有日志就附路径，没有就保持待实测

## 6. 附注与缺项

- 部分和核对阈值是解析界限（`abs=2^{-10}`，`rel=2×2^{-10}`），不是 4090 实测误差，也不是 Nsight。
- 96 次/token 的内核启动开销计入本周口径；CUDA Graph 为后续阶段。
- Nsight metric 名取决于主机 ncu 版本，若 `run_ncu.sh` 失败，把报错附在这里并保持待实测。
- 本微基准不加载 `/data/models/Meta-Llama-3-8B`，不用真实权重。
