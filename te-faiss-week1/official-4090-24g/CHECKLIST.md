# 官方 4090 24GB 第一天工具包清单（给开发总管）

日期：2026-11-26（Asia/Shanghai）。规划基线：V2.8（`/workspace/te-faiss-v2.8-plan.md`）。

## 已准备

| 路径 | 说明 |
| --- | --- |
| `official-4090-24g/assert_official_4090_24g.sh` | 共享硬件断言：单卡、名称含 RTX 4090、sm_89、显存 24000–25000 MiB（典型约 24564）；≥40000 / 约 49140 立即失败 |
| `official-4090-24g/create_conda_env.sh` | 创建/激活 `torch2.8_cuda12.8`（Python 3.12.0）并安装 `torch==2.8.0+cu128` |
| `official-4090-24g/download_llama3_8b.sh` | 仅在新主机上下载官方 Meta-Llama-3-8B（约 16 GB，排除 `original/*`，镜像默认为 hf-mirror） |
| `official-4090-24g/download_wheelhouse.sh` | 预下载冻结栈轮子（可重入） |
| `official-4090-24g/wheelhouse/` | 本地轮子缓存（含 torch 2.8.0+cu128 与 cuDNN 9.28.1.4；可能另有 torch 元数据的 9.10.2.21，不得作为最终引脚） |
| `official-4090-24g/wheelhouse-constraints.txt` | 冻结引脚备查；求解时仍须省略 cuDNN 行 |
| `official-4090-24g/measure_bandwidth.py` | 实测 D2D 带宽，替换规划值 0.9 TB/s |
| `official-4090-24g/measure_usable_vram.py` | 实测可用显存（MiB），替换约 23.5 GB 假设 |
| `official-4090-24g/DAY1_RUNBOOK.md` | 简体中文当日顺序 |
| `official-4090-24g/NOTES_W2_32K_PATCH.md` | `gather_microbench` 若需 32 KB 对照的小补丁说明 |
| `official-4090-24g/QWEN32B_VS_AWQ_PREP.md` | 32B vs AWQ 只准备、不作为默认 W0 |
| `official-4090-24g/history/` | 改装 48G 锁文件与脚本备份 |

## 已把官方路径从「48G 硬检」改为「官方 24GB」

下列已验收脚本/配置已改，使 **24GB 能通过、48G 会失败**（不再出现「拒绝把 24G 当本机」，`setup_env.sh` 不再对非 49140 只警告后继续）：

- `setup_env.sh`（优先调用共享断言，失败即停止）
- `VERSION_LOCK.md`（GPU 行改为官方 24GB；软件引脚未改）。备份：`VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126`
- `w1_gate0/configs/protocol.yaml`、`protocol.json`
- `w1_gate0/gate0_common.py`、`run_gate0.sh`
- `w2_gather/lut_gather_bench.cu`（显存按 **MiB** 检查，不用 `totalGlobalMem/1e9`；断言失败则不开跑）
- `w2_gather/src/gather_microbench.cu`、`scripts/run_4090.sh`
- `w3_pq/pq_io.py`、`configs/default.json`、`run_w3_pq.sh`

未改：`eval_protocol.md`、`templates/*`（文档评审方维护）。W1/W2/W3 README 与目录内报告副本未改，避免与评审同步冲突。

## 冻结引脚（未改）

Python 3.12.0；`torch2.8_cuda12.8`；PyTorch 2.8.0+cu128；CUDA 12.8；TE 2.21.0；hub 2.4.0；faiss-cpu 1.16.0；transformers 5.24.1；accelerate 1.16.0；datasets 5.2.0；lm-eval 0.4.13；tokenizers 0.23.3；safetensors 0.8.0；sentencepiece 0.2.3；peft 0.22.0。cuDNN 最终 9.28.1.4，`version()≥91200`。默认不装量化包。

## 明天仍需要用户提供

- 新主机的 SSH / IP / 端口（不是昨天的改装机，不要用 175.155.64.171）
- 确认 `nvidia-smi` 为官方 24GB（24000–25000 MiB）
- 若官方 Meta-Llama-3-8B 仍 gated：只在环境变量中提供 HF token，不要写入仓库或本包
- conda 实际路径若不是 `~/anaconda3/etc/profile.d/conda.sh`，设置 `CONDA_SH`
- W3 正式跑还需要本地 calib/eval jsonl；没有则不要从网上拉，也不要编造

## 明确不在范围

完整 CUDA 解码内核；70B；把 48G 当 24G；软件显存上限当官方验收；未批准量化包；TE 2.19 回退；冻结栈升级；默认下载 32B / 安装 AWQ；把 V2.8 第 2/7/10 节估算当实测；目标任务（仍待补）。
