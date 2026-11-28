# 第 1 周 W1：Gate 0 测量脚本

撰写日期：2026-11-26（Asia/Shanghai）。
修订：2026-11-26（Asia/Shanghai）。按 V2.8 与开发总管分派，将验收口径同步为官方 RTX 4090 24GB；不改 `VERSION_LOCK.md`、`setup_env.sh`、`configs/protocol.yaml`、`configs/protocol.json`、`gate0_common.py`。

本目录只提供**可复现脚本与报告模板**，不在本箱子上跑测量，也不登录主机、不下载模型权重、不推 GitHub、不索要或写入密码 / SSH / HF token。不要改 `../VERSION_LOCK.md` 或 `../setup_env.sh`。

评测口径以同级目录的 [../eval_protocol.md](../eval_protocol.md) 与 [../templates/gate0_baseline_report.md](../templates/gate0_baseline_report.md) 为准。以更新后的官方 24GB 锁定表 [../VERSION_LOCK.md](../VERSION_LOCK.md)（2026-11-26 22:06）为准，旧 48G 记录见 [../VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126](../VERSION_LOCK.md.bak-48g-ubuntu-30041-20261126)。软件钉死项保持不变。新机尚未连接，主机名、驱动快照与安装状态仍【待实测】；报告须抄录官方 24GB 机实际版本与显存 MiB，不要把备份里的 48G 主机 / 显存写成验收事实。本目录 `configs/protocol.yaml` / `protocol.json` 已由工具链改为官方 24GB 显存窗；本文不改这些文件，也不改 `gate0_common.py`。

## 1. 在目标主机上怎么跑

官方验收机必须是 **官方 RTX 4090 24GB（sm_89）**，conda 环境 **`torch2.8_cuda12.8`**。改装 48G 或非 24GB 卡的结果不得写成 V2.8 正式验收。Tesla P4 及其它非目标卡、以及 `torch2.14_cuda13.0` 不得用于官方验收。注意：`gate0_common.py` 的硬检查文案可能仍写旧 48G，本文不改该脚本；不要把改装 48G 或非官方 24GB 卡的结果写成正式验收。

```bash
# 把本目录放到目标主机后（例如 /data/te-faiss-week1/w1_gate0）
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch2.8_cuda12.8     # 非交互 shell 必须先 source conda.sh

cd /path/to/te-faiss-week1/w1_gate0
./run_gate0.sh
```

脚本默认顺序仍是 **BF16** 与 **TE FP8 逐张量 DelayedScaling** 的性能 + 质量。这是脚本默认项，**不得**把旧主机「默认可跑只含 BF16 与 TE FP8」写成新机已装事实；新机安装状态【待实测】，未装项写「未装 / 未测」。常用参数：

```bash
./run_gate0.sh --configs bf16                 # 只跑教师
./run_gate0.sh --configs bf16,te_fp8 --skip-quality
./run_gate0.sh --configs bf16,te_fp8 --skip-perf
./run_gate0.sh --configs bf16,fp16,te_fp8     # fp16 为可选加测，不替代 BF16 教师
```

可选量化配置（AWQ / GPTQ / GGUF / AQLM / 注意力 W4+FFN）在脚本里默认跳过。V2.8 Gate 0 的 8B 矩阵仍要求记录这些对照（有公开权重时含 AQLM）。新机上是否已装以实测为准，未装写「未装 / 未测」，不得沿用旧 48G 锁定表的「本轮默认未装」作为新机状态。未批准前不要加 `--allow-optional`。本脚本**不会安装**这些包，也**禁止**安装 `gptqmodel 7.12.2` 及任何要求 `torch>=2.14` 的版本。

权重只读取 `/data/models/Meta-Llama-3-8B`（官方 Meta-Llama-3-8B HF safetensors，排除 `original/*`）。不会调用 `huggingface-cli download`。评测集若需从 Hub 拉取，走 `HF_ENDPOINT=https://hf-mirror.com`。

输出位置（默认可重入，每次新时间戳，不覆盖旧日志）：

| 位置 | 内容 |
| --- | --- |
| `results/gate0-YYYYMMDD-HHMMSS/` | 本轮 JSON、子进程日志、`manifest.json`、`table_draft.md` |
| `results/latest` | 指向最近一次运行目录的符号链接 |
| `logs/gate0-YYYYMMDD-HHMMSS.log` | 控制台完整日志 |

可用环境变量 `W1_OUTPUT_DIR` 把结果改写到例如 `/data/te-faiss-week1/w1_gate0/results`。把 `table_draft.md` 里**已出现的实测**抄进 `gate0_baseline_report.md`；草稿里没有的格子保持「待实测」。不要把草稿当作已签字的报告。

## 2. 依赖（软件钉死项与锁定表一致）

脚本**不安装**任何包。运行前环境须已按目标机上的安装脚本装好。下表软件钉死项与现有 `VERSION_LOCK.md` 一致，保持不变；**驱动、主机名、显存等硬件行以官方 24GB 新机实测为准**，不得把旧 48G 锁定行抄成验收事实（偏离软件钉死项则应失败退出）：

| 名称 | 锁定版本 |
| --- | --- |
| Python | 3.12.0 |
| PyTorch | 2.8.0+cu128 |
| CUDA 工具包 | 12.8（`/usr/local/cuda-12.8`） |
| 驱动 | 以新主机实测为准（更新后的锁定表不再把旧 48G 的 580.159.03 作为新机硬门槛） |
| nvidia-cudnn-cu12 | 9.28.1.4 |
| Transformer Engine | 2.21.0（TE 路径需要；若尚未装完，TE 配置应失败而不是静默回退成 BF16） |
| huggingface_hub | 2.4.0 |
| transformers | 5.24.1 |
| accelerate | 1.16.0 |
| datasets | 5.2.0 |
| lm-eval | 0.4.13 |
| tokenizers | 0.23.3 |
| safetensors | 0.8.0 |

协议文件同时提供 `configs/*.json` 旁路。目标环境若没有 PyYAML（锁定表未列该包），脚本自动读 JSON，不会自行 `pip install`。

`VERSION_LOCK.md` 第 8、10 节记载的是旧 48G 主机那次安装进度，**不能自动写成新 24GB 机的事实**。新机上 TE 是否已装、能否导入、DelayedScaling 是否真正走 FP8 kernel，均【待实测】；未通过前不得把 TE 路径写成已测。

## 3. 配置矩阵（新机状态以实测为准）

| 配置 | 本机状态（以官方 24GB 机实测为准） |
| --- | --- |
| `bf16` | 可跑 / 未装 / 未测（择一）；质量与性能教师 |
| `te_fp8` | 可跑 / 未装 / 未测（择一）；逐张量 DelayedScaling；必须记录是否真正走到 FP8 kernel |
| `fp16` | 可选加测，须显式 `--configs`；不替代教师 |
| AWQ 4-bit、GPTQ 4-bit（词表 BF16 / 1 字节）、GGUF Q4_K_M 与更低比特代表、AQLM 1×16 与 2×8、注意力 W4 + FFN 原方案 | 可跑 / 未装 / 未测（择一）；未装不得写成已测 |

TE 路径的实现：把 `nn.Linear` 换成 `te.Linear`，并在 `te.autocast(recipe=DelayedScaling(...))` 下前向。这**不等于**官方 tutorial 把整个 `LlamaDecoderLayer` 换成 `TransformerLayer`。注意力分数仍走 Hugging Face 实现。检测逻辑会记录：`is_fp8_available`、替换成功的层数、隔离 `te.Linear` 与一次 decode 的 CUDA kernel 名、amax history 是否非零。没有足够证据时结论为「未证实 / 待实测」，禁止改写成已走 FP8。

## 4. 测量口径（脚本已写死）

- warm-up 3 次后再测；每配置正式至少 3 次；报告中位数和波动（含 P50 / P95）
- prefill 与 decode 分列；decode 先 `batch=1`
- 性能主表已冻结（eval_protocol 第 5.1 节，开发总管 2026-11-26）：`B=1`，prefill / 上下文 `L=2048`，decode 256。prefill 512 / 8192 与短 prompt decode 为 A7 可选补充表，不替代主表，也不另设通过线
- 采样已冻结：`temperature=0`（greedy），关闭 `top_p`；报告须抄录实际值
- 质量：lm-eval 0.4.13 官方任务 `wikitext`、`c4`、`mmlu --num_fewshot 5`；PPL 主指标 `word_perplexity`；不另切未说明的滚动窗口；C4 必须用官方分片，未完成则不得进入门禁
- 质量评测种子已冻结为 `0`（`--seed 0`）。正式比分不得改种子；重复运行也用 0，只观察实现噪声
- 目标任务：待补，本周不能据此判定 Gate 2（C 门）或 S1；Gate 2 硬门在业务方确认前不得关闭
- S2 按 V2.8 原文：8B FFN 查表 > 2 ms/token 且无法缓解。2 ms 不是本卡已测停线，且只针对 gather 主体；通过了也不等于批准内核开工。**禁止**把 H100 的 1.2 ms 写进官方 24GB 停线。同卡 TE FP8 基线仍须实测，供日后 Gate 3 对照，本周只记录
- 失败即退出（编排器会保留已完成配置的日志）；不要在失败后手工改锁定版本

## 5. 缺项

- 目标任务与业务「非劣」口径仍待补
- 同卡 TE FP8 基线：待在官方 24GB 机实测；S2 对照的是 V2.8 原文 2 ms/token（gather 主体），不是把 2 ms 或 H100 的 1.2 ms 写成已测停线，也不得把新机数据回填进 48G 备份或改写当前官方 24GB 锁定表冒充已测
- TE 2.21.0 在官方 24GB 机上是否已装好、能否导入、DelayedScaling 是否真正走 FP8 kernel：待实测
- `lm-eval 0.4.13` 与 `transformers 5.24.1` 的运行时相容性：待实测
- 检查点目录是否已就绪：待实测
- 质量种子、性能主表与吞吐采样已冻结；脚本按冻结值写死，覆盖步数的结果不得作为主表正式比分
- 预填 / 解码逐层余弦一致性：本周脚本未测，报告标明未测；V2.8 初值 ≥ 0.999、待校准，不是已锁定验收线，不得据此关闭 Gate 2
- 官方整层替换 `TE.TransformerLayer` 路径未实现，与当前 Linear 替换路径的差异待另批
- PPL 相对 BF16 增幅 5% 只是原文候选，不得写成已生效的本卡验收线

填写报告请用本目录 [gate0_baseline_report.md](gate0_baseline_report.md)（与 `../templates/gate0_baseline_report.md` 对齐）。
