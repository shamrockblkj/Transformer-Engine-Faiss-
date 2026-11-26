# Transformer Engine 与 Faiss

方案正文是 [Transformer Engine 与 Faiss 混合压缩及推理优化方案V2.6_修订稿.md](Transformer%20Engine%20与%20Faiss%20混合压缩及推理优化方案V2.6_修订稿.md)。

第 1 周三件事同时准备，不按一周日历往下排。全量 CUDA 解码内核仍未批准。测量主机是晨涧云 RTX 4090 48G（sm_89）。这台云端机器没有 GPU、没有 nvcc、没有 Llama 3 8B 权重，不能在这里计时，也不能把接口自检写成全尺寸结论。

1.2 ms/token 只是方案里 H100 的原文口径，不是 4090 的停线。S2 阈值留空，要用同卡的 TE FP8 DelayedScaling 重标定。

## 云端可以跑的自检

```bash
make test
python3 -m pip install -r requirements-pq.txt
python3 scripts/pq_faiss_layer.py --self-check
python3 scripts/gate0_host.py --list
```

`make test` 只做查表索引对拍，以及 128×64 的自制 k-means 算法自检。那个自检不是 Faiss，不能当作 S3。口径和复测矩阵在 [contract/pq_ablation.yaml](contract/pq_ablation.yaml)。

`pq_faiss_layer.py --self-check` 只确认 faiss-cpu 1.16.0 的 `train` / `compute_codes` 接口。输出里的误差来自缩小随机矩阵。

## 测量主机

三件事可以同时开，不必串行。

查表微基准。不要在云端编译：

```bash
make gpu
./gather_bench
```

`make gpu` 同时生成 sm_89 和 sm_90。4090 上若共享内存放不下方案正文里的 128 KB 切片，程序会改用放得下的 split，并写明这不是 H100 原文配置。打印出来的外推时间不能用来判断通过或不通过。

单层全尺寸 PQ。权重放在测量主机，脚本不下载第二份：

```bash
python3 -m pip install -r requirements-pq.txt
python3 scripts/pq_faiss_layer.py \
    --weights /路径/Meta-Llama-3-8B \
    --activations-dir /路径/校准激活 \
    --layers 0 \
    --out out/pq_layer0
```

激活文件由测量主机从本地模型导出，脚本不下载权重。默认最多 256 个 token，不是 512×2048 的全量校准：

```bash
python3 scripts/export_layer_activations.py \
    --weights /路径/Meta-Llama-3-8B \
    --prompts /路径/prompt.txt \
    --layers 0 \
    --max-tokens 256 \
    --out /路径/校准激活
```

导出结果是 C 顺序 float32 npy：

- `layer0.gate_up.npy`，形状 `[token 数, 4096]`，进入 MLP 的隐藏状态
- `layer0.down.npy`，形状 `[token 数, 14336]`，`down_proj` 的输入

四组都会写入 `out/pq_layer0/report.txt`：原样、行 RMS、子空间加权、两者都用。缩小自检里「两者都用」的小行相对误差是 0.2260，高于原样 0.1979；全尺寸报告也要四组都留。这一步不能判定 S3。

Gate 0 基线不在云端测：

```bash
python3 scripts/gate0_host.py --check-env
python3 scripts/gate0_host.py --execute --baseline bf16 \
    --model-dir /路径/Meta-Llama-3-8B \
    --prompts /路径/prompt.txt \
    --out out/gate0
```

除 BF16 外要另给 `--quant-dir`。注意力 W4 + FFN 原方案还没有权重，脚本会拒绝测量。没有 GPU 时不会写出吞吐或 PPL。
