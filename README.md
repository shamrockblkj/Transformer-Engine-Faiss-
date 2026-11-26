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

白化往返用 v2，检查的是 `pq_layer` 里的 `prepare_work` / `restore_work`。`prepare_work` 和 `restore_work` 都已通过，0.4642 是原始空间的输出误差，而且是样本内、n=32：估计 Σ 和计算误差用的是同一 32 条激活。完整矩阵上的恒等式在这里不成立。现在的 Σ 是 16 个 4×4 块，白化只压输出误差的块内部分，跨块项不受控制。

```bash
mkdir -p /tmp/pq_wx
./pq_layer --dump-wx /tmp/pq_wx
python3 scripts/whiten_roundtrip_check_v2.py --w /tmp/pq_wx/W.bin --x /tmp/pq_wx/X.bin
python3 scripts/pq_faiss_layer.py --diagnose /tmp/pq_wx/X.bin
```

条件数最小 2.165e4、中位 4.231e4、最大 6.564e4，以及「能量在某一个坐标的方差里」，都来自坐标独立的合成自检，不是 Llama 3 8B。ε/λmin ≈ 1e-6 × 条件数 / 4，最大约 1.64%，中位约 1.06%。这批数据上对角白化 0.4630 和块白化 0.4642 几乎相同，只说明两条路径一致。

阳性对照把每个 4 维块先拉成不同方差，再乘该块的随机正交阵。同一套权重、同样 n=32 上，子空间白化的输出 MSE 是 0.5175，对角白化是 1.0450。小行相对误差从 0.2019 升到 0.3267，约 +62%。坐标独立那次相对原样只有 +3.2%。X 仍等真实激活再定。3.3% 和 0.5% 的交叉项差别不是跨块结构。

① 要和坐标独立那组的 0.578 比。全局旋转后是 0.880 和 0.951，旋转作用在 X 上。旧的残差列相关三组都在 0.07 附近，单一方向被其余坐标冲淡。把残差投到 Σ 的最大特征向量上以后，按块标准化的相关仍然只有约 −0.015。按投影二次型算，一个大潜变量是 −35.8%，和交叉项 −35.3% 对齐，这个方向占迹的 99.7%。另外两组主方向的二次型也大约 −30%，但那个方向只占迹的 16% 左右。

交叉项只是辅助指标，可能为负，不能当损失。同一批秩 1 的 X 上，5 个 k-means 种子的交叉项是 −35.3%、−11.1%、−43.4%、−14.7%、−5.6%，负号 5/5。块对角近似在这个对照里靠残差抵消占了便宜。

完整白化对块白化还没有结论。`scripts/pq_retest.py` 是 numpy 种子，和 C++ 的 0.0216 / 0.0230 不是同一套数。两条路径各自在前 192 条上估计 Σ、后 64 条上选 ε；192 条估计 64×64 更噪，选出的 ε 偏大，用回 256 条是偏保守的。网格到 3，四组都不在端点，块白化的曲线也没有降到 3。秩 1 上块白化选中 1e-3，评估集 0.0211±0.0016；完整白化选中 1e-1，评估集 0.0120±0.0018，领先 0.0091，大于两边标准差之和 0.0034。这一档是弱方向被收缩到各向同性。rho 用五个种子的评估贡献平均后加权，中位 1.074，最大 4.226，加权 1.355。4、6、8 个潜变量上留在 ε 之上的特征值有 4 或 5 个。n=32 的自检仍不把那次求逆当成这条的结果。

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
