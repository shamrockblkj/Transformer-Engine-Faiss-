# Llama 3 8B 第 15 层摸底

下一步只做这一件。整模型评测、效率优化、70B、以及 down 的白化，都等这次结果出来再定。到现在为止的结论都来自合成数据。真实的 r 如果有几百，或者低秩白化并不比块白化好，后面的计划都要改。

脚本用分支 `cursor/cpp-gate0-proto-2034` 的 `086daa8` 及之后。这一版已经同时包含：

- 校准倍数的 `>=N (未收敛)`、`converged`、失败原因 `r_mismatch` / `mse_tol`、完整 r 序列、单档时的「仅一档，不判收敛」和「仅一档，不归类」
- `Accum` 每 512 个 token 用 FP32 算 bᵀb，再加进 FP64；`--accum-chunk` 可调
- `collect` 里 `torch.backends.cuda.matmul.allow_tf32 = False`
- rho 的 `--floor-rel`，默认 `1e-5`

当前这台云端代理没有 GPU，也没有 Llama 权重，也没有测量主机的地址、端口和用户名。摸底还不能开始。

跑 `collect` 之前先做两件已有流程里的事，通过后再跑下面的命令：

1. 官方 RTX 4090 24GB 硬件断言。
2. 使用冻结环境。

仓库里没有这两步的命令文本，按测量侧自己的流程执行。然后再确认这台机器能看见 GPU，并且已经放好 Meta-Llama-3-8B 权重。

## 语料

来源按 `contract/pq_ablation.yaml` 的 `llama_retest.corpus`，不是另造的：

- 校准 `cal.jsonl`：C4 **训练集**里互不重叠的文档。
- 评估 `eval.jsonl`：WikiText-2 **训练集**里互不重叠的文档。
- 不用 WikiText-2 的 test，也不用日后要报 C4 困惑度的那一份验证划分。
- 切分以文档 / 序列为单位，不在 token 级别随机切开。每行一个 `{"text": ...}`。

`--mult 1`、序列长 1024 时，校准和评估各约 4096 条 token，大约各要 4 条完整序列。多备几篇，坏序列丢掉之后仍凑满。不要半截序列去补。

主机上若已有这两份原始语料，直接切成 jsonl。若没有，用 Hugging Face 上的官方划分装到本机后再切：C4 训练集、`wikitext` 的 `wikitext-2-raw-v1` 训练集。这是合同点名的语料，不是另拉一份。

## 命令

```bash
python3 scripts/llama3_retest.py collect \
    --model /路径/Meta-Llama-3-8B \
    --cal cal.jsonl --eval eval.jsonl \
    --layers 15 --positions mlp_in --mult 1 \
    --out smoke/

python3 scripts/llama3_retest.py analyze \
    --stats smoke/ --seeds 2 --select-seeds 1
```

`--layers 15` 是 `model.layers` 的下标，也就是第 16 层。请记下 `collect` 和 `analyze` 各自的墙钟时间，后面只优化最慢的那一段。

## 回传

把 `smoke/spike_report.json` 和 analyze 的整段输出发回。先看三样：

1. 尖峰是情况 A 还是 B。没有超阈值尖峰时不判情况。pos 0 占比 ≥ 95% 是 A，否则是 B。
2. 排除首 token 后的有效秩 r。这一轮测的是 `mlp_in`，也就是 4096 维那份 Σ，只属于 gate 和 up。
3. 低秩对块白化的配对 z。

倍数那一行如果是 `1 (仅一档，不判收敛)`，r 序列如果是 `1:r (仅一档，不归类)`，这两行都不拿来下结论。分档判断要等 1、2、4、8、16 五档的完整 r 序列。

## 残差流断言

日志里的报错文字以测量机器上的脚本为准。仓库里这句是「残差流 RMS 和投影 peak 必须同时存在。缺 resid 时不能改用 peak 打分。」另一份脚本写的是「残差流 hook 没有在投影 hook 之前触发：不能用 RMSNorm 之后的 peak 代替打分」。

正常前向时，decoder layer 的 pre-hook 先于内部投影层，所以这条断言不一定是两个 hook 的顺序反了。更常见的是 layer hook 没触发，或者没拿到 `hidden_states`（后者通常先报 `KeyError`），以及 `resid` 和 `peak` 的 token 数不同。带 `new_seq` 的那份会在每条序列开始时把 `resid` 清成 None，采满之后 `new_seq`、`on_resid`、`on_proj` 直接返回。真的报错时，把完整 traceback，以及 transformers 和 torch 的版本一起发回。不能改用 peak 打分。

## 位数怎么读

推理时按 FP16 摊。码固定每个权重 2 比特。一层 gate、up、down 共 1.762 亿个权重，码本再加 0.524 比特，基础开销是 2.524。

块白化的 4×4 和子空间对齐，反白化可以离线乘进码本。推理时不存、不多算，那 0.007 比特只在训练时出现。低秩的 `V_r` 跨子空间，必须在激活端单独乘，位数是实的。两者的差距就是 `V_r`，不会被块白化的 0.007 抵消。

`V_r` 是 4096×r 个 FP16，由 gate 和 up 两张矩阵一起分摊。表里 r=8 的 0.004 是分摊之后的数。如果只摊到其中一张上，这个数要翻倍。

| 矩阵 | 码 + 码本 | 再加的 V_r（gate 与 up 分摊） |
|---|---:|---|
| gate、up（14336 行） | 2.286 | r=8 为 0.004；r=32 为 0.018；r=128 为 0.071；r=256 为 0.143；r=512 为 0.286；r=1024 为 0.572 |
| down（4096 行） | 3.000 | 这一轮不填。down 的 r 要等 `down_in` |

摸底的 r 只代入 gate/up 那一列。down 暂时按块白化，推理时白化是 0 位，单张保持 3.000。down 的码本已经占 1 比特，比这份 `V_r` 更贵。要往大约 2 位靠，先考虑多个子空间共用码本，或减小 k。注意力和词表走 FP8，不在这张表里。

## 这次不做

不跑 16d，不加到 5 个正式种子，不测 `down_in`，不测 70B，不做整模型困惑度。这三样回来之后再排后面的步骤。
