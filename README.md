# Transformer Engine 与 Faiss

方案正文是 [Transformer Engine 与 Faiss 混合压缩及推理优化方案V2.6_修订稿.md](Transformer%20Engine%20与%20Faiss%20混合压缩及推理优化方案V2.6_修订稿.md)。当前批准的是第 1 周的判定实验，不是全量解码内核。

`src/` 里是这两项实验的 C++ 实现：

- `gather_ref` 检查全局查表和 SMEM 切片查表的索引是否一致，并打印 1.2 ms/token 停机线对应的查找率。它不测量 GPU 时间。
- `gather_bench.cu` 在 Hopper 上用 CUDA 事件对比这两条路径，并把 `gate` 的耗时按查找次数外推到整个 FFN。
- `pq_layer` 在一块缩小的单层权重上对比四种离线路径：原样、按行 RMS、按 4 维子空间白化、两者都用。指标是小行相对误差和输出 MSE。

```bash
make test
```

有 NVCC 和 Hopper GPU 时：

```bash
make gpu
./gather_bench
```

全尺寸码本训练仍使用 Faiss 的 `ProductQuantizer`。这里的 k-means 只用来确认行尺度和子空间白化的方向，不代替 Faiss。
