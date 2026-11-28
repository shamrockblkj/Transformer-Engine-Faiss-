# W2：64 KB 与 32 KB 对照说明

V2.8 Gate 1 要求对比 L2 直查、SMEM 64 KB 切片、SMEM 32 KB 切片。

## 已具备 32 KB 路径的文件

`w2_gather/lut_gather_bench.cu`（2026-11-26 写入的 V2.8 Gate 1 小基准）已实现：

- `lut_smem<8>`：每块 8 组，64 KB
- `lut_smem<4>`：每块 4 组，32 KB

在官方 24GB 主机上编译运行即可做对照。该文件已加入官方 24GB / sm_89 硬断言（显存按 **MiB**，不用 `totalGlobalMem/1e9`）；断言失败则不得开始计时。

## 已验收微基准 `gather_microbench` 仍只有 64 KB

`src/llama3_8b_shapes.h` / `src/gather_microbench.cu` 的官方 4090 默认瓦片仍是 **64 KiB**：

- gate/up：`SPLIT_K=8`，8 packed groups/block，grid 448，block=256
- down：`SPLIT_K=28`，同样 8 packed groups、grid 448

若还要用这一套二进制做 32 KB 对照，连接后只需小补丁（不要扩成内核开发）：

- 32 KiB = 4 packed groups × 16 × 256 × 2 B
- gate/up：64 packed groups，`SPLIT_K=16`（每块 4 packed），grid = 14336/256 × 16 = 896
- down：224 packed groups，`SPLIT_K=56`，grid = 4096/256 × 56 = 896
- block 仍为 256；`--opt-half2` / `--opt-bank-replicate` 保持关闭

未落地该补丁前，不要把「只有 64 KB」的 `gather_microbench` 结果写成已完成 32 KB 对照。优先使用 `lut_gather_bench.cu` 覆盖 Gate 1 的 64 vs 32 KB 要求。
