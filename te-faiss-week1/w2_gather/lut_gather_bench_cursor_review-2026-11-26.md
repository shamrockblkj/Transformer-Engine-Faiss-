# lut_gather_bench.cu 独立评估意见（交开发总管汇总）

- 评估来源：Cursor 云代理 [Review lut_gather_bench.cu](https://cursor.com/agents/bc-8e6d9726-b143-5322-830b-7b582b651e0b)（bc-8e6d9726-b143-5322-830b-7b582b651e0b），代码审查员已对照本地文件 `/workspace/te-faiss-week1/w2_gather/lut_gather_bench.cu` 抽查关键结论。
- 范围：V2.8 Gate 1 实验①查表微基准，不是完整解码内核。未开拉取请求，未另开第二个评估代理。
- 仓库主干（https://github.com/shamrockblkj/Transformer-Engine-Faiss- ，当时仍为 fb1bdb2）只有 `README.md` 和 V2.6 方案正文；评估对象是上传的源文件。
- 本次没有上机，没有任何实测时间。下列带宽下限与查找次数是规格算术，不得写成已测结果。

## 总论

合法 tile 上，GEMV 主体的索引、Split-K 覆盖、SMEM 协作装载和 `__syncthreads` 之后的提前退出是对的。它还不能当作官方 RTX 4090 24GB 的验收结果：预检挡不住 48GB 改卡，编译目标只写在注释里，`lut_smem<4>` 和 `codes_only` 不是同一张网格，8B 停机线从仓库 V2.6 的 1.2 ms 改成了 2 ms，而且「未触发」只看三条查表路径里最快的那条。

## 发现（按严重程度）

### 高：没有官方 4090 24GB 预检，编译目标也没有落地

`main` 只打印 `cudaGetDeviceProperties`，显存用 `totalGlobalMem / 1e9`（十进制 GB）。SMEM 不足 64 KiB 时只警告后继续。计算能力、显存窗口、SM 数、设备名都没有拒绝路径，进程最后总是 `return 0`。

因此同一份二进制可以在这些设备上跑完并打印「未触发」：非 sm_89 的卡、官方 RTX 4090 D（仍是 24GB，但 SM 是 114 不是 128）、魔改 48GB 4090，以及同样是 sm_89 的 48GB Ada 专业卡。

库存 4090 的 `totalGlobalMem` 低于铭牌 24 GiB。云代理引用的公开 `deviceQuery` 样本是 24217 MiB（约 23.65 GiB）；按 `/1e9` 会显示成约 25.4 GB。写成「必须等于 25769803776」会把真卡拒掉；只看「大约 24」又会把 48GB 改卡放进来。

文件头的 `nvcc -O3 -std=c++17 -arch=sm_89` 只是注释，没有构建规则，也没有 `CUDART_VERSION == 12080` 或内核里的 `__CUDA_ARCH__ == 890`。

### 高：8B 停机线的判定比打印文案宽

只对名称恰好是 `"Llama 3 8B"` 的项生效，算式为 `32 * (2 * t_gate + t_down)`。`best` 是 L2 直查、SMEM 64 KiB、SMEM 32 KiB 三者的最小值，不包括 `codes_only`。

这个数只覆盖查表 GEMV 主体（读码、查表、写出本 split 的 FP32 部分和），不包含建表、FWHT、残差 GEMV、设备端 Split-K 归约与 SiLU/GeGLU epilogue、注意力 / KV / 输出头，以及真实逐层 launch 空隙。文件头和文末已写明范围。

查找次数与 V2.6 的 1.41×10⁹ 一致，这是结构恒等式，不是计时：每层码字 44,040,192，32 层共 1,409,286,144 次查找（也是同样字节的码字）。按 4090 公开峰值带宽 1008 GB/s，这些码字从 HBM 读一遍的规格下限约为 1.398 ms/token，不是本次测量。

仓库 V2.6 停机线是 1.2 ms/token，低于上述 1.398 ms 下限，旧门槛在码字真从 HBM 读一遍时不可能通过。文件改成 2.0 ms 后约是该下限的 1.43 倍，正文没有写换算和改线理由。「且无法缓解」没有被执行：超过 2 ms 只多打印「需分析」，不退出、不改布局、不重跑。L2 直查单独跑进 2 ms 时，SMEM 路径即使更慢也会显示「未触发」。

### 中：`lut_smem<4>` 和 `codes_only` 不是同一场实验

A/B/C 共用 `groups/8` 网格；D 的 launch 用 `split4 = 2 * split8`，网格和 Split-K 都是 A/B/C 的两倍。8B 上 A/B/C 网格 448，D 为 896。D 可以和 A/B/C 比「哪种切分更快」，但不能把 D 除以只在 8 组网格上跑过的 `codes_only`，解释成「查表已被同一条读码流水掩盖」。

占用率一列张冠李戴：`codes_only` 打印的是 `lut_l2` 的 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`，没有查询 `codes_only` 自己。

停机线用的是 8B 这张较小网格（每 SM 约 3.5 个 block），更大模型是另一种饱和程度。

### 中：对拍只覆盖每个 split 的第 0 个 row tile

CPU 参考是 FP32 累加，合法形状下分组覆盖无重叠、无遗漏，码字布局与主机下标一致。核对范围是每个 split 的前 256 行（`check_rows = min(256, N)`）。四个模型的 N 都远大于 256，后续 row tile 出错这次对拍发现不了。`codes_only` 不对拍，与其角色相符。相对误差门限 `1e-3` 对这条路径偏松。

N 为 256 的倍数时，提前退出写在协作装载和 `__syncthreads` 之后，合法 launch 下没有死锁或越界。

### 低：计时方法基本合格，顺序和错误检查仍有缺口

固定种子全范围 `uint8`、多副本冲 L2、预热 10 次再用 CUDA event 包默认 200 次，和文件声明的测量对象一致。缺口：四个内核顺序固定，`smem<4>` 总在更热时跑；`cudaGetLastError` 只在预热后、计时循环前检查，计时循环之后没有再查；被跳过的形状 `Timing` 为零，`best / t_codes` 会变成 NaN，且 `best == 0` 会打印「未触发」（当前四个模型不会被跳过）；打印的「码字有效带宽」分母只有码字字节。

## 形状和网格

用户点名的 8B 合法 tile（block=256，gate/up Split-K=8 网格=448，down Split-K=28 网格=448）对应 A/B/C 这一档；D 是另一档。四个模型的 `M/16` 都能被 8 整除，`N` 都能被 256 整除。公开配置对得上 Llama 3 8B、Qwen2.5-32B、Gemma 2 27B、Mistral Small 24B（不是旧的 22B / 56 层写法）。

相对仓库 V2.6 的 L2 直查草图：本文件把退出放到屏障之后，并用编译期 `code_at` 取字节，避免打包码落到 local memory；SMEM 从按 H100 写的 128/112 KiB 改成 sm_89 放得进的 64/32 KiB。仓库里没有另一份 W2 gather 实现可再比。

## 站得住的部分

- 码字布局 `[M/16, N, 16]`、`uint4` 向量加载、小端字节抽取，与 CPU 下标一致。
- 8 组和 4 组都把全部子空间各算一次；部分和缓冲按更大的 `groups/4` 分配，两种 Split-K 都写在界内。
- SMEM 装载在 `__syncthreads` 之前由全体线程参加，提前退出在屏障之后；64 KiB 有 `cudaFuncSetAttribute` opt-in。
- 随机 `uint8`、LUT 复用、多副本冲 L2、event 计时，与声明的测量对象一致。
- 四个模型的 8 组网格与 block=256 的 V2.8 合法 tile 一致。

## 上机前建议（仍只改这个微基准，不扩展成解码内核）

1. 编译钉死 CUDA 12.8 与 sm_89：`CUDART_VERSION` 必须落在 12080，内核里 `__CUDA_ARCH__ == 890`。命令保持 `nvcc -O3 -std=c++17 -arch=sm_89`。
2. 运行时同时满足才继续，否则非零退出：`major == 8 && minor == 9`；`multiProcessorCount == 128`；名称是 `NVIDIA GeForce RTX 4090`；`totalGlobalMem` 落在约 23000–25000 MiB。大于约 40 GiB 的直接拒绝，用来挡住 48GB 改卡。打印用 MiB，不要用 `/1e9`。非官方 24GB 卡的结果不得写成 V2.8 官方验收。
3. `codes_only` 的占用率改查它自己。再加一次 `groups_per_split = 4`、网格 `n_blocks * split4` 的 `codes_only`，只拿它和 `lut_smem<4>` 比。文件头把 D 写成另一组网格。
4. 对拍加上每个 split 的最后一个 row tile（行 `N-256 .. N-1`）。`N % 256 != 0` 继续跳过。
5. 计时循环后检查 `cudaGetLastError`。四个内核轮换顺序，或先锁住显存频率和 SM 频率。
6. 停机线同时打印三条查表路径，不要只报 `min`。写明 2.0 ms 相对 V2.6 的 1.2 ms，以及相对约 1.398 ms 峰值读码下限的位置。超过 2 ms 时以非零状态退出。跳过的形状不要参与 `best / t_codes`。通过 8B 停机线也不等于批准内核开工。

完整 decode、FWHT、残差和 epilogue 仍在这个基准外面。内核开发尚未批准。
