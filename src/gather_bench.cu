// 查表微基准：随机 uint8 码 + 随机 FP16 LUT，对比 L2 直查和 SMEM 切片。
// 不是融合 epilogue，不是残差微内核，也不使用 CUDA Graph。
// 测量主机编译（云端没有 nvcc，不要在云端编译）：
//   make gpu
// 默认同时生成 sm_89（RTX 4090）和 sm_90（H100）。
// 1.2 ms/token 是方案正文里 H100 的原文口径，不是本机阈值。
// S2 阈值留空，必须用同卡的 TE FP8 DelayedScaling 重标定。

#include "gather_layout.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void gather_l2(const __half* __restrict__ lut, const uint8_t* __restrict__ codes, float* __restrict__ partial,
                          int n_rows, int groups_per_split) {
    const int split = blockIdx.y;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    const int g0 = split * groups_per_split;
    float acc = 0.f;
    for (int g = 0; g < groups_per_split; ++g) {
        const uint8_t* packed = codes + code_offset(g0 + g, row, n_rows);
        const int m_base = (g0 + g) * 16;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc += __half2float(lut[lut_offset(m_base + k, packed[k])]);
        }
    }
    partial[(static_cast<size_t>(split) * n_rows) + row] = acc;
}

__global__ void gather_smem(const __half* __restrict__ lut, const uint8_t* __restrict__ codes, float* __restrict__ partial,
                            int n_rows, int groups_per_split) {
    extern __shared__ __half tile[];
    const int split = blockIdx.y;
    const int g0 = split * groups_per_split;
    const int tile_subspaces = groups_per_split * 16;
    const int tile_vals = tile_subspaces * 256;
    for (int i = threadIdx.x; i < tile_vals; i += blockDim.x) {
        const int local = i >> 8;
        const int code = i & 255;
        tile[i] = lut[lut_offset(g0 * 16 + local, code)];
    }
    __syncthreads();

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }
    float acc = 0.f;
    for (int g = 0; g < groups_per_split; ++g) {
        const uint8_t* packed = codes + code_offset(g0 + g, row, n_rows);
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            const int local = g * 16 + k;
            acc += __half2float(tile[local * 256 + packed[k]]);
        }
    }
    partial[(static_cast<size_t>(split) * n_rows) + row] = acc;
}

static void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

static float time_kernel(void (*launch)(const __half*, const uint8_t*, float*, int, int, int, size_t), const __half* lut,
                         const uint8_t* codes, float* partial, const GatherShape& shape, size_t smem, int repeats) {
    launch(lut, codes, partial, shape.n_rows, groups_per_split(shape), shape.split_k, smem);
    check(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");
    check(cudaEventRecord(start), "record");
    for (int i = 0; i < repeats; ++i) {
        launch(lut, codes, partial, shape.n_rows, groups_per_split(shape), shape.split_k, smem);
    }
    check(cudaEventRecord(stop), "record");
    check(cudaEventSynchronize(stop), "sync");
    float ms = 0.f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / static_cast<float>(repeats);
}

static void launch_l2(const __half* lut, const uint8_t* codes, float* partial, int n_rows, int per, int split_k, size_t) {
    const dim3 block(128);
    const dim3 grid((n_rows + block.x - 1) / block.x, split_k);
    gather_l2<<<grid, block>>>(lut, codes, partial, n_rows, per);
}

static void launch_smem(const __half* lut, const uint8_t* codes, float* partial, int n_rows, int per, int split_k,
                        size_t smem) {
    const dim3 block(128);
    const dim3 grid((n_rows + block.x - 1) / block.x, split_k);
    gather_smem<<<grid, block, smem>>>(lut, codes, partial, n_rows, per);
}

static std::uint32_t mix_bits(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// 方案正文里 H100 的 gate 切片是 split_k = 4，约 128 KB。Ada 的共享内存上限约 100 KB，
// 128 KB 的切片在 RTX 4090 上启动会失败。这里从 4 开始选本卡放得下的 split，不改内核。
static int choose_split(int groups, size_t max_smem) {
    const int candidates[] = {4, 8, 16, 32, 64};
    for (int split : candidates) {
        if (split <= 0 || groups % split != 0) {
            continue;
        }
        const size_t bytes = static_cast<size_t>(groups / split) * 16 * 256 * sizeof(__half);
        if (bytes <= max_smem) {
            return split;
        }
    }
    return groups;
}

int main() {
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "device prop");
    int max_smem = 0;
    check(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device), "smem cap");
    if (max_smem <= 0) {
        std::fprintf(stderr, "读不到可用的共享内存上限\n");
        return 1;
    }
    const int groups = 1024 / 16;
    const int split = choose_split(groups, static_cast<size_t>(max_smem));
    const GatherShape gate{14336, 1024, split};
    const int per = groups_per_split(gate);
    const size_t smem = static_cast<size_t>(per) * 16 * 256 * sizeof(__half);
    check(cudaFuncSetAttribute(gather_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem)),
          "smem opt-in");

    std::printf("设备 %s，计算能力 %d.%d，可选共享内存 %d KB。\n", prop.name, prop.major, prop.minor,
                max_smem / 1024);
    std::printf("1.2 ms/token 是方案正文里 H100 的原文口径，不是本机阈值。\n");
    std::printf("S2 阈值留空。必须用同卡的 TE FP8 DelayedScaling 重标定后，才能判断这张卡是否通过。\n");
    std::printf("禁止用 1.2 ms 判断 RTX 4090（sm_89）是否通过。\n");
    std::printf("方案正文的 H100 gate 切片是 split_k=4、约 128 KB。本卡实际 split_k=%d，切片 %.1f KB。\n", split,
                static_cast<double>(smem) / 1024.0);
    std::printf("只比较随机码和随机 LUT 的两条查表路径。不含融合 epilogue、残差和 CUDA Graph。\n");

    constexpr std::uint32_t kSeed = 20261126u;
    std::vector<__half> lut(static_cast<size_t>(gate.n_subspaces) * 256);
    std::vector<uint8_t> codes(static_cast<size_t>(gate.n_subspaces / 16) * gate.n_rows * 16);
    std::uint32_t state = kSeed;
    for (size_t i = 0; i < lut.size(); ++i) {
        state = mix_bits(state + 0x9e3779b9u);
        const float value = static_cast<float>(static_cast<int>(state & 1023u) - 512) / 64.f;
        lut[i] = __float2half(value);
    }
    for (size_t i = 0; i < codes.size(); ++i) {
        state = mix_bits(state + 0x9e3779b9u);
        codes[i] = static_cast<uint8_t>(state & 255u);
    }
    std::printf("随机码和随机 LUT 的种子是 %u。\n", kSeed);

    __half* d_lut = nullptr;
    uint8_t* d_codes = nullptr;
    float* d_partial = nullptr;
    check(cudaMalloc(&d_lut, lut.size() * sizeof(__half)), "malloc lut");
    check(cudaMalloc(&d_codes, codes.size()), "malloc codes");
    check(cudaMalloc(&d_partial, static_cast<size_t>(gate.split_k) * gate.n_rows * sizeof(float)), "malloc partial");
    check(cudaMemcpy(d_lut, lut.data(), lut.size() * sizeof(__half), cudaMemcpyHostToDevice), "copy lut");
    check(cudaMemcpy(d_codes, codes.data(), codes.size(), cudaMemcpyHostToDevice), "copy codes");

    const float l2_ms = time_kernel(launch_l2, d_lut, d_codes, d_partial, gate, 0, 20);
    const float smem_ms = time_kernel(launch_smem, d_lut, d_codes, d_partial, gate, smem, 20);
    const double gate_lookups = static_cast<double>(gate.n_rows) * gate.n_subspaces;
    const double scale = static_cast<double>(kFfnLookups) / gate_lookups;
    std::printf("L2 直查 %.3f ms/次，按查找次数外推到全部 FFN %.3f ms/token\n", l2_ms, l2_ms * scale);
    std::printf("SMEM 切片 %.3f ms/次，按查找次数外推到全部 FFN %.3f ms/token\n", smem_ms, smem_ms * scale);
    std::printf("外推只按查找次数缩放，不含建表、残差和注意力，也不能用来判断通过或不通过。\n");
    std::printf("方案里 H100 原文口径是 %.1f ms/token。S2 阈值留空，这个数字不是本机停线。\n", kGatherStopMs);

    cudaFree(d_lut);
    cudaFree(d_codes);
    cudaFree(d_partial);
    return 0;
}
