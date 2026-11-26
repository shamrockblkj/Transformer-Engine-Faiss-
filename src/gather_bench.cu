// H100 上对比两条查表路径。本文件不参与 CPU 自检。
// 编译：nvcc -O3 -std=c++17 -arch=sm_90 src/gather_bench.cu -o gather_bench

#include "gather_layout.h"

#include <cuda_fp16.h>

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

int main() {
    const GatherShape gate{14336, 1024, 4};
    const int per = groups_per_split(gate);
    const size_t smem = static_cast<size_t>(per) * 16 * 256 * sizeof(__half);
    std::vector<__half> lut(static_cast<size_t>(gate.n_subspaces) * 256);
    std::vector<uint8_t> codes(static_cast<size_t>(gate.n_subspaces / 16) * gate.n_rows * 16);
    for (size_t i = 0; i < lut.size(); ++i) {
        lut[i] = __float2half(static_cast<float>(static_cast<int>(i & 255) - 128) / 32.f);
    }
    for (size_t i = 0; i < codes.size(); ++i) {
        codes[i] = static_cast<uint8_t>(i * 17u);
    }

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
    std::printf("gate 切片 SMEM %.1f KB\n", static_cast<double>(smem) / 1024.0);
    std::printf("L2 直查 %.3f ms/次，外推到全部 FFN %.3f ms/token\n", l2_ms, l2_ms * scale);
    std::printf("SMEM 切片 %.3f ms/次，外推到全部 FFN %.3f ms/token\n", smem_ms, smem_ms * scale);
    std::printf("停机线 %.1f ms/token。外推只按查找次数缩放，不含建表、残差和注意力。\n", kGatherStopMs);

    cudaFree(d_lut);
    cudaFree(d_codes);
    cudaFree(d_partial);
    return 0;
}
