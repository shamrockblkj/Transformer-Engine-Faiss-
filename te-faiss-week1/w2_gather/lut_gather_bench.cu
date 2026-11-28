// =============================================================================
// lut_gather_bench.cu —— PQ 查表 GEMV 速度小基准（V2.8 Gate 1 实验 ①）
//
// 目的：在 RTX 4090（sm_89）上实测"每 token 约 1.41e9 次 LUT 查找"是否会成为瓶颈。
// 对比三条路径（同一网格、同一 Split-K、同一码字布局 [M/16, N, 16]）：
//   A. codes_only   —— 只读码字、不查表：读码字的带宽下限（对照）
//   B. lut_l2       —— LUT 放在全局内存，查表走 L1/L2（V2.4 原写法）
//   C. lut_smem<8>  —— 每块把 8 组 LUT 切片（64 KB）装进 SMEM 后查表（V2.8 默认）
//   D. lut_smem<4>  —— 每块 4 组（32 KB），每个 SM 可驻留更多 block
//
// 编译： nvcc -O3 -std=c++17 -arch=sm_89 lut_gather_bench.cu -o lut_gather_bench
// 运行： ./lut_gather_bench            （默认 200 次计时）
//        ./lut_gather_bench 50         （自定义计时次数）
// 看 bank conflict / L2 / DRAM 流量（Nsight Compute）：
//   ncu --kernel-name regex:lut_ --launch-count 8 \
//       --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
//                 lts__t_sectors_srcunit_tex_op_read.sum,dram__bytes_read.sum \
//       ./lut_gather_bench 5
//
// 说明：
// * 码字是随机 uint8（最坏情况：查表地址完全随机）。真实码字分布不均匀，冲突可能略少。
// * 码字会准备多份副本、每次调用轮换，总量 > 256 MB，避免码字常驻 72 MB L2 而虚高带宽。
// * LUT 每次复用同一份（真实场景中 LUT 很小、是热数据）。
// * 本基准只测 GEMV 主体，不含建表、FWHT、残差、Epilogue；这些在后续单层 profile 中测。
// =============================================================================
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <cctype>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  std::printf("CUDA error %s at %s:%d -> %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
  std::exit(1); } } while (0)

constexpr int BLOCK = 256;          // 每 block 8 warp
constexpr int SUB_PER_GROUP = 16;   // 一个 uint4 = 16 个 uint8 码 = 16 个子空间
constexpr int CENTROIDS = 256;

__device__ __forceinline__ uint32_t code_at(const uint4& p, int k) {
  // k 为编译期常量（循环已展开），避免把 uint4 落到 local memory
  const uint32_t w = (k < 4) ? p.x : (k < 8) ? p.y : (k < 12) ? p.z : p.w;
  return (w >> ((k & 3) * 8)) & 0xFFu;
}

// ---------------- A. 只读码字（带宽下限） ----------------
__global__ void codes_only(const uint4* __restrict__ codes, float* __restrict__ partial,
                           int N, int groups_per_split) {
  const int n_blocks = (N + blockDim.x - 1) / blockDim.x;
  const int split = blockIdx.x / n_blocks;
  const int row = (blockIdx.x % n_blocks) * blockDim.x + threadIdx.x;
  if (row >= N) return;
  const int g0 = split * groups_per_split;
  uint32_t x = 0;
  for (int g = g0; g < g0 + groups_per_split; ++g) {
    const uint4 p = codes[static_cast<size_t>(g) * N + row];
    x += p.x ^ p.y ^ p.z ^ p.w;     // 防止编译器消除读
  }
  partial[static_cast<size_t>(split) * N + row] = static_cast<float>(x & 0xFF);
}

// ---------------- B. LUT 在全局内存（L1/L2 直查） ----------------
__global__ void lut_l2(const __half* __restrict__ lut, const uint4* __restrict__ codes,
                       float* __restrict__ partial, int N, int groups_per_split) {
  const int n_blocks = (N + blockDim.x - 1) / blockDim.x;
  const int split = blockIdx.x / n_blocks;
  const int row = (blockIdx.x % n_blocks) * blockDim.x + threadIdx.x;
  if (row >= N) return;
  const int g0 = split * groups_per_split;
  float acc = 0.f;
  for (int g = g0; g < g0 + groups_per_split; ++g) {
    const uint4 p = codes[static_cast<size_t>(g) * N + row];
    const __half* base = lut + static_cast<size_t>(g) * SUB_PER_GROUP * CENTROIDS;
    #pragma unroll
    for (int k = 0; k < SUB_PER_GROUP; ++k)
      acc += __half2float(base[k * CENTROIDS + code_at(p, k)]);
  }
  partial[static_cast<size_t>(split) * N + row] = acc;
}

// ---------------- C/D. LUT 切片装进 SMEM ----------------
template <int GPB>
__global__ void lut_smem(const __half* __restrict__ lut, const uint4* __restrict__ codes,
                         float* __restrict__ partial, int N) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  const __half* s_lut = reinterpret_cast<const __half*>(smem_raw);

  const int n_blocks = (N + blockDim.x - 1) / blockDim.x;
  const int split = blockIdx.x / n_blocks;
  const int row = (blockIdx.x % n_blocks) * blockDim.x + threadIdx.x;
  const int g0 = split * GPB;

  // 协作装载：GPB*16*256 个 half，按 uint4（8 个 half）向量化
  constexpr int NV = GPB * SUB_PER_GROUP * CENTROIDS / 8;
  const uint4* src = reinterpret_cast<const uint4*>(lut + static_cast<size_t>(g0) * SUB_PER_GROUP * CENTROIDS);
  uint4* dst = reinterpret_cast<uint4*>(smem_raw);
  for (int i = threadIdx.x; i < NV; i += blockDim.x) dst[i] = src[i];
  __syncthreads();
  if (row >= N) return;

  float acc = 0.f;
  #pragma unroll 2
  for (int g = 0; g < GPB; ++g) {
    const uint4 p = codes[static_cast<size_t>(g0 + g) * N + row];
    const __half* base = s_lut + g * SUB_PER_GROUP * CENTROIDS;
    #pragma unroll
    for (int k = 0; k < SUB_PER_GROUP; ++k)
      acc += __half2float(base[k * CENTROIDS + code_at(p, k)]);
  }
  partial[static_cast<size_t>(split) * N + row] = acc;
}

// ---------------- 测试用例 ----------------
struct MatCase { const char* name; int M; int N; };   // M = 子空间数（= K/4），N = 输出行数
struct ModelCase { const char* name; int layers; MatCase gate; MatCase down; };

struct Timing { double codes_only, l2, smem8, smem4; };

static float max_rel_err(const std::vector<float>& ref, const std::vector<float>& got) {
  float e = 0.f;
  for (size_t i = 0; i < ref.size(); ++i)
    e = std::max(e, std::fabs(ref[i] - got[i]) / std::max(1e-3f, std::fabs(ref[i])));
  return e;
}

static Timing run_case(const MatCase& c, int reps, size_t l2_bytes) {
  const int groups = c.M / SUB_PER_GROUP;
  if (c.M % SUB_PER_GROUP || groups % 8 || c.N % BLOCK) {
    std::printf("  [跳过] %s：M/16 需被 8 整除、N 需被 %d 整除\n", c.name, BLOCK); return {};
  }
  const size_t code_bytes = static_cast<size_t>(c.M) * c.N;     // 每子空间每行 1 字节
  const size_t lut_elems  = static_cast<size_t>(c.M) * CENTROIDS;

  // 多份码字副本，总量 > max(256 MB, 4×L2)，轮换使用以冲掉 L2
  const size_t target = std::max<size_t>(256ull << 20, 4 * l2_bytes);
  const int copies = static_cast<int>(std::min<size_t>(64, (target + code_bytes - 1) / code_bytes));

  // 主机数据
  std::mt19937 rng(1234);
  std::vector<uint8_t> h_codes(code_bytes);
  for (auto& b : h_codes) b = static_cast<uint8_t>(rng() & 0xFF);
  std::vector<__half> h_lut(lut_elems);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);
  for (auto& v : h_lut) v = __float2half(uni(rng));

  uint8_t* d_codes = nullptr; __half* d_lut = nullptr; float* d_part = nullptr;
  CK(cudaMalloc(&d_codes, code_bytes * copies));
  for (int i = 0; i < copies; ++i)
    CK(cudaMemcpy(d_codes + code_bytes * i, h_codes.data(), code_bytes, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&d_lut, lut_elems * sizeof(__half)));
  CK(cudaMemcpy(d_lut, h_lut.data(), lut_elems * sizeof(__half), cudaMemcpyHostToDevice));

  const int split8 = groups / 8, split4 = groups / 4;
  CK(cudaMalloc(&d_part, sizeof(float) * static_cast<size_t>(split4) * c.N));

  const int n_blocks = c.N / BLOCK;
  const size_t smem8 = 8 * SUB_PER_GROUP * CENTROIDS * sizeof(__half);  // 64 KB
  const size_t smem4 = 4 * SUB_PER_GROUP * CENTROIDS * sizeof(__half);  // 32 KB
  CK(cudaFuncSetAttribute(lut_smem<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem8));
  CK(cudaFuncSetAttribute(lut_smem<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem4));

  auto codes_ptr = [&](int i) { return reinterpret_cast<const uint4*>(d_codes + code_bytes * (i % copies)); };

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  auto bench = [&](auto launch) -> double {
    for (int i = 0; i < 10; ++i) launch(i);                     // warm-up
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; ++i) launch(i);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, e0, e1));
    return static_cast<double>(ms) / reps;
  };

  Timing t{};
  t.codes_only = bench([&](int i) { codes_only<<<n_blocks * split8, BLOCK>>>(codes_ptr(i), d_part, c.N, 8); });
  t.l2    = bench([&](int i) { lut_l2<<<n_blocks * split8, BLOCK>>>(d_lut, codes_ptr(i), d_part, c.N, 8); });
  t.smem8 = bench([&](int i) { lut_smem<8><<<n_blocks * split8, BLOCK, smem8>>>(d_lut, codes_ptr(i), d_part, c.N); });
  t.smem4 = bench([&](int i) { lut_smem<4><<<n_blocks * split4, BLOCK, smem4>>>(d_lut, codes_ptr(i), d_part, c.N); });

  // ---- 正确性：前 256 行，与 CPU FP32 参考对比 ----
  const int check_rows = std::min(256, c.N);
  std::vector<float> ref(check_rows, 0.f);
  for (int r = 0; r < check_rows; ++r) {
    float acc = 0.f;
    for (int g = 0; g < groups; ++g)
      for (int k = 0; k < SUB_PER_GROUP; ++k) {
        const uint8_t code = h_codes[(static_cast<size_t>(g) * c.N + r) * SUB_PER_GROUP + k];
        acc += __half2float(h_lut[static_cast<size_t>(g * SUB_PER_GROUP + k) * CENTROIDS + code]);
      }
    ref[r] = acc;
  }
  auto reduce_check = [&](int split, const char* tag) {
    std::vector<float> part(static_cast<size_t>(split) * c.N);
    CK(cudaMemcpy(part.data(), d_part, part.size() * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> got(check_rows, 0.f);
    for (int s = 0; s < split; ++s)
      for (int r = 0; r < check_rows; ++r) got[r] += part[static_cast<size_t>(s) * c.N + r];
    const float e = max_rel_err(ref, got);
    std::printf("    校验 %-10s 最大相对误差 %.2e %s\n", tag, e, e < 1e-3f ? "OK" : "<< 失败");
  };
  lut_l2<<<n_blocks * split8, BLOCK>>>(d_lut, codes_ptr(0), d_part, c.N, 8);
  CK(cudaDeviceSynchronize()); reduce_check(split8, "lut_l2");
  lut_smem<8><<<n_blocks * split8, BLOCK, smem8>>>(d_lut, codes_ptr(0), d_part, c.N);
  CK(cudaDeviceSynchronize()); reduce_check(split8, "smem<8>");
  lut_smem<4><<<n_blocks * split4, BLOCK, smem4>>>(d_lut, codes_ptr(0), d_part, c.N);
  CK(cudaDeviceSynchronize()); reduce_check(split4, "smem<4>");

  // ---- 占用率 ----
  int occ8 = 0, occ4 = 0, occl2 = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ8, lut_smem<8>, BLOCK, smem8));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ4, lut_smem<4>, BLOCK, smem4));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occl2, lut_l2, BLOCK, 0));

  // ---- 输出 ----
  const double lookups = static_cast<double>(c.M) * c.N;
  auto line = [&](const char* tag, double ms, int occ, int grid) {
    std::printf("    %-11s %8.4f ms  %7.1f G查找/s  码字有效带宽 %6.1f GB/s  网格 %5d  每SM驻留 %d 块\n",
                tag, ms, lookups / (ms * 1e-3) / 1e9, code_bytes / (ms * 1e-3) / 1e9, grid, occ);
  };
  std::printf("  %s  M=%d N=%d  码字 %.1f MB × %d 份  LUT %.2f MB\n",
              c.name, c.M, c.N, code_bytes / 1e6, copies, lut_elems * 2 / 1e6);
  line("codes_only", t.codes_only, occl2, n_blocks * split8);
  line("lut_l2",     t.l2,         occl2, n_blocks * split8);
  line("lut_smem<8>", t.smem8,     occ8,  n_blocks * split8);
  line("lut_smem<4>", t.smem4,     occ4,  n_blocks * split4);

  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  CK(cudaFree(d_codes)); CK(cudaFree(d_lut)); CK(cudaFree(d_part));
  return t;
}

int main(int argc, char** argv) {
  const int reps = (argc > 1) ? std::max(1, std::atoi(argv[1])) : 200;

  int ndev = 0;
  CK(cudaGetDeviceCount(&ndev));
  if (ndev != 1) {
    std::printf("FATAL: V2.8 official acceptance requires a single GPU; cudaGetDeviceCount=%d. "
                "Software memory caps are not official 24GB acceptance. Refusing to start benches.\n",
                ndev);
    return 2;
  }

  cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, 0));
  // Official 24GB check uses integer MiB (1024-based). Do NOT use totalGlobalMem/1e9.
  const unsigned long long mem_mib =
      static_cast<unsigned long long>(prop.totalGlobalMem / (1024ull * 1024ull));
  std::printf("设备: %s  sm_%d%d  SM=%d  L2=%.0f MB  每block最大SMEM(opt-in)=%zu KB  显存=%llu MiB\n",
              prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize / 1048576.0,
              prop.sharedMemPerBlockOptin / 1024, mem_mib);

  const std::string gpu_name(prop.name);
  const bool sm89 = (prop.major == 8 && prop.minor == 9);
  const bool official_24g = (mem_mib >= 24000ull && mem_mib <= 25000ull);
  const bool class_48g_or_larger = (mem_mib >= 40000ull);
  std::string name_u = gpu_name;
  for (char& ch : name_u) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  const bool looks_4090d = name_u.find("4090D") != std::string::npos ||
                           name_u.find("4090 D") != std::string::npos;
  const bool looks_rtx_4090 = name_u.find("RTX") != std::string::npos &&
                              name_u.find("4090") != std::string::npos && !looks_4090d;
  if (!sm89 || !looks_rtx_4090 || class_48g_or_larger || !official_24g) {
    std::printf("FATAL: not an official GeForce RTX 4090 24GB (need name with RTX 4090, sm_89, "
                "memory 24000-25000 MiB, typical ~24564 MiB from nvidia-smi / device props).\n");
    std::printf("FATAL: got name=\"%s\"  sm_%d%d  memory=%llu MiB.\n",
                prop.name, prop.major, prop.minor, mem_mib);
    if (class_48g_or_larger) {
      std::printf("FATAL: memory %llu MiB is the >=40000 MiB class (modified 48G historical record "
                  "was ~49140 MiB). Results are not V2.8 official 24GB acceptance.\n",
                  mem_mib);
    }
    std::printf("FATAL: software memory caps are not official 24GB acceptance. Refusing to start benches.\n");
    return 2;
  }
  if (prop.sharedMemPerBlockOptin < 64 * 1024)
    std::printf("警告：每 block 可用 SMEM < 64 KB，lut_smem<8> 将无法启动\n");
  std::printf("计时次数: %d\n\n", reps);

  // gate 与 up 形状相同，每层算 2 次 gate + 1 次 down
  const std::vector<ModelCase> models = {
    {"Llama 3 8B",        32, {"gate/up", 1024, 14336}, {"down", 3584, 4096}},
    {"Mistral Small 24B", 40, {"gate/up", 1280, 32768}, {"down", 8192, 5120}},
    {"Gemma 2 27B",       46, {"gate/up", 1152, 36864}, {"down", 9216, 4608}},
    {"Qwen2.5-32B",       64, {"gate/up", 1280, 27648}, {"down", 6912, 5120}},
  };

  std::printf("================ 每 token 外推（仅 FFN 查表 GEMV 主体，B=1） ================\n");
  for (const auto& m : models) {
    std::printf("\n[%s]  %d 层\n", m.name, m.layers);
    const Timing g = run_case(m.gate, reps, prop.l2CacheSize);
    const Timing d = run_case(m.down, reps, prop.l2CacheSize);
    auto tok = [&](double tg, double td) { return m.layers * (2 * tg + td); };
    const double t_codes = tok(g.codes_only, d.codes_only);
    const double t_l2 = tok(g.l2, d.l2), t_s8 = tok(g.smem8, d.smem8), t_s4 = tok(g.smem4, d.smem4);
    const double best = std::min({t_l2, t_s8, t_s4});
    std::printf("  >> 每 token FFN：只读码字 %.3f ms | L2 直查 %.3f ms | SMEM64K %.3f ms | SMEM32K %.3f ms\n",
                t_codes, t_l2, t_s8, t_s4);
    std::printf("  >> 最佳查表 / 只读码字 = %.2f×（接近 1.0 说明查表已被读码字掩盖）\n", best / t_codes);
    if (std::string(m.name) == "Llama 3 8B")
      std::printf("  >> V2.8 停机规则：8B FFN 查表 > 2 ms/token 且无法缓解则放弃自研解码内核 —— 当前最佳 %.3f ms：%s\n",
                  best, best > 2.0 ? "触发停机，需分析" : "未触发");
  }
  std::printf("\n注：结果只含 GEMV 主体；建表、FWHT、残差、Epilogue、注意力和输出头另计。\n");
  return 0;
}
