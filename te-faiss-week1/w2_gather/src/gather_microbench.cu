// W2 gather throughput microbenchmark (L2 direct vs SMEM-tiled LUT).
// Target: official GeForce RTX 4090 24GB, sm_89, CUDA 12.8.
// Scope: gather only. No full decode GEMV productization, no fused epilogue,
// no residual kernel, no CUDA Graph, no L2 persisting window.
//
// This file is meant to be compiled on the target host; it is not run here.

#include "llama3_8b_shapes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                      \
    do {                                                                      \
        cudaError_t _err = (expr);                                            \
        if (_err != cudaSuccess) {                                            \
            std::fprintf(stderr,                                              \
                         "CUDA error %s at %s:%d: %s\n",                      \
                         #expr,                                               \
                         __FILE__,                                            \
                         __LINE__,                                            \
                         cudaGetErrorString(_err));                           \
            std::exit(2);                                                     \
        }                                                                     \
    } while (0)

namespace {

using llama3_8b::GemvShape;
using llama3_8b::kBlock;
using llama3_8b::kCentroids;
using llama3_8b::kLayers;
using llama3_8b::kLookupsPerToken;
using llama3_8b::kPack;
using llama3_8b::kShapeDown;
using llama3_8b::kShapeGate;
using llama3_8b::kShapeUp;

enum class Path { L2, Smem, Both };

struct Options {
    Path path = Path::Both;
    int warmup = 5;
    int iters = 11;
    int layers = kLayers;
    bool do_gate = true;
    bool do_up = true;
    bool do_down = true;
    unsigned seed = 42;
    bool allow_non_target = false;
    bool opt_half2 = false;          // unapproved this week; default off
    bool opt_bank_replicate = false; // unapproved this week; default off
    bool print_config_only = false;
    bool skip_partial_check = false; // default: check ON before timing
    std::string json_path;
};

// Analytical L2-vs-SMEM partial-sum tolerances. NOT a 4090 measurement.
// Each Split-K partial is an FP32 sum of k4090PackedPerSplit * kPack = 128
// terms. __half2float is an exact promotion, and both kernels walk the same
// codes in the same order, so a correct SMEM preload should match L2
// bit-for-bit. The band below is a conservative analytical margin for
// benign codegen differences (for example unexpected contraction), not an
// empirical host error:
//   abs_tol = 2^{-10}          binary16 ulp at 1.0
//   rel_tol = 2 * 2^{-10}      two binary16 machine epsilons
// Pass iff |a-b| <= abs_tol + rel_tol * max(|a|, |b|).
// A single wrong LUT term under this generator is typically O(0.5-8)
// (FP16 exponents 14..17), far above this band. This is not Nsight.
constexpr float kPartialAbsTol = 0x1.0p-10f;       // 9.765625e-4f
constexpr float kPartialRelTol = 2.0f * 0x1.0p-10f;  // 1.953125e-3f

void usage(const char* argv0) {
    std::fprintf(
        stderr,
        "Usage: %s [options]\n"
        "  --path l2|smem|both     lookup path (default: both)\n"
        "  --warmup N              warmup launches (default: 5)\n"
        "  --iters N               timed repeats; report median (default: 11)\n"
        "  --layers N              FFN layers to simulate (default: 32)\n"
        "  --matrices LIST         comma list: gate,up,down (default: all)\n"
        "  --seed N                PRNG seed (default: 42)\n"
        "  --json PATH             write machine-readable results\n"
        "  --print-config-only     print shapes/tiles and exit (no launch)\n"
        "  --allow-non-target-gpu  allow other sm_89 cards (not Tesla P4)\n"
        "  --opt-half2             OPTIONAL, default off, not approved this week\n"
        "  --opt-bank-replicate    OPTIONAL, default off, not approved this week\n"
        "  --skip-partial-check    skip L2 vs SMEM partial-sum check (debug only; default ON)\n"
        "  -h, --help\n",
        argv0);
}

bool parse_matrices(const char* s, Options* o) {
    o->do_gate = o->do_up = o->do_down = false;
    std::string t(s);
    std::size_t start = 0;
    while (start <= t.size()) {
        auto comma = t.find(',', start);
        std::string tok = t.substr(start, comma == std::string::npos ? std::string::npos
                                                                     : comma - start);
        if (tok == "gate") o->do_gate = true;
        else if (tok == "up") o->do_up = true;
        else if (tok == "down") o->do_down = true;
        else if (!tok.empty()) {
            std::fprintf(stderr, "Unknown matrix name in --matrices: %s\n", tok.c_str());
            return false;
        }
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    if (!o->do_gate && !o->do_up && !o->do_down) {
        std::fprintf(stderr, "--matrices must include at least one of gate,up,down\n");
        return false;
    }
    return true;
}

bool parse_args(int argc, char** argv, Options* o) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        auto need = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for %s\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (!std::strcmp(a, "-h") || !std::strcmp(a, "--help")) {
            usage(argv[0]);
            std::exit(0);
        } else if (!std::strcmp(a, "--path")) {
            const char* v = need(a);
            if (!std::strcmp(v, "l2")) o->path = Path::L2;
            else if (!std::strcmp(v, "smem")) o->path = Path::Smem;
            else if (!std::strcmp(v, "both")) o->path = Path::Both;
            else {
                std::fprintf(stderr, "Invalid --path %s (use l2|smem|both)\n", v);
                return false;
            }
        } else if (!std::strcmp(a, "--warmup")) {
            o->warmup = std::atoi(need(a));
        } else if (!std::strcmp(a, "--iters")) {
            o->iters = std::atoi(need(a));
        } else if (!std::strcmp(a, "--layers")) {
            o->layers = std::atoi(need(a));
        } else if (!std::strcmp(a, "--matrices")) {
            if (!parse_matrices(need(a), o)) return false;
        } else if (!std::strcmp(a, "--seed")) {
            o->seed = static_cast<unsigned>(std::strtoul(need(a), nullptr, 10));
        } else if (!std::strcmp(a, "--json")) {
            o->json_path = need(a);
        } else if (!std::strcmp(a, "--print-config-only")) {
            o->print_config_only = true;
        } else if (!std::strcmp(a, "--allow-non-target-gpu")) {
            o->allow_non_target = true;
        } else if (!std::strcmp(a, "--opt-half2")) {
            o->opt_half2 = true;
        } else if (!std::strcmp(a, "--opt-bank-replicate")) {
            o->opt_bank_replicate = true;
        } else if (!std::strcmp(a, "--skip-partial-check")) {
            o->skip_partial_check = true;
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", a);
            usage(argv[0]);
            return false;
        }
    }
    if (o->warmup < 0 || o->iters < 1 || o->layers < 1) {
        std::fprintf(stderr, "--warmup must be >= 0, --iters >= 1, --layers >= 1\n");
        return false;
    }
    return true;
}

// ---- kernels ------------------------------------------------------------------

__device__ __forceinline__ unsigned xorshift32(unsigned x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

// Fill random uint8 codes already packed as uint4, and random finite FP16 LUT.
__global__ void init_random_payload(uint4* codes,
                                    std::size_t n_codes,
                                    __half* lut,
                                    std::size_t n_lut,
                                    unsigned seed,
                                    unsigned tag) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
    for (std::size_t c = i; c < n_codes; c += stride) {
        unsigned r0 = xorshift32(seed ^ tag ^ static_cast<unsigned>(c) ^ 0xA511E9B5u);
        unsigned r1 = xorshift32(r0);
        unsigned r2 = xorshift32(r1);
        unsigned r3 = xorshift32(r2);
        codes[c] = make_uint4(r0, r1, r2, r3);
    }
    for (std::size_t c = i; c < n_lut; c += stride) {
        // Keep a normal FP16 (no NaN/Inf); values do not affect gather timing.
        unsigned r = xorshift32(seed ^ 0x9E3779B9u ^ tag ^ static_cast<unsigned>(c * 0x85EBCA6Bu));
        unsigned mant = r & 0x3FFu;
        unsigned exp = 14u + ((r >> 10) & 0x3u);  // 14..17, modest magnitude
        unsigned sign = (r >> 31) & 1u;
        unsigned bits = (sign << 15) | (exp << 10) | mant;
        lut[c] = __ushort_as_half(static_cast<unsigned short>(bits));
    }
}

// Host-side bank-conflict hint for one packed group of random codes.
// Ada SMEM has 32 banks of 4-byte words. A 256-entry FP16 row is 512 B.
// Entry i sits in bank ((i / 2) % 32) if the row is 512 B aligned.
// This is a static address-pattern estimate, NOT an Nsight measurement.
__device__ __forceinline__ int bank_of_fp16_index(int centroid) {
    return (centroid / 2) % 32;
}

__global__ void bank_conflict_hint_kernel(const uint4* codes,
                                          int packed_groups,
                                          int N,
                                          int sample_groups,
                                          unsigned long long* conflict_sum,
                                          unsigned long long* warp_count,
                                          unsigned* max_degree) {
    const int g = blockIdx.y;
    if (g >= sample_groups || g >= packed_groups) return;
    const int row0 = (blockIdx.x * blockDim.x);
    if (row0 + 31 >= N) return;
    if ((threadIdx.x % 32) != 0) return;  // one thread per warp

    const int row = row0 + threadIdx.x;  // warp leader = first lane of this warp
    // Re-scan 32 consecutive rows starting at row (must be 32-aligned).
    if ((row % 32) != 0) return;
    if (row + 31 >= N) return;

    uint8_t codes16[32][16];
    for (int lane = 0; lane < 32; ++lane) {
        uint4 p = codes[static_cast<std::size_t>(g) * N + (row + lane)];
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&p);
        for (int k = 0; k < 16; ++k) codes16[lane][k] = b[k];
    }

    int local_max = 1;
    int local_sum = 0;
    for (int k = 0; k < 16; ++k) {
        int hist[32];
        for (int b = 0; b < 32; ++b) hist[b] = 0;
        for (int lane = 0; lane < 32; ++lane) {
            hist[bank_of_fp16_index(static_cast<int>(codes16[lane][k]))] += 1;
        }
        int deg = 1;
        for (int b = 0; b < 32; ++b) deg = hist[b] > deg ? hist[b] : deg;
        local_max = deg > local_max ? deg : local_max;
        local_sum += deg;
    }
    atomicAdd(conflict_sum, static_cast<unsigned long long>(local_sum));
    atomicAdd(warp_count, 16ull);  // 16 subspaces per packed group
    unsigned old = atomicMax(max_degree, static_cast<unsigned>(local_max));
    (void)old;
}

// Plan 4.1 L2-direct gather (control path). Dummy accumulate + store of
// Split-K partials only to keep the work live. Not a production decode GEMV.
__global__ void pq_lut_gather_l2(const __half* __restrict__ lut,
                                 const uint4* __restrict__ codes,
                                 float* __restrict__ partial,
                                 int N,
                                 int M,
                                 int split_k) {
    const int n_blocks = N / blockDim.x;
    const int split_id = blockIdx.x / n_blocks;
    const int row = (blockIdx.x % n_blocks) * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const int groups = M >> 4;
    const int per = groups / split_k;
    const int g0 = split_id * per;
    float acc = 0.f;

    for (int g = g0; g < g0 + per; ++g) {
        uint4 packed = codes[static_cast<std::size_t>(g) * N + row];
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&packed);
        const int m_base = g << 4;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc += __half2float(lut[static_cast<std::size_t>(m_base + k) * 256 + b[k]]);
        }
    }
    partial[static_cast<std::size_t>(split_id) * N + row] = acc;
}

// V2.6 default: load this split's LUT tile into SMEM, then gather from SMEM.
__global__ void pq_lut_gather_smem(const __half* __restrict__ lut,
                                   const uint4* __restrict__ codes,
                                   float* __restrict__ partial,
                                   int N,
                                   int M,
                                   int split_k) {
    extern __shared__ char smem[];
    __half* s_lut = reinterpret_cast<__half*>(smem);

    const int n_blocks = N / blockDim.x;
    const int split_id = blockIdx.x / n_blocks;
    const int row = (blockIdx.x % n_blocks) * blockDim.x + threadIdx.x;

    const int groups = M >> 4;
    const int per = groups / split_k;
    const int g0 = split_id * per;
    const int m_base = g0 << 4;
    const int tile_elems = per * 16 * 256;

    // Coalesced L2/HBM -> SMEM preload of a contiguous [per*16, 256] slice.
    const __half* gmem = lut + static_cast<std::size_t>(m_base) * 256;
    const int nvec = tile_elems / 8;  // 8 halfs per int4
    const int4* in = reinterpret_cast<const int4*>(gmem);
    int4* out = reinterpret_cast<int4*>(s_lut);
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
        out[i] = in[i];
    }
    __syncthreads();

    if (row >= N) return;

    float acc = 0.f;
    for (int g = g0; g < g0 + per; ++g) {
        uint4 packed = codes[static_cast<std::size_t>(g) * N + row];
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&packed);
        const int local_base = (g - g0) << 4;
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            acc += __half2float(s_lut[static_cast<int>((local_base + k) * 256 + b[k])]);
        }
    }
    partial[static_cast<std::size_t>(split_id) * N + row] = acc;
}

// ---- host helpers -------------------------------------------------------------

struct DeviceInfo {
    std::string name;
    int major = 0;
    int minor = 0;
    int sm_count = 0;
    std::size_t smem_per_sm = 0;
    std::size_t smem_per_block = 0;
    std::size_t smem_per_block_optin = 0;
    std::size_t l2_bytes = 0;
    std::size_t global_mem = 0;
};

bool name_looks_like_p4(const std::string& name) {
    std::string u = name;
    for (char& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return u.find("P4") != std::string::npos &&
           (u.find("TESLA") != std::string::npos || u.find("QUADRO") != std::string::npos ||
            u.find("TESLA P4") != std::string::npos || u == "P4" || u.find(" P4") != std::string::npos);
}

bool name_looks_like_4090(const std::string& name) {
    std::string u = name;
    for (char& c : u) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return u.find("4090") != std::string::npos;
}

DeviceInfo query_device_or_die(const Options& opt) {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev <= 0) {
        std::fprintf(stderr, "No CUDA device found. This microbench must run on the official RTX 4090 24GB.\n");
        std::exit(2);
    }
    if (ndev != 1) {
        std::fprintf(stderr,
                     "FATAL: official V2.8 acceptance requires a single GPU; cudaGetDeviceCount=%d. "
                     "Software memory caps are not official 24GB acceptance. Refusing to start benches.\n",
                     ndev);
        std::exit(2);
    }
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    DeviceInfo info;
    info.name = prop.name;
    info.major = prop.major;
    info.minor = prop.minor;
    info.sm_count = prop.multiProcessorCount;
    info.smem_per_sm = prop.sharedMemPerMultiprocessor;
    info.smem_per_block = prop.sharedMemPerBlock;
    info.smem_per_block_optin = prop.sharedMemPerBlockOptin;
    info.l2_bytes = static_cast<std::size_t>(prop.l2CacheSize);
    info.global_mem = prop.totalGlobalMem;

    std::printf("device: %s  cc=%d.%d  SMs=%d  SMEM/SM=%zu B  SMEM/block=%zu B  "
                "SMEM/block_optin=%zu B  L2=%zu B  global=%zu B\n",
                info.name.c_str(),
                info.major,
                info.minor,
                info.sm_count,
                info.smem_per_sm,
                info.smem_per_block,
                info.smem_per_block_optin,
                info.l2_bytes,
                info.global_mem);

    const bool too_old = (info.major < 8) || (info.major == 8 && info.minor < 9);
    if (too_old || name_looks_like_p4(info.name)) {
        std::fprintf(stderr,
                     "Refusing to run on %s (compute %d.%d). Target is RTX 4090 sm_89. "
                     "Tesla P4 and pre-Ada GPUs are out of scope.\n",
                     info.name.c_str(),
                     info.major,
                     info.minor);
        std::exit(2);
    }
    if (info.major != 8 || info.minor != 9) {
        std::fprintf(stderr,
                     "Refusing to run on %s (compute %d.%d). Binary is built for sm_89.\n",
                     info.name.c_str(),
                     info.major,
                     info.minor);
        std::exit(2);
    }
    if (!name_looks_like_4090(info.name) && !opt.allow_non_target) {
        std::fprintf(stderr,
                     "GPU name '%s' does not look like an RTX 4090. "
                     "Pass --allow-non-target-gpu only for another sm_89 card. "
                     "Do not mix SKUs into the 4090 S2 threshold.\n",
                     info.name.c_str());
        std::exit(2);
    }
    // Official 24GB check uses integer MiB (1024-based). Do NOT use totalGlobalMem/1e9.
    const unsigned long long mem_mib =
        static_cast<unsigned long long>(info.global_mem / (1024ull * 1024ull));
    std::printf("device memory: %llu MiB (official 24GB accept 24000-25000 MiB, typical ~24564)\n",
                mem_mib);
    std::string name_u = info.name;
    for (char& ch : name_u) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    const bool looks_4090d = name_u.find("4090D") != std::string::npos ||
                             name_u.find("4090 D") != std::string::npos;
    const bool official_24g = (mem_mib >= 24000ull && mem_mib <= 25000ull);
    const bool class_48g_or_larger = (mem_mib >= 40000ull);
    // Memory class is not waivable: a modified 48G card is not official 24GB acceptance.
    if (class_48g_or_larger || !official_24g || looks_4090d) {
        std::fprintf(stderr,
                     "FATAL: not an official GeForce RTX 4090 24GB. name='%s' memory=%llu MiB "
                     "(need 24000-25000 MiB, typical ~24564; fail on ~49140 or >=40000 MiB). "
                     "Software memory caps are not official 24GB acceptance. "
                     "Refusing to start benches.\n",
                     info.name.c_str(), mem_mib);
        std::exit(2);
    }
    if (info.smem_per_block_optin < static_cast<std::size_t>(llama3_8b::k4090TileBytes)) {
        std::fprintf(stderr,
                     "Device opt-in SMEM cap %zu B is smaller than the 4090 tile %d B. Stop.\n",
                     info.smem_per_block_optin,
                     llama3_8b::k4090TileBytes);
        std::exit(2);
    }
    return info;
}

void print_config(const Options& opt, const DeviceInfo* info) {
    std::printf("\n=== W2 gather microbench config ===\n");
    std::printf("model: official Meta-Llama-3-8B  hidden=%d  intermediate=%d  layers=%d\n",
                llama3_8b::kHidden,
                llama3_8b::kIntermediate,
                llama3_8b::kLayers);
    std::printf("PQ: dsub=%d  centroids=%d  lookups/token (full 32-layer FFN) = %llu\n",
                llama3_8b::kDSub,
                llama3_8b::kCentroids,
                static_cast<unsigned long long>(kLookupsPerToken));
    std::printf("  (source: plan B2.1, FFN params %llu / 4; this run uses --layers=%d)\n",
                static_cast<unsigned long long>(llama3_8b::kFfnParams),
                opt.layers);
    std::printf("H100 plan tiles (NOT used): gate SPLIT_K=%d tile=%d B (128 KiB); "
                "down SPLIT_K=%d tile=%d B (112 KiB)\n",
                llama3_8b::kH100GateSplitK,
                llama3_8b::kH100GateTileBytes,
                llama3_8b::kH100DownSplitK,
                llama3_8b::kH100DownTileBytes);
    std::printf("4090 legal tiles (used): gate/up SPLIT_K=%d, down SPLIT_K=%d, "
                "packed_per_split=%d, tile=%d B (64 KiB); block=%d\n",
                llama3_8b::k4090GateSplitK,
                llama3_8b::k4090DownSplitK,
                llama3_8b::k4090PackedPerSplit,
                llama3_8b::k4090TileBytes,
                kBlock);
    std::printf("  gate grid=%d  down grid=%d\n",
                llama3_8b::grid_blocks(kShapeGate),
                llama3_8b::grid_blocks(kShapeDown));
    std::printf("protocol: warmup=%d  iters=%d  median of timed iters; batch=1 decode gather\n",
                opt.warmup,
                opt.iters);
    std::printf("out of scope: fused epilogue, residual kernel, CUDA Graph, L2 window, "
                "full decode GEMV product.\n");
    std::printf("optional unapproved: --opt-half2=%s  --opt-bank-replicate=%s\n",
                opt.opt_half2 ? "on" : "off",
                opt.opt_bank_replicate ? "on" : "off");
    std::printf("partial-sum consistency check: %s  (abs_tol=%.6g  rel_tol=%.6g; "
                "analytical FP16/FP32 bound, not a 4090 measurement, not Nsight)\n",
                opt.skip_partial_check ? "SKIPPED" : "ON (default)",
                kPartialAbsTol,
                kPartialRelTol);
    if (info) {
        std::printf("runtime SM count=%d; grids must be >= this (gate %d, down %d)\n",
                    info->sm_count,
                    llama3_8b::grid_blocks(kShapeGate),
                    llama3_8b::grid_blocks(kShapeDown));
    }
}

struct MatBuf {
    GemvShape shape;
    int layers = 0;
    __half* lut = nullptr;     // [layers, M, 256]
    uint4* codes = nullptr;    // [layers, M/16, N]
    float* partial = nullptr;  // [split_k, N] reused across layers
};

void alloc_mat(MatBuf* b, const GemvShape& s, int layers) {
    b->shape = s;
    b->layers = layers;
    CUDA_CHECK(cudaMalloc(&b->lut, layers * llama3_8b::lut_elems(s) * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&b->codes, layers * llama3_8b::code_packed_elems(s) * sizeof(uint4)));
    CUDA_CHECK(cudaMalloc(&b->partial, llama3_8b::partial_elems(s) * sizeof(float)));
    CUDA_CHECK(cudaMemset(b->partial, 0, llama3_8b::partial_elems(s) * sizeof(float)));
}

void free_mat(MatBuf* b) {
    cudaFree(b->lut);
    cudaFree(b->codes);
    cudaFree(b->partial);
    b->lut = nullptr;
    b->codes = nullptr;
    b->partial = nullptr;
}

void init_mat(const MatBuf& b, unsigned seed, unsigned tag_base) {
    const int threads = 256;
    const int blocks = 1024;
    for (int layer = 0; layer < b.layers; ++layer) {
        __half* lut = b.lut + static_cast<std::size_t>(layer) * llama3_8b::lut_elems(b.shape);
        uint4* codes =
            b.codes + static_cast<std::size_t>(layer) * llama3_8b::code_packed_elems(b.shape);
        init_random_payload<<<blocks, threads>>>(codes,
                                                 llama3_8b::code_packed_elems(b.shape),
                                                 lut,
                                                 llama3_8b::lut_elems(b.shape),
                                                 seed,
                                                 tag_base + static_cast<unsigned>(layer) * 17u);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

void set_smem_attributes() {
    CUDA_CHECK(cudaFuncSetAttribute(pq_lut_gather_smem,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    llama3_8b::k4090TileBytes));
    CUDA_CHECK(cudaFuncSetAttribute(pq_lut_gather_smem,
                                    cudaFuncAttributePreferredSharedMemoryCarveout,
                                    cudaSharedmemCarveoutMaxShared));
}

void launch_one(Path path,
                const MatBuf& b,
                int layer,
                cudaStream_t stream,
                float* partial_out = nullptr) {
    const GemvShape& s = b.shape;
    float* partial = partial_out ? partial_out : b.partial;
    dim3 block(kBlock);
    dim3 grid(llama3_8b::grid_blocks(s));
    __half* lut = b.lut + static_cast<std::size_t>(layer) * llama3_8b::lut_elems(s);
    uint4* codes = b.codes + static_cast<std::size_t>(layer) * llama3_8b::code_packed_elems(s);
    if (path == Path::L2) {
        pq_lut_gather_l2<<<grid, block, 0, stream>>>(lut, codes, partial, s.N, s.M, s.split_k);
    } else if (path == Path::Smem) {
        pq_lut_gather_smem<<<grid, block, llama3_8b::tile_bytes(s), stream>>>(
            lut, codes, partial, s.N, s.M, s.split_k);
    }
}

struct Workload {
    MatBuf* mats[3]{};
    int n_mats = 0;
    int layers = 0;
    std::uint64_t lookups = 0;
};

void launch_token(Path path, const Workload& w, cudaStream_t stream) {
    for (int layer = 0; layer < w.layers; ++layer) {
        for (int i = 0; i < w.n_mats; ++i) {
            launch_one(path, *w.mats[i], layer, stream);
        }
    }
}

struct PartialCheckResult {
    bool ran = false;
    bool passed = false;
    std::uint64_t elems = 0;
    std::uint64_t n_fail = 0;
    float max_abs = 0.f;
    float max_rel = 0.f;
    const char* worst_name = "";
    int worst_layer = -1;
    int worst_split = -1;
    int worst_row = -1;
    float worst_l2 = 0.f;
    float worst_smem = 0.f;
};

// Compare L2-direct vs SMEM-tile Split-K partials on the same random codes
// and LUT, for every matrix and layer in the workload. Runs before warmup
// and timing. Returns passed=false on disagreement; the caller must exit
// without recording timings.
PartialCheckResult check_partial_consistency(const Workload& w, cudaStream_t stream) {
    PartialCheckResult r;
    r.ran = true;
    if (w.n_mats == 0) {
        r.passed = true;
        return r;
    }

    std::size_t max_elems = 0;
    for (int i = 0; i < w.n_mats; ++i) {
        max_elems = std::max(max_elems, llama3_8b::partial_elems(w.mats[i]->shape));
    }
    float* d_l2 = nullptr;
    float* d_smem = nullptr;
    CUDA_CHECK(cudaMalloc(&d_l2, max_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_smem, max_elems * sizeof(float)));
    std::vector<float> h_l2(max_elems);
    std::vector<float> h_smem(max_elems);

    for (int i = 0; i < w.n_mats; ++i) {
        const MatBuf& b = *w.mats[i];
        const std::size_t n = llama3_8b::partial_elems(b.shape);
        const int N = b.shape.N;
        for (int layer = 0; layer < w.layers; ++layer) {
            launch_one(Path::L2, b, layer, stream, d_l2);
            launch_one(Path::Smem, b, layer, stream, d_smem);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_l2.data(), d_l2, n * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_smem.data(), d_smem, n * sizeof(float), cudaMemcpyDeviceToHost));
            for (std::size_t e = 0; e < n; ++e) {
                const float a = h_l2[e];
                const float bval = h_smem[e];
                const float abs_err = std::fabs(a - bval);
                const float denom = std::max(std::fabs(a), std::fabs(bval));
                const float rel_err = (denom > 0.f) ? (abs_err / denom) : 0.f;
                r.elems += 1;
                if (rel_err > r.max_rel) r.max_rel = rel_err;
                if (abs_err > r.max_abs) {
                    r.max_abs = abs_err;
                    r.worst_name = b.shape.name;
                    r.worst_layer = layer;
                    r.worst_split = static_cast<int>(e / static_cast<std::size_t>(N));
                    r.worst_row = static_cast<int>(e % static_cast<std::size_t>(N));
                    r.worst_l2 = a;
                    r.worst_smem = bval;
                }
                if (abs_err > kPartialAbsTol + kPartialRelTol * denom) {
                    r.n_fail += 1;
                }
            }
        }
    }
    cudaFree(d_l2);
    cudaFree(d_smem);
    r.passed = (r.n_fail == 0);
    return r;
}

void print_partial_check(const PartialCheckResult& r) {
    if (r.passed) {
        std::printf("partial-sum consistency check: PASS  elems=%llu  "
                    "max_abs_err=%.6g  max_rel_err=%.6g  (abs_tol=%.6g  rel_tol=%.6g)\n",
                    static_cast<unsigned long long>(r.elems),
                    r.max_abs,
                    r.max_rel,
                    kPartialAbsTol,
                    kPartialRelTol);
        std::printf("  Same random uint8 codes + FP16 LUT; L2 vs SMEM Split-K partials. "
                    "Analytical tolerance, not a 4090 timing result, not Nsight.\n");
        return;
    }
    std::fprintf(stderr,
                 "error: L2 and SMEM Split-K partials disagree; refusing to record timings.\n"
                 "  worst: matrix=%s layer=%d split=%d row=%d  l2=%.9g  smem=%.9g\n"
                 "  max_abs_err=%.6g  max_rel_err=%.6g\n"
                 "  fails=%llu / elems=%llu\n"
                 "  threshold: |a-b| <= %.6g + %.6g * max(|a|,|b|)\n"
                 "  (analytical FP16/FP32 bound, not a measured 4090 error)\n",
                 r.worst_name,
                 r.worst_layer,
                 r.worst_split,
                 r.worst_row,
                 r.worst_l2,
                 r.worst_smem,
                 r.max_abs,
                 r.max_rel,
                 static_cast<unsigned long long>(r.n_fail),
                 static_cast<unsigned long long>(r.elems),
                 kPartialAbsTol,
                 kPartialRelTol);
}

double median_ms(std::vector<float> samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const int n = static_cast<int>(samples.size());
    if (n % 2 == 1) return samples[n / 2];
    return 0.5 * (static_cast<double>(samples[n / 2 - 1]) + samples[n / 2]);
}

struct Timed {
    std::vector<float> ms;
    double median = 0;
    float min_ms = 0;
    float max_ms = 0;
    double mean = 0;
    double p25 = 0;
    double p75 = 0;
};

Timed time_path(Path path, const Workload& w, int warmup, int iters, cudaStream_t stream) {
    for (int i = 0; i < warmup; ++i) {
        launch_token(path, w, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    Timed t;
    t.ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch_token(path, w, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        t.ms.push_back(ms);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    t.median = median_ms(t.ms);
    t.min_ms = *std::min_element(t.ms.begin(), t.ms.end());
    t.max_ms = *std::max_element(t.ms.begin(), t.ms.end());
    double sum = 0;
    for (float x : t.ms) sum += x;
    t.mean = sum / static_cast<double>(t.ms.size());
    auto sorted = t.ms;
    std::sort(sorted.begin(), sorted.end());
    auto at = [&](double q) {
        double idx = q * (sorted.size() - 1);
        int lo = static_cast<int>(std::floor(idx));
        int hi = static_cast<int>(std::ceil(idx));
        if (lo == hi) return static_cast<double>(sorted[lo]);
        double w = idx - lo;
        return sorted[lo] * (1.0 - w) + sorted[hi] * w;
    };
    t.p25 = at(0.25);
    t.p75 = at(0.75);
    return t;
}

struct BankHint {
    bool ok = false;
    double mean_degree = 0;
    unsigned max_degree = 0;
    unsigned long long warps = 0;
};

BankHint bank_conflict_hint(const MatBuf& b) {
    BankHint h;
    const int sample_groups = std::min(8, llama3_8b::packed_groups(b.shape));
    const int sample_row_blocks = std::min(8, b.shape.N / kBlock);
    unsigned long long* d_sum = nullptr;
    unsigned long long* d_cnt = nullptr;
    unsigned* d_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_cnt, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_cnt, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_max, 0, sizeof(unsigned)));
    dim3 block(kBlock);
    dim3 grid(sample_row_blocks, sample_groups);
    bank_conflict_hint_kernel<<<grid, block>>>(b.codes,
                                               llama3_8b::packed_groups(b.shape),
                                               b.shape.N,
                                               sample_groups,
                                               d_sum,
                                               d_cnt,
                                               d_max);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long sum = 0, cnt = 0;
    unsigned mx = 0;
    CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(sum), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&cnt, d_cnt, sizeof(cnt), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&mx, d_max, sizeof(mx), cudaMemcpyDeviceToHost));
    cudaFree(d_sum);
    cudaFree(d_cnt);
    cudaFree(d_max);
    if (cnt == 0) return h;
    h.ok = true;
    h.mean_degree = static_cast<double>(sum) / static_cast<double>(cnt);
    h.max_degree = mx;
    h.warps = cnt;
    return h;
}

void json_escape(FILE* f, const std::string& s) {
    std::fputc('"', f);
    for (char c : s) {
        if (c == '"' || c == '\\') {
            std::fputc('\\', f);
            std::fputc(c, f);
        } else if (c == '\n') {
            std::fputs("\\n", f);
        } else {
            std::fputc(c, f);
        }
    }
    std::fputc('"', f);
}

void write_path_json(FILE* f, const char* name, const Timed* t, std::uint64_t lookups) {
    std::fprintf(f, "    \"%s\": ", name);
    if (!t) {
        std::fprintf(f, "null");
        return;
    }
    const double ms_per_token = t->median * (static_cast<double>(kLookupsPerToken) /
                                             static_cast<double>(lookups));
    const double ns_per_lookup = (t->median * 1.0e6) / static_cast<double>(lookups);
    std::fprintf(f, "{\n");
    std::fprintf(f, "      \"median_ms_this_run\": %.6f,\n", t->median);
    std::fprintf(f, "      \"ms_per_token_converted\": %.6f,\n", ms_per_token);
    std::fprintf(f, "      \"ns_per_lookup\": %.6f,\n", ns_per_lookup);
    std::fprintf(f, "      \"min_ms\": %.6f,\n", t->min_ms);
    std::fprintf(f, "      \"max_ms\": %.6f,\n", t->max_ms);
    std::fprintf(f, "      \"mean_ms\": %.6f,\n", t->mean);
    std::fprintf(f, "      \"p25_ms\": %.6f,\n", t->p25);
    std::fprintf(f, "      \"p75_ms\": %.6f,\n", t->p75);
    std::fprintf(f, "      \"samples_ms\": [");
    for (std::size_t i = 0; i < t->ms.size(); ++i) {
        std::fprintf(f, "%s%.6f", i ? ", " : "", t->ms[i]);
    }
    std::fprintf(f, "],\n");
    std::fprintf(f,
                 "      \"l2_traffic_bytes_measured\": null,\n"
                 "      \"l2_traffic_note\": "
                 "\"待实测（请用 scripts/run_ncu.sh；不要把计划里的 22 GB 估算当成实测）\",\n"
                 "      \"bank_conflict_nsight\": null\n");
    std::fprintf(f, "    }");
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) return 2;

    if (opt.opt_half2 || opt.opt_bank_replicate) {
        std::fprintf(stderr,
                     "Optional path requested (--opt-half2=%d --opt-bank-replicate=%d). "
                     "These switches are left only as placeholders, default off, and are "
                     "NOT approved for this week's W2 measurement. Bank replication of the "
                     "64 KiB tile would also exceed the Ada 99 KB per-block cap unless "
                     "SPLIT_K is increased further. Exiting without running.\n",
                     opt.opt_half2 ? 1 : 0,
                     opt.opt_bank_replicate ? 1 : 0);
        return 3;
    }

    if (opt.print_config_only) {
        print_config(opt, nullptr);
        return 0;
    }

    DeviceInfo info = query_device_or_die(opt);
    print_config(opt, &info);
    if (llama3_8b::grid_blocks(kShapeGate) < info.sm_count ||
        llama3_8b::grid_blocks(kShapeDown) < info.sm_count) {
        std::fprintf(stderr,
                     "Grid (gate %d, down %d) is smaller than SM count %d. "
                     "Retune SPLIT_K before measuring.\n",
                     llama3_8b::grid_blocks(kShapeGate),
                     llama3_8b::grid_blocks(kShapeDown),
                     info.sm_count);
        return 2;
    }

    set_smem_attributes();

    MatBuf gate{}, up{}, down{};
    Workload work;
    work.layers = opt.layers;
    if (opt.do_gate) {
        alloc_mat(&gate, kShapeGate, opt.layers);
        init_mat(gate, opt.seed, 0x100u);
        work.mats[work.n_mats++] = &gate;
        work.lookups += static_cast<std::uint64_t>(opt.layers) * llama3_8b::lookups_one(kShapeGate);
    }
    if (opt.do_up) {
        alloc_mat(&up, kShapeUp, opt.layers);
        init_mat(up, opt.seed, 0x200u);
        work.mats[work.n_mats++] = &up;
        work.lookups += static_cast<std::uint64_t>(opt.layers) * llama3_8b::lookups_one(kShapeUp);
    }
    if (opt.do_down) {
        alloc_mat(&down, kShapeDown, opt.layers);
        init_mat(down, opt.seed, 0x300u);
        work.mats[work.n_mats++] = &down;
        work.lookups += static_cast<std::uint64_t>(opt.layers) * llama3_8b::lookups_one(kShapeDown);
    }

    std::printf("allocated unique random uint8 codes + FP16 LUTs for %d layer(s), "
                "lookups this run = %llu  (full-token equivalent = %llu)\n",
                opt.layers,
                static_cast<unsigned long long>(work.lookups),
                static_cast<unsigned long long>(kLookupsPerToken));
    std::printf("conversion: ms/token = median_ms * (%llu / lookups_this_run)\n",
                static_cast<unsigned long long>(kLookupsPerToken));
    std::printf("L2 traffic and Nsight bank conflicts: 待实测 (see scripts/run_ncu.sh)\n");

    // Software bank-conflict hint on layer-0 of the first matrix (迹象, not Nsight).
    BankHint hint{};
    if (work.n_mats > 0) {
        hint = bank_conflict_hint(*work.mats[0]);
        if (hint.ok) {
            std::printf("bank-conflict hint (host/device address pattern on %s layer 0, "
                        "NOT Nsight): mean replay degree = %.3f, max = %u, samples = %llu\n",
                        work.mats[0]->shape.name,
                        hint.mean_degree,
                        hint.max_degree,
                        static_cast<unsigned long long>(hint.warps));
            std::printf("  (Ada 32 banks, FP16 entry i -> bank (i/2)%%32; plan B2.1 estimated "
                        "3-4 way on H100 — that number is not a 4090 measurement.)\n");
        } else {
            std::printf("bank-conflict hint: 待实测 (sample collected no warps)\n");
        }
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    PartialCheckResult check_res{};
    if (opt.skip_partial_check) {
        std::printf("partial-sum consistency check: SKIPPED (--skip-partial-check); "
                    "timings may be recorded even if L2 and SMEM disagree.\n");
    } else {
        check_res = check_partial_consistency(work, stream);
        print_partial_check(check_res);
        if (!check_res.passed) {
            CUDA_CHECK(cudaStreamDestroy(stream));
            free_mat(&gate);
            free_mat(&up);
            free_mat(&down);
            return 4;
        }
    }

    const Timed* l2_ptr = nullptr;
    const Timed* smem_ptr = nullptr;
    Timed l2_t, smem_t;

    auto report = [&](const char* label, const Timed& t) {
        const double ms_tok = t.median * (static_cast<double>(kLookupsPerToken) /
                                          static_cast<double>(work.lookups));
        const double ns = (t.median * 1.0e6) / static_cast<double>(work.lookups);
        std::printf("\n[%s] median = %.4f ms  (min %.4f  p25 %.4f  p75 %.4f  max %.4f  mean %.4f)\n",
                    label,
                    t.median,
                    t.min_ms,
                    t.p25,
                    t.p75,
                    t.max_ms,
                    t.mean);
        std::printf("      converted ms/token = %.4f   ns/lookup = %.4f\n", ms_tok, ns);
        std::printf("      L2 traffic = 待实测   Nsight bank conflict = 待实测\n");
        std::printf("      samples_ms =");
        for (float x : t.ms) std::printf(" %.4f", x);
        std::printf("\n");
    };

    if (opt.path == Path::L2 || opt.path == Path::Both) {
        l2_t = time_path(Path::L2, work, opt.warmup, opt.iters, stream);
        l2_ptr = &l2_t;
        report("L2-direct", l2_t);
    }
    if (opt.path == Path::Smem || opt.path == Path::Both) {
        smem_t = time_path(Path::Smem, work, opt.warmup, opt.iters, stream);
        smem_ptr = &smem_t;
        report("SMEM-tile", smem_t);
    }

    std::printf("\nS2 threshold on this 4090: 待实测 / 留空. "
                "Calibrate only after the same-card TE FP8 DelayedScaling baseline. "
                "Do not copy H100 1.2 ms or the 0.5-0.7 ms SMEM estimate.\n");

    if (!opt.json_path.empty()) {
        FILE* f = std::fopen(opt.json_path.c_str(), "w");
        if (!f) {
            std::fprintf(stderr, "Failed to open --json path: %s\n", opt.json_path.c_str());
            return 2;
        }
        std::fprintf(f, "{\n");
        std::fprintf(f, "  \"device_name\": ");
        json_escape(f, info.name);
        std::fprintf(f, ",\n");
        std::fprintf(f, "  \"compute_capability\": \"%d.%d\",\n", info.major, info.minor);
        std::fprintf(f, "  \"sm_count\": %d,\n", info.sm_count);
        std::fprintf(f, "  \"smem_per_sm_bytes\": %zu,\n", info.smem_per_sm);
        std::fprintf(f, "  \"smem_per_block_optin_bytes\": %zu,\n", info.smem_per_block_optin);
        std::fprintf(f, "  \"l2_bytes\": %zu,\n", info.l2_bytes);
        std::fprintf(f, "  \"layers\": %d,\n", opt.layers);
        std::fprintf(f, "  \"warmup\": %d,\n", opt.warmup);
        std::fprintf(f, "  \"iters\": %d,\n", opt.iters);
        std::fprintf(f, "  \"partial_check_enabled\": %s,\n",
                     check_res.ran ? "true" : "false");
        if (check_res.ran) {
            std::fprintf(f, "  \"partial_check_passed\": %s,\n",
                         check_res.passed ? "true" : "false");
            std::fprintf(f, "  \"partial_check_elems\": %llu,\n",
                         static_cast<unsigned long long>(check_res.elems));
            std::fprintf(f, "  \"partial_check_max_abs_err\": %.8g,\n", check_res.max_abs);
            std::fprintf(f, "  \"partial_check_max_rel_err\": %.8g,\n", check_res.max_rel);
        } else {
            std::fprintf(f, "  \"partial_check_passed\": null,\n");
        }
        std::fprintf(f, "  \"partial_check_abs_tol\": %.8g,\n", kPartialAbsTol);
        std::fprintf(f, "  \"partial_check_rel_tol\": %.8g,\n", kPartialRelTol);
        std::fprintf(f,
                     "  \"partial_check_note\": "
                     "\"Analytical FP16/FP32 bound (abs=2^-10, rel=2*2^-10), not a 4090 "
                     "measurement, not Nsight.\",\n");
        std::fprintf(f, "  \"lookups_this_run\": %llu,\n",
                     static_cast<unsigned long long>(work.lookups));
        std::fprintf(f, "  \"lookups_per_token_full_ffn\": %llu,\n",
                     static_cast<unsigned long long>(kLookupsPerToken));
        std::fprintf(f,
                     "  \"conversion\": \"ms_per_token = median_ms * (lookups_per_token_full_ffn / "
                     "lookups_this_run)\",\n");
        std::fprintf(f, "  \"tile_bytes_4090\": %d,\n", llama3_8b::k4090TileBytes);
        std::fprintf(f, "  \"split_k_gate\": %d,\n", llama3_8b::k4090GateSplitK);
        std::fprintf(f, "  \"split_k_down\": %d,\n", llama3_8b::k4090DownSplitK);
        std::fprintf(f, "  \"s2_threshold_ms_per_token\": null,\n");
        std::fprintf(f,
                     "  \"s2_threshold_note\": "
                     "\"待实测。须等同卡 TE FP8 DelayedScaling 基线测完再标定；禁止填入 H100 的 "
                     "1.2 ms 或 0.5-0.7 ms 估算。\",\n");
        if (hint.ok) {
            std::fprintf(f,
                         "  \"bank_conflict_hint_mean_degree\": %.6f,\n"
                         "  \"bank_conflict_hint_max_degree\": %u,\n"
                         "  \"bank_conflict_hint_note\": "
                         "\"静态地址模式估算，非 Nsight 实测。\",\n",
                         hint.mean_degree,
                         hint.max_degree);
        } else {
            std::fprintf(f,
                         "  \"bank_conflict_hint_mean_degree\": null,\n"
                         "  \"bank_conflict_hint_max_degree\": null,\n");
        }
        std::fprintf(f, "  \"paths\": {\n");
        write_path_json(f, "l2", l2_ptr, work.lookups);
        std::fprintf(f, ",\n");
        write_path_json(f, "smem", smem_ptr, work.lookups);
        std::fprintf(f, "\n  }\n}\n");
        std::fclose(f);
        std::printf("wrote %s\n", opt.json_path.c_str());
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    free_mat(&gate);
    free_mat(&up);
    free_mat(&down);
    return 0;
}
