#pragma once

#include <cstddef>
#include <cstdint>

// Official Meta-Llama-3-8B shapes (not Llama 3.1 / 3.2).
// Source: meta-llama/Meta-Llama-3-8B config.json
//   hidden_size=4096, intermediate_size=14336, num_hidden_layers=32
// Plan reference: /workspace/te-faiss-v2.6-plan.md (核算对象与 B2.1).
// PQ: dsub=4, 256 centroids, 2 bit/parameter. Lookups per token = FFN params / 4.

namespace llama3_8b {

constexpr int kHidden = 4096;
constexpr int kIntermediate = 14336;
constexpr int kLayers = 32;
constexpr int kDSub = 4;
constexpr int kCentroids = 256;
constexpr int kPack = 16;       // uint8 codes packed as uint4; plan layout [M/16, N]
constexpr int kBlock = 256;     // 8 warps, same as plan 4.1 / B2.1
constexpr int kHalfBytes = 2;

// gate / up: N = intermediate, M = hidden / dsub
// down:      N = hidden,       M = intermediate / dsub
constexpr int kGateN = kIntermediate;          // 14336
constexpr int kGateM = kHidden / kDSub;        // 1024
constexpr int kDownN = kHidden;                // 4096
constexpr int kDownM = kIntermediate / kDSub;  // 3584

// Lookups = N * M per matrix. Per token, 32 layers * (gate + up + down).
constexpr std::uint64_t kLookupsGatePerLayer =
    static_cast<std::uint64_t>(kGateN) * static_cast<std::uint64_t>(kGateM);
constexpr std::uint64_t kLookupsDownPerLayer =
    static_cast<std::uint64_t>(kDownN) * static_cast<std::uint64_t>(kDownM);
constexpr std::uint64_t kLookupsPerLayer =
    2 * kLookupsGatePerLayer + kLookupsDownPerLayer;  // 44,040,192
constexpr std::uint64_t kLookupsPerToken =
    kLookupsPerLayer * static_cast<std::uint64_t>(kLayers);  // 1,409,286,144

// FFN parameter count used in plan B2.1: 5,637,144,576 = 32 * 3 * 4096 * 14336
constexpr std::uint64_t kFfnParams =
    static_cast<std::uint64_t>(kLayers) * 3ull * kHidden * kIntermediate;

// ---------------------------------------------------------------------------
// Split-K / SMEM tiles
//
// H100 estimate in plan B2.1 / 4.1 (NOT legal on 4090, documented only):
//   H100 SMEM 228 KB/SM, 227 KB/block (CUDA CC 9.0).
//   gate SPLIT_K=4  -> 16 packed groups * 16 * 256 * 2 B = 128 KiB
//   down SPLIT_K=16 -> 14 packed groups * 16 * 256 * 2 B = 112 KiB
//
// Ada / sm_89 (RTX 4090) official limits (CUDA Programming Guide Table 34;
// Ada Tuning Guide): 100 KB SMEM/SM, 99 KB/block (1 KB reserved), carveouts
// 0/8/16/32/64/100 KB. Static SMEM cap 48 KB; above that needs dynamic + opt-in.
// 128 KiB and 112 KiB both exceed 99 KB, so H100 tiles cannot be used.
//
// 4090 legal default (this microbench):
//   Keep block=256 (8 warps). Increase SPLIT_K so each block's LUT tile is
//   64 KiB (a legal Ada carveout), and packed-groups % SPLIT_K == 0.
//   gate/up: packed groups = 1024/16=64; SPLIT_K=8  -> 8 packed * 16*256*2 = 64 KiB
//   down:    packed groups = 3584/16=224; SPLIT_K=28 -> 8 packed * 16*256*2 = 64 KiB
//   Grid = (N/256)*SPLIT_K: gate 448, down 448, both >= public RTX 4090 128 SMs
//   (16384 CUDA cores / 128 cores per Ada SM). Runtime still queries SM count.
// ---------------------------------------------------------------------------

constexpr int kH100GateSplitK = 4;
constexpr int kH100DownSplitK = 16;
constexpr int kH100GateTileBytes = 16 * 16 * kCentroids * kHalfBytes;  // 131072
constexpr int kH100DownTileBytes = 14 * 16 * kCentroids * kHalfBytes;  // 114688

constexpr int k4090GateSplitK = 8;
constexpr int k4090DownSplitK = 28;
constexpr int k4090PackedPerSplit = 8;
constexpr int k4090TileBytes =
    k4090PackedPerSplit * kPack * kCentroids * kHalfBytes;  // 65536

static_assert(kFfnParams == 5637144576ull, "FFN param count must match plan B2.1");
static_assert(kLookupsPerToken == kFfnParams / 4ull, "lookups = FFN params / 4");
static_assert(kLookupsPerToken == 1409286144ull, "exact 1,409,286,144 lookups/token");
static_assert(kGateM % kPack == 0 && kDownM % kPack == 0, "M must be multiple of 16");
static_assert((kGateM / kPack) % k4090GateSplitK == 0, "gate packed groups % SPLIT_K");
static_assert((kDownM / kPack) % k4090DownSplitK == 0, "down packed groups % SPLIT_K");
static_assert((kGateM / kPack) / k4090GateSplitK == k4090PackedPerSplit, "gate tile");
static_assert((kDownM / kPack) / k4090DownSplitK == k4090PackedPerSplit, "down tile");
static_assert(kGateN % kBlock == 0 && kDownN % kBlock == 0, "N multiple of block");
static_assert(k4090TileBytes == 65536, "64 KiB tile");
static_assert(k4090TileBytes <= 99 * 1024, "must fit Ada 99 KB per-block opt-in cap");
static_assert(kH100GateTileBytes > 99 * 1024, "H100 gate tile must not silently fit 4090");
static_assert(kH100DownTileBytes > 99 * 1024, "H100 down tile must not silently fit 4090");

struct GemvShape {
    const char* name;
    int N;
    int M;
    int split_k;
};

constexpr GemvShape kShapeGate{"gate", kGateN, kGateM, k4090GateSplitK};
constexpr GemvShape kShapeUp{"up", kGateN, kGateM, k4090GateSplitK};
constexpr GemvShape kShapeDown{"down", kDownN, kDownM, k4090DownSplitK};

inline int packed_groups(const GemvShape& s) { return s.M / kPack; }
inline int packed_per_split(const GemvShape& s) { return packed_groups(s) / s.split_k; }
inline int tile_elems(const GemvShape& s) { return packed_per_split(s) * kPack * kCentroids; }
inline int tile_bytes(const GemvShape& s) { return tile_elems(s) * kHalfBytes; }
inline int grid_blocks(const GemvShape& s) { return (s.N / kBlock) * s.split_k; }
inline std::size_t lut_elems(const GemvShape& s) {
    return static_cast<std::size_t>(s.M) * kCentroids;
}
inline std::size_t code_packed_elems(const GemvShape& s) {
    return static_cast<std::size_t>(packed_groups(s)) * static_cast<std::size_t>(s.N);
}
inline std::size_t partial_elems(const GemvShape& s) {
    return static_cast<std::size_t>(s.split_k) * static_cast<std::size_t>(s.N);
}
inline std::uint64_t lookups_one(const GemvShape& s) {
    return static_cast<std::uint64_t>(s.N) * static_cast<std::uint64_t>(s.M);
}

}  // namespace llama3_8b
