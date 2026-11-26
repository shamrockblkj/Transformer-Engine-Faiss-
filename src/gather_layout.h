#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define GATHER_HD __host__ __device__
#else
#define GATHER_HD
#endif

// Llama 3 8B 的 FFN 参数量是 5,637,144,576。dsub = 4 时，每个参数对应一次查找。
inline constexpr std::int64_t kFfnLookups = 5637144576LL / 4;
inline constexpr double kGatherStopMs = 1.2;

// 码的交错布局是 [M / 16, N, 16]，每个 uint4 装 16 个码。
GATHER_HD inline std::size_t code_offset(int group, int row, int n_rows) {
    return (static_cast<std::size_t>(group) * static_cast<std::size_t>(n_rows) +
            static_cast<std::size_t>(row)) *
           16;
}

GATHER_HD inline std::size_t lut_offset(int subspace, int code) {
    return static_cast<std::size_t>(subspace) * 256 + static_cast<std::size_t>(code);
}

struct GatherShape {
    int n_rows;
    int n_subspaces;
    int split_k;
};

inline int groups_per_split(const GatherShape& shape) {
    const int groups = shape.n_subspaces / 16;
    return groups / shape.split_k;
}
