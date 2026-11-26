#include "gather_layout.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

std::uint32_t mix(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

void fill_problem(const GatherShape& shape, std::vector<std::uint8_t>* codes, std::vector<float>* lut) {
    codes->assign(static_cast<std::size_t>(shape.n_subspaces / 16) * shape.n_rows * 16, 0);
    lut->assign(static_cast<std::size_t>(shape.n_subspaces) * 256, 0.f);
    for (int m = 0; m < shape.n_subspaces; ++m) {
        for (int c = 0; c < 256; ++c) {
            const std::uint32_t h = mix(static_cast<std::uint32_t>(m * 256 + c + 1));
            (*lut)[lut_offset(m, c)] = static_cast<float>(static_cast<int>(h & 1023) - 512) / 64.f;
        }
    }
    for (int g = 0; g < shape.n_subspaces / 16; ++g) {
        for (int row = 0; row < shape.n_rows; ++row) {
            std::uint8_t* dst = codes->data() + code_offset(g, row, shape.n_rows);
            for (int k = 0; k < 16; ++k) {
                dst[k] = static_cast<std::uint8_t>(mix(static_cast<std::uint32_t>(g * 10007 + row * 17 + k)) & 255);
            }
        }
    }
}

std::vector<float> gather_global(const GatherShape& shape, const std::vector<std::uint8_t>& codes,
                                 const std::vector<float>& lut) {
    std::vector<float> out(static_cast<std::size_t>(shape.n_rows), 0.f);
    const int per = groups_per_split(shape);
    const int groups = shape.n_subspaces / 16;
    for (int split = 0; split < shape.split_k; ++split) {
        const int g0 = split * per;
        for (int row = 0; row < shape.n_rows; ++row) {
            float acc = 0.f;
            for (int g = g0; g < g0 + per; ++g) {
                const std::uint8_t* packed = codes.data() + code_offset(g, row, shape.n_rows);
                const int m_base = g * 16;
                for (int k = 0; k < 16; ++k) {
                    acc += lut[lut_offset(m_base + k, packed[k])];
                }
            }
            out[static_cast<std::size_t>(row)] += acc;
        }
    }
    (void)groups;
    return out;
}

std::vector<float> gather_smem_tile(const GatherShape& shape, const std::vector<std::uint8_t>& codes,
                                    const std::vector<float>& lut) {
    std::vector<float> out(static_cast<std::size_t>(shape.n_rows), 0.f);
    const int per = groups_per_split(shape);
    std::vector<float> tile(static_cast<std::size_t>(per) * 16 * 256);
    for (int split = 0; split < shape.split_k; ++split) {
        const int g0 = split * per;
        for (int local = 0; local < per * 16; ++local) {
            const int m = g0 * 16 + local;
            for (int c = 0; c < 256; ++c) {
                tile[static_cast<std::size_t>(local) * 256 + c] = lut[lut_offset(m, c)];
            }
        }
        for (int row = 0; row < shape.n_rows; ++row) {
            float acc = 0.f;
            for (int g = 0; g < per; ++g) {
                const std::uint8_t* packed = codes.data() + code_offset(g0 + g, row, shape.n_rows);
                for (int k = 0; k < 16; ++k) {
                    const int local = g * 16 + k;
                    acc += tile[static_cast<std::size_t>(local) * 256 + packed[k]];
                }
            }
            out[static_cast<std::size_t>(row)] += acc;
        }
    }
    return out;
}

bool close(const std::vector<float>& a, const std::vector<float>& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (std::fabs(a[i] - b[i]) > 1e-4f) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    const GatherShape shape{128, 64, 4};
    if (shape.n_subspaces % 16 != 0 || (shape.n_subspaces / 16) % shape.split_k != 0) {
        std::fprintf(stderr, "形状不能整除\n");
        return 1;
    }
    std::vector<std::uint8_t> codes;
    std::vector<float> lut;
    fill_problem(shape, &codes, &lut);
    const std::vector<float> global = gather_global(shape, codes, lut);
    const std::vector<float> tiled = gather_smem_tile(shape, codes, lut);
    if (!close(global, tiled)) {
        std::fprintf(stderr, "全局查表与 SMEM 切片查表的累加结果不一致\n");
        return 1;
    }
    const double lookups = static_cast<double>(shape.n_rows) * shape.n_subspaces;
    const double required = static_cast<double>(kFfnLookups) / (kGatherStopMs * 1e-3);
    std::printf("索引自检通过。样例查找 %g 次，两条路径结果一致。\n", lookups);
    std::printf("全模型 FFN 查找次数 %lld。方案里 H100 原文口径 %.1f ms/token，对应查找率不低于 %.3e 次/秒。\n",
                static_cast<long long>(kFfnLookups), kGatherStopMs, required);
    std::printf("这不是 RTX 4090 的阈值。S2 阈值留空，须用同卡 TE FP8 DelayedScaling 重标定。\n");
    std::printf("本机不编 CUDA、不计时。测量主机用 make gpu 编译 src/gather_bench.cu，默认包含 sm_89 和 sm_90。\n");
    std::printf("同一子空间内的随机码落在同一行，把 LUT 行距改成非 32 的倍数不能消除这次 bank conflict。\n");
    return 0;
}
