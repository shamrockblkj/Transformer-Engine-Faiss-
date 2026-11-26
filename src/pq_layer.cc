#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kDsub = 4;

struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> v;

    double& at(int r, int c) { return v[static_cast<size_t>(r) * cols + c]; }
    double at(int r, int c) const { return v[static_cast<size_t>(r) * cols + c]; }
};

Matrix make_matrix(int rows, int cols, double fill = 0.0) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.v.assign(static_cast<size_t>(rows) * cols, fill);
    return m;
}

void jacobi_eigen_4(double a_in[4][4], double q[4][4], double eval[4]) {
    double a[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            a[i][j] = a_in[i][j];
            q[i][j] = i == j ? 1.0 : 0.0;
        }
    }
    for (int sweep = 0; sweep < 12; ++sweep) {
        for (int p = 0; p < 3; ++p) {
            for (int r = p + 1; r < 4; ++r) {
                const double app = a[p][p];
                const double aqq = a[r][r];
                const double apq = a[p][r];
                if (std::fabs(apq) < 1e-15) {
                    continue;
                }
                const double tau = (aqq - app) / (2.0 * apq);
                const double t = (tau >= 0 ? 1.0 : -1.0) / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;
                for (int k = 0; k < 4; ++k) {
                    const double aik = a[k][p];
                    const double aiq = a[k][r];
                    a[k][p] = c * aik - s * aiq;
                    a[k][r] = s * aik + c * aiq;
                }
                for (int k = 0; k < 4; ++k) {
                    const double akp = a[p][k];
                    const double akr = a[r][k];
                    a[p][k] = c * akp - s * akr;
                    a[r][k] = s * akp + c * akr;
                    const double qkp = q[k][p];
                    const double qkr = q[k][r];
                    q[k][p] = c * qkp - s * qkr;
                    q[k][r] = s * qkp + c * qkr;
                }
                a[p][r] = 0.0;
                a[r][p] = 0.0;
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        eval[i] = a[i][i];
    }
}

// 特征值下限。低于此值时 Σ^{1/2} 与 Σ^{-1/2} 用同一个抬高后的值，往返仍成立，但变换不再是原始 Σ。
inline constexpr double kEigenFloor = 1e-8;

void apply_sym(const double q[4][4], const double scale[4], const double v[4], double out[4]) {
    double rot[4] = {};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            rot[j] += q[i][j] * v[i];
        }
        rot[j] *= scale[j];
    }
    for (int i = 0; i < 4; ++i) {
        out[i] = 0.0;
        for (int j = 0; j < 4; ++j) {
            out[i] += q[i][j] * rot[j];
        }
    }
}

struct SubspaceBasis {
    double sqrt_scale[4] = {};
    double inv_sqrt_scale[4] = {};
    double q[4][4] = {};
};

SubspaceBasis basis_from_sigma(const double sigma[4][4]) {
    SubspaceBasis b;
    double eval[4];
    double a[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            a[i][j] = sigma[i][j];
        }
    }
    jacobi_eigen_4(a, b.q, eval);
    for (int i = 0; i < 4; ++i) {
        eval[i] = std::max(eval[i], kEigenFloor);
        b.sqrt_scale[i] = std::sqrt(eval[i]);
        b.inv_sqrt_scale[i] = 1.0 / b.sqrt_scale[i];
    }
    return b;
}

void kmeans(const std::vector<double>& points, int n, int k, std::vector<double>* centroids, std::vector<int>* assign,
            std::mt19937* rng) {
    centroids->assign(static_cast<size_t>(k) * kDsub, 0.0);
    std::uniform_int_distribution<int> pick(0, n - 1);
    for (int c = 0; c < k; ++c) {
        const int row = pick(*rng);
        for (int d = 0; d < kDsub; ++d) {
            (*centroids)[static_cast<size_t>(c) * kDsub + d] = points[static_cast<size_t>(row) * kDsub + d];
        }
    }
    assign->assign(static_cast<size_t>(n), 0);
    std::vector<double> sum(static_cast<size_t>(k) * kDsub);
    std::vector<int> count(static_cast<size_t>(k));
    for (int iter = 0; iter < 8; ++iter) {
        std::fill(sum.begin(), sum.end(), 0.0);
        std::fill(count.begin(), count.end(), 0);
        for (int i = 0; i < n; ++i) {
            int best = 0;
            double best_d = 1e300;
            for (int c = 0; c < k; ++c) {
                double dist = 0.0;
                for (int d = 0; d < kDsub; ++d) {
                    const double diff = points[static_cast<size_t>(i) * kDsub + d] -
                                        (*centroids)[static_cast<size_t>(c) * kDsub + d];
                    dist += diff * diff;
                }
                if (dist < best_d) {
                    best_d = dist;
                    best = c;
                }
            }
            (*assign)[static_cast<size_t>(i)] = best;
            count[static_cast<size_t>(best)] += 1;
            for (int d = 0; d < kDsub; ++d) {
                sum[static_cast<size_t>(best) * kDsub + d] += points[static_cast<size_t>(i) * kDsub + d];
            }
        }
        for (int c = 0; c < k; ++c) {
            if (count[static_cast<size_t>(c)] == 0) {
                const int row = pick(*rng);
                for (int d = 0; d < kDsub; ++d) {
                    (*centroids)[static_cast<size_t>(c) * kDsub + d] = points[static_cast<size_t>(row) * kDsub + d];
                }
                continue;
            }
            for (int d = 0; d < kDsub; ++d) {
                (*centroids)[static_cast<size_t>(c) * kDsub + d] =
                    sum[static_cast<size_t>(c) * kDsub + d] / count[static_cast<size_t>(c)];
            }
        }
    }
}

struct Report {
    double weight_rel_small = 0.0;
    double output_mse = 0.0;
};

Report run_case(const Matrix& weight, const Matrix& activation, bool row_norm, bool whiten, int centroids_per_sub) {
    const int n = weight.rows;
    const int kdim = weight.cols;
    const int m = kdim / kDsub;
    std::vector<double> scale(static_cast<size_t>(n), 1.0);
    Matrix work = weight;
    if (row_norm) {
        for (int r = 0; r < n; ++r) {
            double ss = 0.0;
            for (int c = 0; c < kdim; ++c) {
                ss += weight.at(r, c) * weight.at(r, c);
            }
            scale[static_cast<size_t>(r)] = std::sqrt(ss / kdim);
            scale[static_cast<size_t>(r)] = std::max(scale[static_cast<size_t>(r)], 1e-8);
            for (int c = 0; c < kdim; ++c) {
                work.at(r, c) = weight.at(r, c) / scale[static_cast<size_t>(r)];
            }
        }
    }

    std::vector<SubspaceBasis> bases(static_cast<size_t>(m));
    if (whiten) {
        for (int s = 0; s < m; ++s) {
            double sigma[4][4] = {};
            for (int t = 0; t < activation.rows; ++t) {
                double x[4];
                for (int d = 0; d < 4; ++d) {
                    x[d] = activation.at(t, s * 4 + d);
                }
                for (int a = 0; a < 4; ++a) {
                    for (int b = 0; b < 4; ++b) {
                        sigma[a][b] += x[a] * x[b];
                    }
                }
            }
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    sigma[a][b] /= activation.rows;
                }
            }
            bases[static_cast<size_t>(s)] = basis_from_sigma(sigma);
            for (int r = 0; r < n; ++r) {
                double v[4], out[4];
                for (int d = 0; d < 4; ++d) {
                    v[d] = work.at(r, s * 4 + d);
                }
                apply_sym(bases[static_cast<size_t>(s)].q, bases[static_cast<size_t>(s)].sqrt_scale, v, out);
                for (int d = 0; d < 4; ++d) {
                    work.at(r, s * 4 + d) = out[d];
                }
            }
        }
    }

    Matrix recon = make_matrix(n, kdim);
    std::mt19937 rng(7);
    for (int s = 0; s < m; ++s) {
        std::vector<double> points(static_cast<size_t>(n) * kDsub);
        for (int r = 0; r < n; ++r) {
            for (int d = 0; d < 4; ++d) {
                points[static_cast<size_t>(r) * 4 + d] = work.at(r, s * 4 + d);
            }
        }
        std::vector<double> centroids;
        std::vector<int> assign;
        kmeans(points, n, centroids_per_sub, &centroids, &assign, &rng);
        for (int r = 0; r < n; ++r) {
            const int c = assign[static_cast<size_t>(r)];
            double v[4];
            for (int d = 0; d < 4; ++d) {
                v[d] = centroids[static_cast<size_t>(c) * 4 + d];
            }
            if (whiten) {
                double out[4];
                apply_sym(bases[static_cast<size_t>(s)].q, bases[static_cast<size_t>(s)].inv_sqrt_scale, v, out);
                for (int d = 0; d < 4; ++d) {
                    v[d] = out[d];
                }
            }
            for (int d = 0; d < 4; ++d) {
                recon.at(r, s * 4 + d) = v[d] * scale[static_cast<size_t>(r)];
            }
        }
    }

    std::vector<double> norms(static_cast<size_t>(n));
    for (int r = 0; r < n; ++r) {
        double ss = 0.0;
        for (int c = 0; c < kdim; ++c) {
            ss += weight.at(r, c) * weight.at(r, c);
        }
        norms[static_cast<size_t>(r)] = std::sqrt(ss);
    }
    std::vector<int> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return norms[a] < norms[b]; });
    double rel = 0.0;
    const int take = std::max(n / 4, 1);
    for (int i = 0; i < take; ++i) {
        const int r = order[static_cast<size_t>(i)];
        double err = 0.0;
        for (int c = 0; c < kdim; ++c) {
            const double diff = weight.at(r, c) - recon.at(r, c);
            err += diff * diff;
        }
        rel += std::sqrt(err) / std::max(norms[static_cast<size_t>(r)], 1e-8);
    }
    rel /= take;

    double out_mse = 0.0;
    const int tokens = activation.rows;
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < n; ++r) {
            double y = 0.0;
            double yh = 0.0;
            for (int c = 0; c < kdim; ++c) {
                const double x = activation.at(t, c);
                y += weight.at(r, c) * x;
                yh += recon.at(r, c) * x;
            }
            const double diff = y - yh;
            out_mse += diff * diff;
        }
    }
    out_mse /= static_cast<double>(tokens) * n;

    Report report;
    report.weight_rel_small = rel;
    report.output_mse = out_mse;
    return report;
}

Matrix small_row_weights(int rows, int cols, std::mt19937* rng) {
    Matrix w = make_matrix(rows, cols);
    std::normal_distribution<double> noise(0.0, 0.05);
    std::vector<double> large(static_cast<size_t>(cols));
    std::vector<double> small(static_cast<size_t>(cols));
    std::normal_distribution<double> dir(0.0, 1.0);
    for (int c = 0; c < cols; ++c) {
        large[static_cast<size_t>(c)] = dir(*rng);
        small[static_cast<size_t>(c)] = dir(*rng);
    }
    for (int r = 0; r < rows; ++r) {
        const bool tiny = r < rows / 4;
        const double amp = tiny ? 0.2 : 6.0;
        const std::vector<double>& base = tiny ? small : large;
        for (int c = 0; c < cols; ++c) {
            w.at(r, c) = amp * base[static_cast<size_t>(c)] + noise(*rng);
        }
    }
    return w;
}

Matrix ill_conditioned_activation(int tokens, int cols, std::mt19937* rng) {
    Matrix x = make_matrix(tokens, cols);
    std::normal_distribution<double> unit(0.0, 1.0);
    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < cols; ++c) {
            const double scale = (c % kDsub == 0) ? 8.0 : 0.05;
            x.at(t, c) = scale * unit(*rng);
        }
    }
    return x;
}

struct WhitenCheck {
    double roundtrip = 0.0;
    double direction = 0.0;
    int subspaces = 0;
    int eigenvalues = 0;
    int clamped = 0;
    double min_raw = 0.0;
};

WhitenCheck check_whiten(const Matrix& activation) {
    WhitenCheck out;
    std::mt19937 rng(3);
    std::normal_distribution<double> normal(0.0, 1.0);
    out.min_raw = 1e300;
    for (int trial = 0; trial < 32; ++trial) {
        double factor[4][4];
        double sigma[4][4] = {};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                factor[i][j] = normal(rng);
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                for (int k = 0; k < 4; ++k) {
                    sigma[i][j] += factor[k][i] * factor[k][j];
                }
            }
        }
        const SubspaceBasis basis = basis_from_sigma(sigma);
        for (int sample = 0; sample < 8; ++sample) {
            double v[4];
            double mid[4];
            double back[4];
            for (int d = 0; d < 4; ++d) {
                v[d] = normal(rng);
            }
            apply_sym(basis.q, basis.sqrt_scale, v, mid);
            apply_sym(basis.q, basis.inv_sqrt_scale, mid, back);
            for (int d = 0; d < 4; ++d) {
                out.roundtrip = std::max(out.roundtrip, std::fabs(back[d] - v[d]));
            }
            double quad = 0.0;
            double norm = 0.0;
            for (int i = 0; i < 4; ++i) {
                norm += mid[i] * mid[i];
                for (int j = 0; j < 4; ++j) {
                    quad += v[i] * sigma[i][j] * v[j];
                }
            }
            const double scale = std::max(std::fabs(quad), 1e-12);
            out.direction = std::max(out.direction, std::fabs(norm - quad) / scale);
        }
    }

    const int subspaces = activation.cols / kDsub;
    out.subspaces = subspaces;
    for (int s = 0; s < subspaces; ++s) {
        double sigma[4][4] = {};
        for (int t = 0; t < activation.rows; ++t) {
            double x[4];
            for (int d = 0; d < 4; ++d) {
                x[d] = activation.at(t, s * 4 + d);
            }
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    sigma[a][b] += x[a] * x[b];
                }
            }
        }
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                sigma[a][b] /= activation.rows;
            }
        }
        double raw[4];
        double q[4][4];
        double a[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                a[i][j] = sigma[i][j];
            }
        }
        jacobi_eigen_4(a, q, raw);
        for (int i = 0; i < 4; ++i) {
            out.eigenvalues += 1;
            out.min_raw = std::min(out.min_raw, raw[i]);
            if (raw[i] < kEigenFloor) {
                out.clamped += 1;
            }
        }
    }
    return out;
}

void dump_matrix(const char* path, const Matrix& m) {
    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "无法写出 %s\n", path);
        std::exit(1);
    }
    const std::int32_t rows = m.rows;
    const std::int32_t cols = m.cols;
    const bool ok = std::fwrite(&rows, sizeof(rows), 1, file) == 1 &&
                    std::fwrite(&cols, sizeof(cols), 1, file) == 1 &&
                    std::fwrite(m.v.data(), sizeof(double), m.v.size(), file) == m.v.size();
    std::fclose(file);
    if (!ok) {
        std::fprintf(stderr, "写出 %s 失败\n", path);
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    constexpr int kRows = 128;
    constexpr int kCols = 64;
    constexpr int kTokens = 32;
    constexpr int kCentroids = 8;
    std::mt19937 rng(11);
    const Matrix weight = small_row_weights(kRows, kCols, &rng);
    const Matrix activation = ill_conditioned_activation(kTokens, kCols, &rng);
    if (argc == 3 && std::strcmp(argv[1], "--dump-wx") == 0) {
        const std::string dir = argv[2];
        dump_matrix((dir + "/W.bin").c_str(), weight);
        dump_matrix((dir + "/X.bin").c_str(), activation);
        std::printf("已写出与 128×64 自检相同的 W 和 X：%s\n", argv[2]);
    }
    const WhitenCheck whiten = check_whiten(activation);
    std::printf("白化往返：不经量化，最大绝对误差 %.3e。\n", whiten.roundtrip);
    std::printf("前向核对：||Σ^{1/2} v||^2 与 v^T Σ v 的最大相对差 %.3e。\n", whiten.direction);
    std::printf("Σ 是 %d 个 4×4 块，不是 64×64。特征值下限 %.1e，原始特征值最小 %.3e，被抬高 %d / %d 个。\n",
                whiten.subspaces, kEigenFloor, whiten.min_raw, whiten.clamped, whiten.eigenvalues);
    if (!(whiten.roundtrip < 1e-12)) {
        std::fprintf(stderr, "白化后反白化没有恢复到 1e-12，0.4642 不能当原始空间误差\n");
        return 1;
    }
    if (!(whiten.direction < 1e-8)) {
        std::fprintf(stderr, "量化前的变换不是 Σ^{1/2}\n");
        return 1;
    }

    const Report raw = run_case(weight, activation, false, false, kCentroids);
    const Report rows = run_case(weight, activation, true, false, kCentroids);
    const Report white = run_case(weight, activation, false, true, kCentroids);
    const Report both = run_case(weight, activation, true, true, kCentroids);

    std::printf("单层 PQ 对照，行 %d，列 %d，子空间维 %d，每子空间质心 %d。\n", kRows, kCols, kDsub, kCentroids);
    std::printf("这是 128×64 算法自检，使用自制 k-means，不是 Faiss。不能当作 S3 或全尺寸结论。\n");
    std::printf("%-16s %14s %14s\n", "路径", "小行相对误差", "输出 MSE");
    std::printf("%-16s %14.4f %14.4f\n", "原样", raw.weight_rel_small, raw.output_mse);
    std::printf("%-16s %14.4f %14.4f\n", "行 RMS", rows.weight_rel_small, rows.output_mse);
    std::printf("%-16s %14.4f %14.4f\n", "子空间白化", white.weight_rel_small, white.output_mse);
    std::printf("%-16s %14.4f %14.4f\n", "两者都用", both.weight_rel_small, both.output_mse);

    if (!(rows.weight_rel_small < raw.weight_rel_small)) {
        std::fprintf(stderr, "行 RMS 没有降低小行的相对误差\n");
        return 1;
    }
    if (!(white.output_mse < raw.output_mse)) {
        std::fprintf(stderr, "子空间白化没有降低输出 MSE\n");
        return 1;
    }
    std::printf("对照成立：行 RMS 降低小行相对误差，子空间白化降低输出 MSE。\n");
    return 0;
}
