#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
            std::mt19937* rng, int iterations = 8) {
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
    for (int iter = 0; iter < iterations; ++iter) {
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
    double within_mse = 0.0;
    double cross_mse = 0.0;
};

struct Prepared {
    Matrix work;
    std::vector<double> scale;
    std::vector<SubspaceBasis> bases;
    bool whiten = false;
};

Prepared prepare_work(const Matrix& weight, const Matrix& activation, bool row_norm, bool whiten, bool diagonal) {
    Prepared prep;
    prep.whiten = whiten;
    const int n = weight.rows;
    const int kdim = weight.cols;
    const int m = kdim / kDsub;
    prep.scale.assign(static_cast<size_t>(n), 1.0);
    prep.work = weight;
    if (row_norm) {
        for (int r = 0; r < n; ++r) {
            double ss = 0.0;
            for (int c = 0; c < kdim; ++c) {
                ss += weight.at(r, c) * weight.at(r, c);
            }
            prep.scale[static_cast<size_t>(r)] = std::sqrt(ss / kdim);
            prep.scale[static_cast<size_t>(r)] = std::max(prep.scale[static_cast<size_t>(r)], 1e-8);
            for (int c = 0; c < kdim; ++c) {
                prep.work.at(r, c) = weight.at(r, c) / prep.scale[static_cast<size_t>(r)];
            }
        }
    }
    prep.bases.resize(static_cast<size_t>(m));
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
            if (diagonal) {
                SubspaceBasis basis;
                for (int i = 0; i < 4; ++i) {
                    basis.q[i][i] = 1.0;
                    const double ev = std::max(sigma[i][i], kEigenFloor);
                    basis.sqrt_scale[i] = std::sqrt(ev);
                    basis.inv_sqrt_scale[i] = 1.0 / basis.sqrt_scale[i];
                }
                prep.bases[static_cast<size_t>(s)] = basis;
            } else {
                prep.bases[static_cast<size_t>(s)] = basis_from_sigma(sigma);
            }
            for (int r = 0; r < n; ++r) {
                double v[4];
                double out[4];
                for (int d = 0; d < 4; ++d) {
                    v[d] = prep.work.at(r, s * 4 + d);
                }
                apply_sym(prep.bases[static_cast<size_t>(s)].q, prep.bases[static_cast<size_t>(s)].sqrt_scale, v, out);
                for (int d = 0; d < 4; ++d) {
                    prep.work.at(r, s * 4 + d) = out[d];
                }
            }
        }
    }
    return prep;
}

Matrix restore_work(const Matrix& coded, const Prepared& prep) {
    Matrix out = make_matrix(coded.rows, coded.cols);
    const int m = coded.cols / kDsub;
    for (int s = 0; s < m; ++s) {
        for (int r = 0; r < coded.rows; ++r) {
            double v[4];
            for (int d = 0; d < 4; ++d) {
                v[d] = coded.at(r, s * 4 + d);
            }
            if (prep.whiten) {
                double back[4];
                apply_sym(prep.bases[static_cast<size_t>(s)].q, prep.bases[static_cast<size_t>(s)].inv_sqrt_scale, v, back);
                for (int d = 0; d < 4; ++d) {
                    v[d] = back[d];
                }
            }
            for (int d = 0; d < 4; ++d) {
                out.at(r, s * 4 + d) = v[d] * prep.scale[static_cast<size_t>(r)];
            }
        }
    }
    return out;
}

void fill_error_split(const Matrix& weight, const Matrix& recon, const Matrix& activation, Report* report) {
    const int n = weight.rows;
    const int m = weight.cols / kDsub;
    const int tokens = activation.rows;
    double within = 0.0;
    double total = 0.0;
    for (int t = 0; t < tokens; ++t) {
        for (int r = 0; r < n; ++r) {
            double sum = 0.0;
            double within_row = 0.0;
            for (int s = 0; s < m; ++s) {
                double dot = 0.0;
                for (int d = 0; d < 4; ++d) {
                    const int c = s * 4 + d;
                    dot += (weight.at(r, c) - recon.at(r, c)) * activation.at(t, c);
                }
                within_row += dot * dot;
                sum += dot;
            }
            within += within_row;
            total += sum * sum;
        }
    }
    const double denom = static_cast<double>(tokens) * n;
    report->within_mse = within / denom;
    report->cross_mse = (total - within) / denom;
}

Report run_case(const Matrix& weight, const Matrix& activation, bool row_norm, bool whiten, int centroids_per_sub,
                bool diagonal, Matrix* recon_out = nullptr, int kmeans_seed = 7, int iterations = 8) {
    const int n = weight.rows;
    const int kdim = weight.cols;
    const int m = kdim / kDsub;
    const Prepared prep = prepare_work(weight, activation, row_norm, whiten, diagonal);
    const Matrix& work = prep.work;
    const std::vector<double>& scale = prep.scale;
    const std::vector<SubspaceBasis>& bases = prep.bases;

    Matrix recon = make_matrix(n, kdim);
    std::mt19937 rng(kmeans_seed);
    for (int s = 0; s < m; ++s) {
        std::vector<double> points(static_cast<size_t>(n) * kDsub);
        for (int r = 0; r < n; ++r) {
            for (int d = 0; d < 4; ++d) {
                points[static_cast<size_t>(r) * 4 + d] = work.at(r, s * 4 + d);
            }
        }
        std::vector<double> centroids;
        std::vector<int> assign;
        kmeans(points, n, centroids_per_sub, &centroids, &assign, &rng, iterations);
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
    fill_error_split(weight, recon, activation, &report);
    if (recon_out != nullptr) {
        *recon_out = recon;
    }
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
    // 合成自检：每个坐标独立，第 0 个坐标方差大。不是 Llama 3 8B 的激活。
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

void random_orthogonal(std::mt19937* rng, double q[4][4]) {
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            q[row][col] = normal(*rng);
        }
        for (int prev = 0; prev < col; ++prev) {
            double dot = 0.0;
            for (int row = 0; row < 4; ++row) {
                dot += q[row][col] * q[row][prev];
            }
            for (int row = 0; row < 4; ++row) {
                q[row][col] -= dot * q[row][prev];
            }
        }
        double norm = 0.0;
        for (int row = 0; row < 4; ++row) {
            norm += q[row][col] * q[row][col];
        }
        norm = std::sqrt(norm);
        for (int row = 0; row < 4; ++row) {
            q[row][col] /= norm;
        }
    }
}

Matrix block_correlated_activation(int tokens, int cols, std::mt19937* rng) {
    // 阳性对照。先在潜变量上给不同方差，再左乘该块的随机正交阵。
    // 先旋转各向同性噪声、再按观测坐标缩放，得到的仍是对角 Σ，分不出两条路径。
    Matrix x = make_matrix(tokens, cols);
    std::normal_distribution<double> unit(0.0, 1.0);
    const double latent_scale[4] = {8.0, 0.05, 0.05, 0.05};
    const int subspaces = cols / kDsub;
    for (int s = 0; s < subspaces; ++s) {
        double q[4][4];
        random_orthogonal(rng, q);
        for (int t = 0; t < tokens; ++t) {
            double latent[4];
            for (int d = 0; d < 4; ++d) {
                latent[d] = latent_scale[d] * unit(*rng);
            }
            for (int row = 0; row < 4; ++row) {
                double value = 0.0;
                for (int col = 0; col < 4; ++col) {
                    value += q[row][col] * latent[col];
                }
                x.at(t, s * 4 + row) = value;
            }
        }
    }
    return x;
}

void random_orthogonal_n(std::mt19937* rng, int n, std::vector<double>* q) {
    q->assign(static_cast<size_t>(n) * n, 0.0);
    std::normal_distribution<double> normal(0.0, 1.0);
    auto at = [&](int row, int col) -> double& { return (*q)[static_cast<size_t>(row) * n + col]; };
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < n; ++row) {
            at(row, col) = normal(*rng);
        }
        for (int prev = 0; prev < col; ++prev) {
            double dot = 0.0;
            for (int row = 0; row < n; ++row) {
                dot += at(row, col) * at(row, prev);
            }
            for (int row = 0; row < n; ++row) {
                at(row, col) -= dot * at(row, prev);
            }
        }
        double norm = 0.0;
        for (int row = 0; row < n; ++row) {
            norm += at(row, col) * at(row, col);
        }
        norm = std::sqrt(std::max(norm, 1e-30));
        for (int row = 0; row < n; ++row) {
            at(row, col) /= norm;
        }
    }
}

void fill_scaled_activation(Matrix* x, const std::vector<double>& q, const std::vector<double>& scale, std::mt19937* rng) {
    std::normal_distribution<double> unit(0.0, 1.0);
    std::vector<double> latent(static_cast<size_t>(x->cols));
    for (int t = 0; t < x->rows; ++t) {
        for (int c = 0; c < x->cols; ++c) {
            latent[static_cast<size_t>(c)] = scale[static_cast<size_t>(c)] * unit(*rng);
        }
        for (int row = 0; row < x->cols; ++row) {
            double value = 0.0;
            for (int col = 0; col < x->cols; ++col) {
                value += q[static_cast<size_t>(row) * x->cols + col] * latent[static_cast<size_t>(col)];
            }
            x->at(t, row) = value;
        }
    }
}

void fill_latent_activation(Matrix* x, const std::vector<double>& q, std::mt19937* rng, bool one_large) {
    std::vector<double> scale(static_cast<size_t>(x->cols));
    for (int c = 0; c < x->cols; ++c) {
        scale[static_cast<size_t>(c)] = one_large ? (c == 0 ? 8.0 : 0.05) : ((c % kDsub == 0) ? 8.0 : 0.05);
    }
    fill_scaled_activation(x, q, scale, rng);
}

Matrix cross_block_activation(int tokens, int cols, std::mt19937* rng, bool one_large) {
    // 先按坐标缩放潜变量，再左乘全局正交阵。旋转必须作用在 X 上。
    // one_large：只有第 0 个潜变量是 8，其余是 0.05。
    // 否则每 4 个坐标里有一个是 8。各向同性噪声再旋转，Σ 仍接近单位阵。
    std::vector<double> q;
    random_orthogonal_n(rng, cols, &q);
    Matrix x = make_matrix(tokens, cols);
    fill_latent_activation(&x, q, rng, one_large);
    return x;
}

void jacobi_eigen_n(std::vector<double>* a, int n, std::vector<double>* vectors, std::vector<double>* eval) {
    vectors->assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        (*vectors)[static_cast<size_t>(i) * n + i] = 1.0;
    }
    auto at = [&](int r, int c) -> double& { return (*a)[static_cast<size_t>(r) * n + c]; };
    for (int sweep = 0; sweep < 50; ++sweep) {
        double off = 0.0;
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                off += at(p, q) * at(p, q);
            }
        }
        if (off < 1e-24 * n) {
            break;
        }
        for (int p = 0; p < n; ++p) {
            for (int q = p + 1; q < n; ++q) {
                const double apq = at(p, q);
                if (std::fabs(apq) < 1e-18) {
                    continue;
                }
                const double app = at(p, p);
                const double aqq = at(q, q);
                const double tau = (aqq - app) / (2.0 * apq);
                const double t = ((tau >= 0.0) ? 1.0 : -1.0) / (std::fabs(tau) + std::sqrt(1.0 + tau * tau));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;
                at(p, p) = app - t * apq;
                at(q, q) = aqq + t * apq;
                at(p, q) = 0.0;
                at(q, p) = 0.0;
                for (int k = 0; k < n; ++k) {
                    if (k == p || k == q) {
                        continue;
                    }
                    const double aik = at(p, k);
                    const double aqk = at(q, k);
                    const double new_pk = c * aik - s * aqk;
                    const double new_qk = s * aik + c * aqk;
                    at(p, k) = new_pk;
                    at(k, p) = new_pk;
                    at(q, k) = new_qk;
                    at(k, q) = new_qk;
                }
                for (int k = 0; k < n; ++k) {
                    const double vip = (*vectors)[static_cast<size_t>(k) * n + p];
                    const double viq = (*vectors)[static_cast<size_t>(k) * n + q];
                    (*vectors)[static_cast<size_t>(k) * n + p] = c * vip - s * viq;
                    (*vectors)[static_cast<size_t>(k) * n + q] = s * vip + c * viq;
                }
            }
        }
    }
    eval->resize(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        (*eval)[static_cast<size_t>(i)] = at(i, i);
    }
}

double held_out_mse(const Matrix& weight, const Matrix& recon, const Matrix& eval_act) {
    double total = 0.0;
    for (int t = 0; t < eval_act.rows; ++t) {
        for (int r = 0; r < weight.rows; ++r) {
            double y = 0.0;
            double yh = 0.0;
            for (int c = 0; c < weight.cols; ++c) {
                const double x = eval_act.at(t, c);
                y += weight.at(r, c) * x;
                yh += recon.at(r, c) * x;
            }
            const double diff = y - yh;
            total += diff * diff;
        }
    }
    return total / (static_cast<double>(eval_act.rows) * weight.rows);
}

Matrix full_whiten_recon(const Matrix& weight, const Matrix& calib, int centroids, int seed, int iterations,
                         double eps_rel = 1e-6, int keep = 0) {
    const int cols = weight.cols;
    const int tokens = calib.rows;
    std::vector<double> sigma(static_cast<size_t>(cols) * cols, 0.0);
    for (int t = 0; t < tokens; ++t) {
        for (int a = 0; a < cols; ++a) {
            for (int b = 0; b < cols; ++b) {
                sigma[static_cast<size_t>(a) * cols + b] += calib.at(t, a) * calib.at(t, b);
            }
        }
    }
    double trace = 0.0;
    for (int i = 0; i < cols; ++i) {
        sigma[static_cast<size_t>(i) * cols + i] /= tokens;
        trace += sigma[static_cast<size_t>(i) * cols + i];
        for (int j = i + 1; j < cols; ++j) {
            const double value = 0.5 * (sigma[static_cast<size_t>(i) * cols + j] + sigma[static_cast<size_t>(j) * cols + i]) / tokens;
            sigma[static_cast<size_t>(i) * cols + j] = value;
            sigma[static_cast<size_t>(j) * cols + i] = value;
        }
    }
    const double eps = eps_rel * trace / cols;
    for (int i = 0; i < cols; ++i) {
        sigma[static_cast<size_t>(i) * cols + i] += eps;
    }
    const std::vector<double> second = sigma;
    std::vector<double> vectors;
    std::vector<double> eval;
    jacobi_eigen_n(&sigma, cols, &vectors, &eval);
    double floor_lam = eps;
    if (keep > 0) {
        const int kept = std::min(keep, cols);
        std::vector<double> order = eval;
        std::nth_element(order.begin(), order.begin() + (kept - 1), order.end(), std::greater<double>());
        floor_lam = std::max(eps, order[static_cast<size_t>(kept - 1)]);
    }
    std::vector<double> used(static_cast<size_t>(cols));
    std::vector<double> root(static_cast<size_t>(cols) * cols, 0.0);
    std::vector<double> inv(static_cast<size_t>(cols) * cols, 0.0);
    for (int i = 0; i < cols; ++i) {
        const double lam = std::max(eval[static_cast<size_t>(i)], floor_lam);
        used[static_cast<size_t>(i)] = lam;
        const double s = std::sqrt(lam);
        const double inv_s = 1.0 / s;
        for (int r = 0; r < cols; ++r) {
            for (int c = 0; c < cols; ++c) {
                const double vir = vectors[static_cast<size_t>(r) * cols + i];
                const double vic = vectors[static_cast<size_t>(c) * cols + i];
                root[static_cast<size_t>(r) * cols + c] += vir * s * vic;
                inv[static_cast<size_t>(r) * cols + c] += vir * inv_s * vic;
            }
        }
    }
    Matrix whitened = make_matrix(weight.rows, cols);
    for (int r = 0; r < weight.rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double value = 0.0;
            for (int k = 0; k < cols; ++k) {
                value += weight.at(r, k) * root[static_cast<size_t>(k) * cols + c];
            }
            whitened.at(r, c) = value;
        }
    }
    Matrix recon_w = make_matrix(weight.rows, cols);
    std::mt19937 rng(seed);
    for (int s = 0; s < cols / kDsub; ++s) {
        std::vector<double> points(static_cast<size_t>(weight.rows) * kDsub);
        for (int r = 0; r < weight.rows; ++r) {
            for (int d = 0; d < kDsub; ++d) {
                points[static_cast<size_t>(r) * kDsub + d] = whitened.at(r, s * kDsub + d);
            }
        }
        std::vector<double> centers;
        std::vector<int> assign;
        kmeans(points, weight.rows, centroids, &centers, &assign, &rng, iterations);
        for (int r = 0; r < weight.rows; ++r) {
            const int chosen = assign[static_cast<size_t>(r)];
            for (int d = 0; d < kDsub; ++d) {
                recon_w.at(r, s * kDsub + d) = centers[static_cast<size_t>(chosen) * kDsub + d];
            }
        }
    }
    std::vector<double> probe(static_cast<size_t>(cols));
    for (int i = 0; i < cols; ++i) {
        probe[static_cast<size_t>(i)] = 0.1 * ((i % 7) - 3);
    }
    double quad = 0.0;
    double norm = 0.0;
    for (int i = 0; i < cols; ++i) {
        double mid = 0.0;
        for (int k = 0; k < cols; ++k) {
            mid += root[static_cast<size_t>(k) * cols + i] * probe[static_cast<size_t>(k)];
        }
        norm += mid * mid;
        for (int j = 0; j < cols; ++j) {
            quad += probe[static_cast<size_t>(i)] * second[static_cast<size_t>(i) * cols + j] * probe[static_cast<size_t>(j)];
        }
    }
    double projected = 0.0;
    for (int i = 0; i < cols; ++i) {
        double dot = 0.0;
        for (int k = 0; k < cols; ++k) {
            dot += vectors[static_cast<size_t>(k) * cols + i] * probe[static_cast<size_t>(k)];
        }
        projected += used[static_cast<size_t>(i)] * dot * dot;
    }
    const double direction = std::fabs(norm - projected) / std::max(std::fabs(projected), 1e-12);
    if (!(direction < 1e-6)) {
        std::fprintf(stderr, "完整白化的前向不是 Σ^{1/2}，相对差 %.3e\n", direction);
        std::exit(1);
    }
    if (keep == 0 && eps_rel <= 1e-6) {
        const double raw_direction = std::fabs(norm - quad) / std::max(std::fabs(quad), 1e-12);
        if (!(raw_direction < 1e-6)) {
            std::fprintf(stderr, "完整白化的前向不是 Σ^{1/2}，相对差 %.3e\n", raw_direction);
            std::exit(1);
        }
    }
    Matrix recon = make_matrix(weight.rows, cols);
    double roundtrip = 0.0;
    for (int r = 0; r < weight.rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            double value = 0.0;
            double back = 0.0;
            for (int k = 0; k < cols; ++k) {
                value += recon_w.at(r, k) * inv[static_cast<size_t>(k) * cols + c];
                back += whitened.at(r, k) * inv[static_cast<size_t>(k) * cols + c];
            }
            recon.at(r, c) = value;
            roundtrip = std::max(roundtrip, std::fabs(back - weight.at(r, c)));
        }
    }
    if (!(roundtrip < 1e-6)) {
        std::fprintf(stderr, "完整白化不量化时没有回到权重，误差 %.3e\n", roundtrip);
        std::exit(1);
    }
    return recon;
}

double sample_mean(const double* values, int count) {
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        sum += values[i];
    }
    return sum / count;
}

double sample_std(const double* values, int count) {
    const double mean = sample_mean(values, count);
    double sum = 0.0;
    for (int i = 0; i < count; ++i) {
        const double delta = values[i] - mean;
        sum += delta * delta;
    }
    return std::sqrt(sum / (count - 1));
}

void print_mean_std(const char* label, const double* values, int count) {
    std::printf("%s %.4f ± %.4f", label, sample_mean(values, count), sample_std(values, count));
}

void symmetrize_moment(const Matrix& act, std::vector<double>* sigma, double* trace) {
    const int cols = act.cols;
    sigma->assign(static_cast<size_t>(cols) * cols, 0.0);
    for (int t = 0; t < act.rows; ++t) {
        for (int a = 0; a < cols; ++a) {
            for (int b = 0; b < cols; ++b) {
                (*sigma)[static_cast<size_t>(a) * cols + b] += act.at(t, a) * act.at(t, b);
            }
        }
    }
    *trace = 0.0;
    for (int i = 0; i < cols; ++i) {
        (*sigma)[static_cast<size_t>(i) * cols + i] /= act.rows;
        *trace += (*sigma)[static_cast<size_t>(i) * cols + i];
        for (int j = i + 1; j < cols; ++j) {
            const double value =
                0.5 * ((*sigma)[static_cast<size_t>(i) * cols + j] + (*sigma)[static_cast<size_t>(j) * cols + i]) / act.rows;
            (*sigma)[static_cast<size_t>(i) * cols + j] = value;
            (*sigma)[static_cast<size_t>(j) * cols + i] = value;
        }
    }
}

double off_block_energy(const Matrix& act) {
    std::vector<double> sigma;
    double trace = 0.0;
    symmetrize_moment(act, &sigma, &trace);
    const int cols = act.cols;
    double total = 0.0;
    double within = 0.0;
    for (int a = 0; a < cols; ++a) {
        for (int b = 0; b < cols; ++b) {
            const double sq = sigma[static_cast<size_t>(a) * cols + b] * sigma[static_cast<size_t>(a) * cols + b];
            total += sq;
            if (a / kDsub == b / kDsub) {
                within += sq;
            }
        }
    }
    return (total - within) / total;
}

double top_energy_share(const Matrix& act, int keep) {
    std::vector<double> sigma;
    double trace = 0.0;
    symmetrize_moment(act, &sigma, &trace);
    std::vector<double> vectors;
    std::vector<double> eval;
    jacobi_eigen_n(&sigma, act.cols, &vectors, &eval);
    std::sort(eval.begin(), eval.end(), std::greater<double>());
    double top = 0.0;
    const int kept = std::min(keep, act.cols);
    for (int i = 0; i < kept; ++i) {
        top += eval[static_cast<size_t>(i)];
    }
    return top / trace;
}

double mean_conditional_ratio(const Matrix& act) {
    std::vector<double> raw;
    double trace = 0.0;
    symmetrize_moment(act, &raw, &trace);
    std::vector<double> sigma = raw;
    const int cols = act.cols;
    const double eps = 1e-8 * trace / cols;
    for (int i = 0; i < cols; ++i) {
        sigma[static_cast<size_t>(i) * cols + i] += eps;
    }
    std::vector<double> vectors;
    std::vector<double> eval;
    jacobi_eigen_n(&sigma, cols, &vectors, &eval);
    std::vector<double> prec(static_cast<size_t>(cols) * cols, 0.0);
    for (int i = 0; i < cols; ++i) {
        const double inv_s = 1.0 / std::max(eval[static_cast<size_t>(i)], eps);
        for (int r = 0; r < cols; ++r) {
            for (int c = 0; c < cols; ++c) {
                prec[static_cast<size_t>(r) * cols + c] +=
                    vectors[static_cast<size_t>(r) * cols + i] * inv_s * vectors[static_cast<size_t>(c) * cols + i];
            }
        }
    }
    double ratio_sum = 0.0;
    const int blocks = cols / kDsub;
    for (int b = 0; b < blocks; ++b) {
        double precision[4][4];
        double raw_f = 0.0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                precision[i][j] = prec[static_cast<size_t>(b * 4 + i) * cols + (b * 4 + j)];
                const double value = raw[static_cast<size_t>(b * 4 + i) * cols + (b * 4 + j)];
                raw_f += value * value;
            }
        }
        double q[4][4];
        double ev[4];
        jacobi_eigen_4(precision, q, ev);
        double cond_f = 0.0;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                double value = 0.0;
                for (int k = 0; k < 4; ++k) {
                    value += q[i][k] * (1.0 / std::max(ev[k], 1e-18)) * q[j][k];
                }
                cond_f += value * value;
            }
        }
        ratio_sum += std::sqrt(cond_f) / std::sqrt(std::max(raw_f, 1e-30));
    }
    return ratio_sum / blocks;
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

double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    if (n % 2 == 1) {
        return values[n / 2];
    }
    return 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

void print_block_spectrum(const Matrix& activation) {
    constexpr double kEpsRel = 1e-6;
    const int subspaces = activation.cols / kDsub;
    const int tokens = activation.rows;
    std::vector<double> conds;
    std::vector<double> centered_conds;
    std::vector<double> ratios;
    int truncated = 0;
    int truncated_floor = 0;
    int above = 0;
    std::printf("每块谱，校准样本 %d。白化仍用 E[xx^T]。减均值条件数只诊断。eps = 1e-6·tr/4。\n", tokens);
    std::printf("%4s %12s %12s %12s %12s %12s %4s %12s\n", "块", "λmax", "λmin", "条件数", "eps", "eps/λmin", "截断",
                "减均值条件数");
    for (int s = 0; s < subspaces; ++s) {
        double sigma[4][4] = {};
        double mean[4] = {};
        for (int t = 0; t < tokens; ++t) {
            double x[4];
            for (int d = 0; d < 4; ++d) {
                x[d] = activation.at(t, s * 4 + d);
                mean[d] += x[d];
            }
            for (int a = 0; a < 4; ++a) {
                for (int b = 0; b < 4; ++b) {
                    sigma[a][b] += x[a] * x[b];
                }
            }
        }
        for (int d = 0; d < 4; ++d) {
            mean[d] /= tokens;
        }
        double trace = 0.0;
        double cov[4][4];
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                sigma[a][b] /= tokens;
                cov[a][b] = sigma[a][b] - mean[a] * mean[b];
            }
            trace += sigma[a][a];
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
        double craw[4];
        double cq[4][4];
        jacobi_eigen_4(cov, cq, craw);
        double lambda_max = raw[0];
        double lambda_min = raw[0];
        double cmax = craw[0];
        double cmin = craw[0];
        for (int i = 1; i < 4; ++i) {
            lambda_max = std::max(lambda_max, raw[i]);
            lambda_min = std::min(lambda_min, raw[i]);
            cmax = std::max(cmax, craw[i]);
            cmin = std::min(cmin, craw[i]);
        }
        const double eps = kEpsRel * trace / 4.0;
        const bool cut = lambda_min < eps;
        if (cut) {
            truncated += 1;
        }
        if (lambda_min < kEigenFloor) {
            truncated_floor += 1;
        }
        const double ratio = lambda_min > 0.0 ? eps / lambda_min : 0.0;
        const double cond = lambda_min > 0.0 ? lambda_max / lambda_min : 0.0;
        const double ccond = cmin > 0.0 ? cmax / cmin : 0.0;
        if (lambda_min > 0.0) {
            conds.push_back(cond);
            ratios.push_back(ratio);
            if (cond > 1e3) {
                above += 1;
            }
        }
        if (cmin > 0.0) {
            centered_conds.push_back(ccond);
        }
        std::printf("%4d %12.3e %12.3e %12.3e %12.3e %12.3e %4s %12.3e\n", s, lambda_max, lambda_min, cond, eps, ratio,
                    cut ? "是" : "否", ccond);
    }
    auto max_of = [](const std::vector<double>& values) {
        return values.empty() ? 0.0 : *std::max_element(values.begin(), values.end());
    };
    auto min_of = [](const std::vector<double>& values) {
        return values.empty() ? 0.0 : *std::min_element(values.begin(), values.end());
    };
    std::printf("条件数最大 %.3e，中位 %.3e，最小 %.3e，大于 1e3 的块 %d / %d。\n", max_of(conds), median_of(conds),
                min_of(conds), above, subspaces);
    std::printf("减均值后条件数最大 %.3e，中位 %.3e，最小 %.3e。这一列不改白化。\n", max_of(centered_conds),
                median_of(centered_conds), min_of(centered_conds));
    std::printf("eps/λmin 最大 %.3e，中位 %.3e。相对 eps 截断 %d 块，绝对下限 %.1e 截断 %d 块。\n", max_of(ratios),
                median_of(ratios), truncated, kEigenFloor, truncated_floor);
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

Matrix load_matrix(const char* path) {
    std::FILE* file = std::fopen(path, "rb");
    if (file == nullptr) {
        std::fprintf(stderr, "无法读取 %s\n", path);
        std::exit(1);
    }
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    const bool header_ok = std::fread(&rows, sizeof(rows), 1, file) == 1 &&
                           std::fread(&cols, sizeof(cols), 1, file) == 1 && rows > 0 && cols > 0;
    Matrix matrix = header_ok ? make_matrix(rows, cols) : Matrix{};
    const bool body_ok = header_ok && std::fread(matrix.v.data(), sizeof(double), matrix.v.size(), file) == matrix.v.size();
    std::fclose(file);
    if (!body_ok) {
        std::fprintf(stderr, "读取 %s 失败\n", path);
        std::exit(1);
    }
    return matrix;
}

void save_state(const char* path, const Prepared& prep) {
    std::FILE* file = std::fopen(path, "wb");
    if (file == nullptr) {
        std::fprintf(stderr, "无法写出 %s\n", path);
        std::exit(1);
    }
    const std::int32_t rows = static_cast<std::int32_t>(prep.scale.size());
    const std::int32_t subspaces = static_cast<std::int32_t>(prep.bases.size());
    const std::int32_t whiten = prep.whiten ? 1 : 0;
    const std::int32_t cols = prep.work.cols;
    std::fwrite(&rows, sizeof(rows), 1, file);
    std::fwrite(&subspaces, sizeof(subspaces), 1, file);
    std::fwrite(&whiten, sizeof(whiten), 1, file);
    std::fwrite(&cols, sizeof(cols), 1, file);
    std::fwrite(prep.scale.data(), sizeof(double), prep.scale.size(), file);
    for (const SubspaceBasis& basis : prep.bases) {
        std::fwrite(basis.q, sizeof(double), 16, file);
        std::fwrite(basis.sqrt_scale, sizeof(double), 4, file);
        std::fwrite(basis.inv_sqrt_scale, sizeof(double), 4, file);
    }
    std::fclose(file);
}

Prepared load_state(const char* path) {
    std::FILE* file = std::fopen(path, "rb");
    if (file == nullptr) {
        std::fprintf(stderr, "无法读取 %s\n", path);
        std::exit(1);
    }
    std::int32_t rows = 0;
    std::int32_t subspaces = 0;
    std::int32_t whiten = 0;
    std::int32_t cols = 0;
    bool ok = std::fread(&rows, sizeof(rows), 1, file) == 1 &&
              std::fread(&subspaces, sizeof(subspaces), 1, file) == 1 &&
              std::fread(&whiten, sizeof(whiten), 1, file) == 1 &&
              std::fread(&cols, sizeof(cols), 1, file) == 1 && rows > 0 && cols > 0 && subspaces > 0;
    Prepared prep;
    if (ok) {
        prep.whiten = whiten != 0;
        prep.scale.resize(static_cast<size_t>(rows));
        prep.bases.resize(static_cast<size_t>(subspaces));
        prep.work = make_matrix(rows, cols);
        ok = std::fread(prep.scale.data(), sizeof(double), prep.scale.size(), file) == prep.scale.size();
    }
    for (SubspaceBasis& basis : prep.bases) {
        ok = ok && std::fread(basis.q, sizeof(double), 16, file) == 16 &&
             std::fread(basis.sqrt_scale, sizeof(double), 4, file) == 4 &&
             std::fread(basis.inv_sqrt_scale, sizeof(double), 4, file) == 4;
    }
    std::fclose(file);
    if (!ok) {
        std::fprintf(stderr, "读取 %s 失败\n", path);
        std::exit(1);
    }
    return prep;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc == 6 && std::strcmp(argv[1], "--whiten") == 0) {
        const Matrix weight = load_matrix(argv[2]);
        const Matrix activation = load_matrix(argv[3]);
        const bool row_norm = std::strcmp(argv[5], "rms") == 0;
        const Prepared prep = prepare_work(weight, activation, row_norm, true, false);
        const std::string dir = argv[4];
        dump_matrix((dir + "/Wp.bin").c_str(), prep.work);
        save_state((dir + "/state.bin").c_str(), prep);
        return 0;
    }
    if (argc == 5 && std::strcmp(argv[1], "--unwhiten") == 0) {
        const Matrix coded = load_matrix(argv[2]);
        const Prepared prep = load_state(argv[3]);
        dump_matrix(argv[4], restore_work(coded, prep));
        return 0;
    }

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
    print_block_spectrum(activation);
    if (!(whiten.roundtrip < 1e-12)) {
        std::fprintf(stderr, "白化后反白化没有恢复到 1e-12，0.4642 不能当原始空间误差\n");
        return 1;
    }
    if (!(whiten.direction < 1e-8)) {
        std::fprintf(stderr, "量化前的变换不是 Σ^{1/2}\n");
        return 1;
    }

    const Report raw = run_case(weight, activation, false, false, kCentroids, false);
    const Report rows = run_case(weight, activation, true, false, kCentroids, false);
    Matrix axis_recon;
    const Report white = run_case(weight, activation, false, true, kCentroids, false, &axis_recon);
    const Report both = run_case(weight, activation, true, true, kCentroids, false);
    const Report diag = run_case(weight, activation, false, true, kCentroids, true);

    std::printf("单层 PQ 对照，行 %d，列 %d，子空间维 %d，每子空间质心 %d。\n", kRows, kCols, kDsub, kCentroids);
    std::printf("这是 128×64 算法自检，使用自制 k-means，不是 Faiss。不能当作 S3 或全尺寸结论。\n");
    std::printf("%-16s %14s %14s\n", "路径", "小行相对误差", "输出 MSE");
    std::printf("%-16s %14.4f %14.4f\n", "原样", raw.weight_rel_small, raw.output_mse);
    std::printf("%-16s %14.4f %14.4f\n", "行 RMS", rows.weight_rel_small, rows.output_mse);
    std::printf("%-16s %14.4f %14.4f\n", "子空间白化", white.weight_rel_small, white.output_mse);
    std::printf("%-16s %14.4f %14.4f\n", "两者都用", both.weight_rel_small, both.output_mse);
    std::printf("%-16s %14.4f %14.4f\n", "对角白化", diag.weight_rel_small, diag.output_mse);
    auto print_split = [](const char* name, const Report& item) {
        const double scale = std::max(std::fabs(item.output_mse), 1e-30);
        std::printf("%s 输出误差分解：块内 %.4f（%.1f%%），交叉项 %.4f（%.1f%%）。样本内，n=32。\n", name,
                    item.within_mse, 100.0 * item.within_mse / scale, item.cross_mse, 100.0 * item.cross_mse / scale);
    };
    print_split("原样", raw);
    print_split("子空间白化", white);
    print_split("对角白化", diag);
    const double split_gap = std::fabs((white.within_mse + white.cross_mse) - white.output_mse);
    if (split_gap > 1e-8 * std::max(white.output_mse, 1e-30)) {
        std::fprintf(stderr, "块内部分加交叉项没有回到输出 MSE\n");
        return 1;
    }

    if (!(rows.weight_rel_small < raw.weight_rel_small)) {
        std::fprintf(stderr, "行 RMS 没有降低小行的相对误差\n");
        return 1;
    }
    if (!(white.output_mse < raw.output_mse)) {
        std::fprintf(stderr, "子空间白化没有降低输出 MSE\n");
        return 1;
    }
    std::printf("对照成立：行 RMS 降低小行相对误差，子空间白化降低输出 MSE。\n");
    std::printf("上面的条件数和 0.4630、0.4642 来自坐标独立的合成激活，不是 Llama 3 8B。\n");

    std::mt19937 corr_rng(29);
    const Matrix correlated = block_correlated_activation(kTokens, kCols, &corr_rng);
    const Report corr_white = run_case(weight, correlated, false, true, kCentroids, false);
    const Report corr_diag = run_case(weight, correlated, false, true, kCentroids, true);
    std::printf("阳性对照：每个 4 维块先给潜变量不同方差，再乘随机正交阵。样本内，n=%d。\n", kTokens);
    std::printf("%-16s %14.4f %14.4f\n", "子空间白化", corr_white.weight_rel_small, corr_white.output_mse);
    std::printf("%-16s %14.4f %14.4f\n", "对角白化", corr_diag.weight_rel_small, corr_diag.output_mse);
    print_split("阳性对照 子空间白化", corr_white);
    print_split("阳性对照 对角白化", corr_diag);
    if (!(corr_white.output_mse < corr_diag.output_mse * 0.8)) {
        std::fprintf(stderr, "块内相关时子空间白化没有明显优于对角白化，问题在脚本或 pq_layer\n");
        return 1;
    }
    const double white_cross = corr_white.cross_mse / std::max(std::fabs(corr_white.output_mse), 1e-30);
    const double diag_cross = corr_diag.cross_mse / std::max(std::fabs(corr_diag.output_mse), 1e-30);
    if (!(std::fabs(white_cross - diag_cross) > 0.02)) {
        std::fprintf(stderr, "块内相关时两条路径的交叉项占比没有分开，问题在脚本或 pq_layer\n");
        return 1;
    }
    std::printf("阳性对照成立：块内相关时，子空间白化优于对角白化。\n");
    std::printf("阳性对照的块与块独立。交叉项占比的差别多半是 32 条样本的噪声被更大的误差放大，不是跨块结构。\n");

    std::mt19937 cross_rng(31);
    const Matrix crossed = cross_block_activation(kTokens, kCols, &cross_rng, true);
    {
        const int tokens = crossed.rows;
        const int cols = crossed.cols;
        std::vector<double> sigma(static_cast<size_t>(cols) * cols, 0.0);
        for (int t = 0; t < tokens; ++t) {
            for (int a = 0; a < cols; ++a) {
                for (int b = 0; b < cols; ++b) {
                    sigma[static_cast<size_t>(a) * cols + b] += crossed.at(t, a) * crossed.at(t, b);
                }
            }
        }
        double total = 0.0;
        double within = 0.0;
        for (int a = 0; a < cols; ++a) {
            for (int b = 0; b < cols; ++b) {
                const double value = sigma[static_cast<size_t>(a) * cols + b] / tokens;
                const double sq = value * value;
                total += sq;
                if (a / kDsub == b / kDsub) {
                    within += sq;
                }
            }
        }
        Matrix zero = make_matrix(weight.rows, weight.cols);
        Report plain;
        fill_error_split(weight, zero, crossed, &plain);
        const double plain_share = plain.cross_mse / std::max(std::fabs(plain.within_mse + plain.cross_mse), 1e-30);
        std::printf("反向对照的 Σ：跨块能量 %.1f%%。不量化时，用原权重算出的交叉项占比 %.1f%%。\n",
                    100.0 * (total - within) / total, 100.0 * plain_share);
    }
    Matrix one_recon;
    const Report cross_white = run_case(weight, crossed, false, true, kCentroids, false, &one_recon);
    const double cross_share = cross_white.cross_mse / std::max(std::fabs(cross_white.output_mse), 1e-30);
    std::printf("反向对照：一个大潜变量、其余坐标很小，再乘一个全局正交阵。样本内，n=%d。\n", kTokens);
    std::printf("%-16s %14.4f %14.4f\n", "子空间白化", cross_white.weight_rel_small, cross_white.output_mse);
    print_split("反向对照 子空间白化", cross_white);
    if (!(std::fabs(cross_share) > std::fabs(white_cross) + 0.10)) {
        std::fprintf(stderr, "全局混合后块白化的交叉项占比没有明显变大，这项诊断还不能用来决定子空间大小\n");
        return 1;
    }
    std::printf("这是一个大潜变量的特例。十六个大潜变量时交叉项仍会接近零，不能用这个占比决定子空间大小。\n");
    const int seeds[5] = {7, 8, 9, 10, 11};
    int negative_seeds = 0;
    std::printf("一个大潜变量，同一批 X，5 个 k-means 种子的交叉项占比：");
    for (int seed : seeds) {
        const Report seeded = run_case(weight, crossed, false, true, kCentroids, false, nullptr, seed);
        const double share = seeded.cross_mse / std::max(std::fabs(seeded.output_mse), 1e-30);
        std::printf(" %d:%.1f%%", seed, 100.0 * share);
        if (share < 0.0) {
            negative_seeds += 1;
        }
    }
    std::printf("。负号 %d / 5。\n", negative_seeds);

    constexpr int kHeld = 256;
    constexpr int kSeedCount = 5;
    std::mt19937 shared_rng(31);
    std::vector<double> shared_q;
    random_orthogonal_n(&shared_rng, kCols, &shared_q);
    Matrix calib = make_matrix(kHeld, kCols);
    Matrix eval_act = make_matrix(kHeld, kCols);
    std::mt19937 calib_rng(101);
    std::mt19937 eval_rng(202);
    fill_latent_activation(&calib, shared_q, &calib_rng, true);
    fill_latent_activation(&eval_act, shared_q, &eval_rng, true);
    std::printf("一个大潜变量，独立校准 %d 条、评估 %d 条。最大特征值占迹的 %.1f%%，其余占 %.1f%%。条件度量平均 %.3f。\n",
                kHeld, kHeld, 100.0 * top_energy_share(calib, 1), 100.0 * (1.0 - top_energy_share(calib, 1)),
                mean_conditional_ratio(calib));
    Matrix block_recons[kSeedCount];
    double block_eval[kSeedCount];
    double block_in[kSeedCount];
    for (int i = 0; i < kSeedCount; ++i) {
        run_case(weight, calib, false, true, kCentroids, false, &block_recons[i], seeds[i], 8);
        block_eval[i] = held_out_mse(weight, block_recons[i], eval_act);
        block_in[i] = held_out_mse(weight, block_recons[i], calib);
        std::printf("  种子 %d  块白化 样本外 %.4f  样本内 %.4f\n", seeds[i], block_eval[i], block_in[i]);
    }
    auto score_full = [&](double eps_rel, int keep, int iterations, double* eval_out, double* in_out) {
        for (int i = 0; i < kSeedCount; ++i) {
            const Matrix full_recon = full_whiten_recon(weight, calib, kCentroids, seeds[i], iterations, eps_rel, keep);
            eval_out[i] = held_out_mse(weight, full_recon, eval_act);
            in_out[i] = held_out_mse(weight, full_recon, calib);
        }
    };
    double full_eval[kSeedCount];
    double full_in[kSeedCount];
    score_full(1e-6, 0, 8, full_eval, full_in);
    for (int i = 0; i < kSeedCount; ++i) {
        std::printf("  种子 %d  完整白化 样本外 %.4f  样本内 %.4f\n", seeds[i], full_eval[i], full_in[i]);
    }
    std::printf("秩 1，ε=1e-6，8 次迭代。样本外 ");
    print_mean_std("块白化", block_eval, kSeedCount);
    std::printf("，");
    print_mean_std("完整白化", full_eval, kSeedCount);
    std::printf("。样本内 ");
    print_mean_std("块白化", block_in, kSeedCount);
    std::printf("，");
    print_mean_std("完整白化", full_in, kSeedCount);
    std::printf("。标准差按 5 个种子、除以 4。\n");
    const double eps_scan[] = {1e-4, 1e-3, 1e-2, 1e-1};
    std::printf("秩 1，扫描 ε，样本外完整白化：\n");
    for (double eps_rel : eps_scan) {
        double scanned_eval[kSeedCount];
        double scanned_in[kSeedCount];
        score_full(eps_rel, 0, 8, scanned_eval, scanned_in);
        std::printf("  ε=%.0e  样本外 ", eps_rel);
        print_mean_std("完整白化", scanned_eval, kSeedCount);
        std::printf("，样本内 ");
        print_mean_std("完整白化", scanned_in, kSeedCount);
        std::printf("\n");
    }
    const int keep_scan[] = {1, 4, 8, 16};
    std::printf("秩 1，弱特征值抬到第 k 个，样本外完整白化：\n");
    for (int keep : keep_scan) {
        double scanned_eval[kSeedCount];
        double scanned_in[kSeedCount];
        score_full(1e-6, keep, 8, scanned_eval, scanned_in);
        std::printf("  保留 %d  样本外 ", keep);
        print_mean_std("完整白化", scanned_eval, kSeedCount);
        std::printf("，样本内 ");
        print_mean_std("完整白化", scanned_in, kSeedCount);
        std::printf("\n");
    }
    Matrix block_long;
    run_case(weight, calib, false, true, kCentroids, false, &block_long, 7, 25);
    const Matrix full_long = full_whiten_recon(weight, calib, kCentroids, 7, 25);
    std::printf("同一划分、种子 7、25 次迭代：样本外块白化 %.4f，完整白化 %.4f；样本内块白化 %.4f，完整白化 %.4f。\n",
                held_out_mse(weight, block_long, eval_act), held_out_mse(weight, full_long, eval_act),
                held_out_mse(weight, block_long, calib), held_out_mse(weight, full_long, calib));
    const int latent_counts[] = {4, 6, 8};
    for (int n_latent : latent_counts) {
        std::vector<double> scales(static_cast<size_t>(kCols), 0.05);
        for (int c = 0; c < n_latent; ++c) {
            scales[static_cast<size_t>(c)] = 8.0 * std::pow(0.5, c);
        }
        std::mt19937 mid_rng(41 + n_latent);
        std::vector<double> mid_q;
        random_orthogonal_n(&mid_rng, kCols, &mid_q);
        Matrix mid_calib = make_matrix(kHeld, kCols);
        Matrix mid_eval = make_matrix(kHeld, kCols);
        std::mt19937 mid_calib_rng(111);
        std::mt19937 mid_eval_rng(222);
        fill_scaled_activation(&mid_calib, mid_q, scales, &mid_calib_rng);
        fill_scaled_activation(&mid_eval, mid_q, scales, &mid_eval_rng);
        double mid_block_eval[kSeedCount];
        double mid_block_in[kSeedCount];
        double mid_full_eval[kSeedCount];
        double mid_full_in[kSeedCount];
        for (int i = 0; i < kSeedCount; ++i) {
            Matrix mid_block;
            run_case(weight, mid_calib, false, true, kCentroids, false, &mid_block, seeds[i], 8);
            const Matrix mid_full = full_whiten_recon(weight, mid_calib, kCentroids, seeds[i], 8);
            mid_block_eval[i] = held_out_mse(weight, mid_block, mid_eval);
            mid_block_in[i] = held_out_mse(weight, mid_block, mid_calib);
            mid_full_eval[i] = held_out_mse(weight, mid_full, mid_eval);
            mid_full_in[i] = held_out_mse(weight, mid_full, mid_calib);
        }
        std::printf("中间情形：%d 个潜变量，尺度从 8 逐级减半，其余 0.05，再全局旋转。跨块能量 %.3f，前 %d 个特征值占迹的 %.1f%%，条件度量平均 %.3f。\n",
                    n_latent, off_block_energy(mid_calib), n_latent, 100.0 * top_energy_share(mid_calib, n_latent),
                    mean_conditional_ratio(mid_calib));
        std::printf("  样本外 ");
        print_mean_std("块白化", mid_block_eval, kSeedCount);
        std::printf("，");
        print_mean_std("完整白化", mid_full_eval, kSeedCount);
        std::printf("。样本内 ");
        print_mean_std("块白化", mid_block_in, kSeedCount);
        std::printf("，");
        print_mean_std("完整白化", mid_full_in, kSeedCount);
        std::printf("。\n");
    }
    std::mt19937 many_rng(31);
    const Matrix many = cross_block_activation(kTokens, kCols, &many_rng, false);
    Matrix many_recon;
    const Report many_white = run_case(weight, many, false, true, kCentroids, false, &many_recon);
    const double many_share = many_white.cross_mse / std::max(std::fabs(many_white.output_mse), 1e-30);
    std::printf("十六个大潜变量再全局旋转：块白化输出 MSE %.4f，交叉项占比 %.1f%%。\n", many_white.output_mse,
                100.0 * many_share);
    if (argc == 3 && std::strcmp(argv[1], "--dump-cross") == 0) {
        const std::string dir = argv[2];
        const std::string mkdir_cmd = "mkdir -p " + dir + "/axis " + dir + "/one " + dir + "/many";
        if (std::system(mkdir_cmd.c_str()) != 0) {
            std::fprintf(stderr, "无法建立 %s\n", dir.c_str());
            return 1;
        }
        dump_matrix((dir + "/axis/W.bin").c_str(), weight);
        dump_matrix((dir + "/axis/X.bin").c_str(), activation);
        dump_matrix((dir + "/axis/recon.bin").c_str(), axis_recon);
        dump_matrix((dir + "/one/W.bin").c_str(), weight);
        dump_matrix((dir + "/one/X.bin").c_str(), crossed);
        dump_matrix((dir + "/one/recon.bin").c_str(), one_recon);
        dump_matrix((dir + "/many/W.bin").c_str(), weight);
        dump_matrix((dir + "/many/X.bin").c_str(), many);
        dump_matrix((dir + "/many/recon.bin").c_str(), many_recon);
        std::printf("已写出三组 W、X、重建：%s\n", dir.c_str());
    }
    return 0;
}
