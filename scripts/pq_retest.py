#!/usr/bin/env python3
"""样本外放大检查。

ε 只在校准集内部选：256 条再切成 192/64，在那 64 条上选。评估集不参与挑选。
大 ε 会把完整白化压向不白化，所以同一批种子上同时报 raw。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from whiten_roundtrip_check_v2 import block_stats


def block_whiteners(activation, d_sub, eps_rel):
    _blocks, forward, inverse, conds = block_stats(np.asarray(activation, dtype=np.float64), d_sub, eps_rel)
    return forward, inverse, conds


def apply_blocks(weight, factors, d_sub):
    parts = []
    for index, factor in enumerate(factors):
        start = index * d_sub
        parts.append(np.asarray(weight[:, start : start + d_sub], dtype=np.float64) @ factor)
    return np.concatenate(parts, axis=1)


def _kmeans(points, k, rng, iterations=8):
    count = len(points)
    centers = points[rng.integers(0, count, size=k)].copy()
    assign = np.zeros(count, dtype=np.int64)
    for _ in range(iterations):
        distance = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = distance.argmin(axis=1)
        for center in range(k):
            chosen = np.flatnonzero(assign == center)
            if chosen.size == 0:
                centers[center] = points[int(rng.integers(0, count))]
            else:
                centers[center] = points[chosen].mean(axis=0)
    distance = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    assign = distance.argmin(axis=1)
    return centers[assign]


def pq_reconstruct(work, d_sub, k, seed, iterations=8):
    matrix = np.asarray(work, dtype=np.float64)
    rng = np.random.default_rng(seed)
    decoded = np.empty_like(matrix)
    for start in range(0, matrix.shape[1], d_sub):
        decoded[:, start : start + d_sub] = _kmeans(matrix[:, start : start + d_sub], k, rng, iterations)
    return decoded


def out_mse(weight, hat, activation):
    residual = np.asarray(activation, dtype=np.float64) @ (np.asarray(hat, dtype=np.float64) - np.asarray(weight, dtype=np.float64)).T
    return float(np.mean(residual**2))


def full_whiteners(X, eps_rel):
    S = X.T.astype(np.float64) @ X / len(X)
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, 0, None)
    w2 = w + eps_rel * w.sum() / len(w)
    return (V * w2**0.5) @ V.T, (V * w2**-0.5) @ V.T, w, V



EPS_GRID = (1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0)


def recon(path, W, Xfit, eps, d_sub, k, seed):
    if path == "block":
        Sh, Sih, _ = block_whiteners(Xfit, d_sub, eps)
        Wq = pq_reconstruct(apply_blocks(W, Sh, d_sub), d_sub, k, seed)
        return apply_blocks(Wq, Sih, d_sub)
    if path == "full":
        Shf, Sihf, _, _ = full_whiteners(Xfit, eps)
        return pq_reconstruct(W @ Shf, d_sub, k, seed) @ Sihf
    if path == "diag":
        scale = np.sqrt(np.maximum(np.mean(np.asarray(Xfit, dtype=np.float64) ** 2, axis=0), 1e-12))
        return pq_reconstruct(W * scale, d_sub, k, seed) / scale
    raise ValueError(path)


def select_eps(path, W, X_cal, n_fit=192, grid=EPS_GRID, d_sub=4, k=8, seeds=range(5)):
    """只用校准集：前 n_fit 条估计 Σ，其余条选 ε。返回选中的 ε 和整条曲线。"""
    Xf, Xv = X_cal[:n_fit], X_cal[n_fit:]
    curve = []
    for eps in grid:
        scores = [out_mse(W, recon(path, W, Xf, eps, d_sub, k, s), Xv) for s in seeds]
        curve.append((eps, float(np.mean(scores)), float(np.std(scores, ddof=1))))
    index = int(np.argmin([item[1] for item in curve]))
    at_edge = index in (0, len(grid) - 1)  # 最优在网格端点：需要继续扩网格
    return grid[index], curve, at_edge


def rho_stats(W, Whs, X_cal, X_eval, eps, weak_tau=1e-3):
    """rho 沿校准集 Σ 的特征向量。加权先把各种子的贡献求平均。"""
    _, _, eigenvalues, vectors = full_whiteners(X_cal, eps)
    second = X_eval.T @ X_eval / len(X_eval)
    weak = eigenvalues < weak_tau * eigenvalues.max()
    moment = np.diag(vectors.T @ second @ vectors)
    rho = moment / np.maximum(eigenvalues, 1e-300)
    hats = Whs if isinstance(Whs, (list, tuple)) else [Whs]
    contrib = np.zeros(len(eigenvalues))
    per_seed = []
    for hat in hats:
        one = (((hat - W) @ vectors) ** 2).sum(axis=0) * moment
        contrib += one
        per_seed.append(float((one[weak] * rho[weak]).sum() / one[weak].sum()))
    contrib /= len(hats)
    weak_contrib = contrib[weak]
    return dict(
        rho_median=float(np.median(rho[weak])),
        rho_max=float(rho[weak].max()),
        rho_weighted=float((weak_contrib * rho[weak]).sum() / weak_contrib.sum()),
        rho_seed0=per_seed[0],
        rho_seed_std=float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
    )


def fair_compare(W, X_cal, X_eval, d_sub=4, k=8, seeds=range(5)):
    out = {}
    for path in ("block", "full"):
        eps, curve, edge = select_eps(path, W, X_cal, d_sub=d_sub, k=k, seeds=seeds)
        hats = [recon(path, W, X_cal, eps, d_sub, k, seed) for seed in seeds]
        scores = [out_mse(W, hat, X_eval) for hat in hats]
        out[path] = dict(
            eps=eps, at_edge=edge, curve=curve,
            oos_mean=float(np.mean(scores)), oos_std=float(np.std(scores, ddof=1)),
            **rho_stats(W, hats, X_cal, X_eval, eps),
        )
    raw = [out_mse(W, pq_reconstruct(W, d_sub, k, seed), X_eval) for seed in seeds]
    out["raw"] = dict(oos_mean=float(np.mean(raw)), oos_std=float(np.std(raw, ddof=1)))
    block_curve = out["block"]["curve"]
    if out["block"]["at_edge"] and block_curve[-1][1] < block_curve[-2][1]:
        diag = [out_mse(W, recon("diag", W, X_cal, 0.0, d_sub, k, seed), X_eval) for seed in seeds]
        out["diag"] = dict(oos_mean=float(np.mean(diag)), oos_std=float(np.std(diag, ddof=1)))
    return out


def floor_report(activation, eps_rel, weak_tau=1e-3):
    """这一档 ε 把哪些特征值抬平。弱方向是小于 weak_tau·λmax 的那些。"""
    second = activation.T.astype(np.float64) @ activation / len(activation)
    second = 0.5 * (second + second.T)
    eigenvalues = np.clip(np.linalg.eigvalsh(second), 0, None)
    largest = float(eigenvalues.max())
    trace = float(eigenvalues.sum())
    eps = eps_rel * trace / len(eigenvalues)
    weak = eigenvalues < weak_tau * largest
    floored = eigenvalues + eps
    weak_mean = float(eigenvalues[weak].mean()) if weak.any() else np.nan
    return dict(
        lam_max=largest, trace=trace, eps=eps, eps_over_lam=eps / largest,
        weak_trace_share=float(eigenvalues[weak].sum() / trace) if weak.any() else 0.0,
        weak_mean_over_lam=weak_mean / largest, eps_over_weak_mean=eps / weak_mean,
        cond_floored=float(floored.max() / floored.min()),
        n_weak=int(weak.sum()), n_above_eps=int(np.sum(eigenvalues > eps)),
    )


def scaled_pair(tokens, cols, scales, orthogonal, rng):
    latent = rng.standard_normal((tokens, cols)) * scales
    return latent @ orthogonal.T


def latent_scales(cols, n_latent):
    scales = np.full(cols, 0.05)
    for index in range(n_latent):
        scales[index] = 8.0 * 0.5**index
    return scales


def small_row_weights(rows, cols, rng):
    large = rng.standard_normal(cols)
    small = rng.standard_normal(cols)
    weight = np.empty((rows, cols))
    for row in range(rows):
        amp = 0.2 if row < rows // 4 else 6.0
        base = small if row < rows // 4 else large
        weight[row] = amp * base + rng.standard_normal(cols) * 0.05
    return weight


def _fmt(mean, std):
    return f"{mean:.4f} ± {std:.4f}"


def run_case(title, weight, calib, eval_act, seeds):
    print(f"\n{title}")
    print("ε 用前 192 条估计 Σ、后 64 条挑选。192 条估计 64×64 比 256 条更噪，选出的 ε 偏大，用回 256 条是偏保守的。")
    result = fair_compare(weight, calib, eval_act, seeds=seeds)
    for path in ("block", "full"):
        item = result[path]
        print(f"{path} 选中 ε={item['eps']:.0e}，端点={item['at_edge']}。64 条上的曲线：")
        for eps, mean, std in item["curve"]:
            print(f"  ε={eps:.0e}  {_fmt(mean, std)}")
        print(
            f"  评估集 {_fmt(item['oos_mean'], item['oos_std'])}。"
            f"rho 中位 {item['rho_median']:.3f}，最大 {item['rho_max']:.3f}，"
            f"五种子贡献平均后加权 {item['rho_weighted']:.3f}，各种子加权的标准差 {item['rho_seed_std']:.3f}。"
        )
        if item["at_edge"] and item["curve"][-1][1] < item["curve"][-2][1]:
            print("  曲线到网格右端还在降，ε 已经大到改变方法本身，这一档先不下结论。")
    block = result["block"]
    full = result["full"]
    lead = block["oos_mean"] - full["oos_mean"]
    spread = block["oos_std"] + full["oos_std"]
    print(f"raw 评估集 {_fmt(result['raw']['oos_mean'], result['raw']['oos_std'])}。")
    if "diag" in result:
        print(f"对角白化评估集 {_fmt(result['diag']['oos_mean'], result['diag']['oos_std'])}。")
    print(f"完整白化领先 {lead:.4f}，两边标准差之和 {spread:.4f}。{'领先超过种子波动。' if lead > spread else '领先没有超过两边标准差之和。'}")
    profile = floor_report(calib, full["eps"])
    print(
        f"完整白化选中的 ε={full['eps']:.0e}：ε/λmax={profile['eps_over_lam']:.4f}，"
        f"弱方向占迹 {profile['weak_trace_share']:.4f}，抬平后条件数 {profile['cond_floored']:.1f}，"
        f"仍高于 ε 的特征值 {profile['n_above_eps']} 个。"
    )
    return result


def main() -> int:
    cols = 64
    weight = small_row_weights(128, cols, np.random.default_rng(11))
    seeds = range(5)
    print("numpy 种子，和 C++ 的 0.0216 / 0.0230 不是同一套实现。8 次迭代，5 个种子，标准差除以 4。")
    orthogonal, _ = np.linalg.qr(np.random.default_rng(31).standard_normal((cols, cols)))
    scales = latent_scales(cols, 1)
    run_case(
        "秩 1。权重种子 11，正交阵 31，校准 101，评估 202。",
        weight,
        scaled_pair(256, cols, scales, orthogonal, np.random.default_rng(101)),
        scaled_pair(256, cols, scales, orthogonal, np.random.default_rng(202)),
        seeds,
    )
    for n_latent, q_seed in ((4, 45), (6, 47), (8, 49)):
        graded, _ = np.linalg.qr(np.random.default_rng(q_seed).standard_normal((cols, cols)))
        run_case(
            f"{n_latent} 个潜变量，尺度从 8 逐级减半，其余 0.05。正交阵种子 {q_seed}，校准 111，评估 222。",
            weight,
            scaled_pair(256, cols, latent_scales(cols, n_latent), graded, np.random.default_rng(111)),
            scaled_pair(256, cols, latent_scales(cols, n_latent), graded, np.random.default_rng(222)),
            seeds,
        )
    graded, _ = np.linalg.qr(np.random.default_rng(49).standard_normal((cols, cols)))
    run_case(
        "8 个潜变量再跑 20 个种子。五种子那组的块白化标准差最大，领先只有标准差之和的 1.4 倍。",
        weight,
        scaled_pair(256, cols, latent_scales(cols, 8), graded, np.random.default_rng(111)),
        scaled_pair(256, cols, latent_scales(cols, 8), graded, np.random.default_rng(222)),
        range(20),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
