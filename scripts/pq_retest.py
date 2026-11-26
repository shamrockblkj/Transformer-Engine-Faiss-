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


def oos_check(W, X_cal, X_eval, d_sub=4, k=8, seeds=range(5),
              eps_list=(1e-6, 1e-4, 1e-3, 1e-2, 1e-1), weak_tau=1e-3):
    rows = []
    for eps in eps_list:
        Shb, Sihb, _ = block_whiteners(X_cal, d_sub, eps)
        Shf, Sihf, w, V = full_whiteners(X_cal, eps)
        weak = w < weak_tau * w.max()
        Se = X_eval.T @ X_eval / len(X_eval)
        # 弱方向上 评估能量 / 校准能量：远大于 1 说明 Σ 弱方向估计不稳
        rho = np.diag(V.T @ Se @ V)[weak] / np.maximum(w[weak], 1e-300)
        Pw = V[:, weak] @ V[:, weak].T
        for s in seeds:
            Wb = apply_blocks(pq_reconstruct(apply_blocks(W, Shb, d_sub), d_sub, k, s), Sihb, d_sub)
            Wf = pq_reconstruct(W @ Shf, d_sub, k, s) @ Sihf
            for name, Wh in (("block", Wb), ("full", Wf)):
                dW = Wh - W
                tot = np.einsum("ij,jk,ik->", dW, Se, dW)
                wk = np.einsum("ij,jk,ik->", dW @ Pw, Se, dW @ Pw)
                energy = ((dW @ V) ** 2).sum(axis=0)
                rho_all = np.diag(V.T @ Se @ V) / np.maximum(w, 1e-300)
                weak_energy = float(energy[weak].sum()) if weak.any() else 0.0
                rho_weighted = float((energy[weak] * rho_all[weak]).sum() / weak_energy) if weak_energy > 0.0 else np.nan
                rows.append(dict(eps=eps, seed=s, path=name,
                    mse_cal=out_mse(W, Wh, X_cal), mse_eval=out_mse(W, Wh, X_eval),
                    weak_share_eval=wk / tot,           # 只取弱子空间那部分，不含它和强方向的交叉
                    n_weak=int(weak.sum()),
                    rho_median=float(np.median(rho)) if weak.any() else np.nan,
                    rho_max=float(np.max(rho)) if weak.any() else np.nan,
                    rho_weighted=rho_weighted))
    return rows


def raw_rows(W, X_cal, X_eval, d_sub=4, k=8, seeds=range(5)):
    rows = []
    for seed in seeds:
        hat = pq_reconstruct(W, d_sub, k, seed)
        rows.append(dict(seed=seed, mse_cal=out_mse(W, hat, X_cal), mse_eval=out_mse(W, hat, X_eval)))
    return rows


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1))


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["eps"], row["path"]), []).append(row)
    stats = {}
    for key, group in grouped.items():
        cal_mean, cal_std = _mean_std([item["mse_cal"] for item in group])
        eval_mean, eval_std = _mean_std([item["mse_eval"] for item in group])
        weak_mean, weak_std = _mean_std([item["weak_share_eval"] for item in group])
        weighted_mean, weighted_std = _mean_std([item["rho_weighted"] for item in group])
        stats[key] = dict(
            mse_cal=cal_mean, mse_cal_std=cal_std, mse_eval=eval_mean, mse_eval_std=eval_std,
            weak_share=weak_mean, weak_share_std=weak_std,
            rho_median=float(group[0]["rho_median"]), rho_max=float(group[0]["rho_max"]),
            rho_weighted=weighted_mean, rho_weighted_std=weighted_std, n_weak=int(group[0]["n_weak"]),
        )
    return stats


def select_eps(W, X_cal, eps_list, d_sub=4, k=8, seeds=range(5), fit_count=192):
    """Σ 用前 fit_count 条估计，ε 用剩下的校准条挑选。评估集不进入这个函数。"""
    fit = X_cal[:fit_count]
    held = X_cal[fit_count:]
    chosen = []
    for eps in eps_list:
        Shb, Sihb, _ = block_whiteners(fit, d_sub, eps)
        Shf, Sihf, _w, _V = full_whiteners(fit, eps)
        block_scores = []
        full_scores = []
        for seed in seeds:
            block = apply_blocks(pq_reconstruct(apply_blocks(W, Shb, d_sub), d_sub, k, seed), Sihb, d_sub)
            full = pq_reconstruct(W @ Shf, d_sub, k, seed) @ Sihf
            block_scores.append(out_mse(W, block, held))
            full_scores.append(out_mse(W, full, held))
        block_mean, block_std = _mean_std(block_scores)
        full_mean, full_std = _mean_std(full_scores)
        chosen.append(dict(eps=eps, block=block_mean, block_std=block_std, full=full_mean, full_std=full_std))
    best_block = min(chosen, key=lambda item: item["block"])
    best_full = min(chosen, key=lambda item: item["full"])
    return chosen, best_block, best_full


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


def _turns_up(values):
    lowest = int(np.argmin(values))
    return lowest not in (0, len(values) - 1)


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


def run_case(title, weight, calib, eval_act, eps_list, seeds):
    print(f"\n{title}")
    print("Σ 的诊断表用全部 256 条校准。ε 的挑选只用前 192 条估计、后 64 条打分，评估集不进入挑选。")
    rows = oos_check(weight, calib, eval_act, seeds=seeds, eps_list=eps_list)
    stats = summarize(rows)
    raw = raw_rows(weight, calib, eval_act, seeds=seeds)
    raw_cal, raw_cal_std = _mean_std([item["mse_cal"] for item in raw])
    raw_eval, raw_eval_std = _mean_std([item["mse_eval"] for item in raw])
    print(f"{'ε':>8} {'路径':<6} {'样本内':>18} {'样本外':>18} {'弱方向占比':>18} {'rho中位':>8} {'rho最大':>8} {'rho加权':>8}")
    for eps in eps_list:
        for path in ("block", "full"):
            item = stats[(eps, path)]
            print(
                f"{eps:8.0e} {path:<6} {_fmt(item['mse_cal'], item['mse_cal_std']):>18} "
                f"{_fmt(item['mse_eval'], item['mse_eval_std']):>18} "
                f"{_fmt(item['weak_share'], item['weak_share_std']):>18} "
                f"{item['rho_median']:8.3f} {item['rho_max']:8.3f} {item['rho_weighted']:8.3f}"
            )
    print(f"raw 不白化  样本内 {_fmt(raw_cal, raw_cal_std)}  样本外 {_fmt(raw_eval, raw_eval_std)}")
    held_rows, best_block, best_full = select_eps(weight, calib, eps_list, seeds=seeds)
    print("两条路径各自在后 64 条上选 ε：")
    for item in held_rows:
        print(f"  ε={item['eps']:.0e}  块白化 {_fmt(item['block'], item['block_std'])}  完整白化 {_fmt(item['full'], item['full_std'])}")
    block_pick = stats[(best_block["eps"], "block")]
    full_pick = stats[(best_full["eps"], "full")]
    print(
        f"块白化选中 ε={best_block['eps']:.0e}，完整白化选中 ε={best_full['eps']:.0e}。"
        "下面的样本外数字是用全部 256 条重新估计 Σ 之后，在评估集上算的。"
    )
    print(f"  块白化样本外 {_fmt(block_pick['mse_eval'], block_pick['mse_eval_std'])}")
    print(f"  完整白化样本外 {_fmt(full_pick['mse_eval'], full_pick['mse_eval_std'])}")
    full_held = [item["full"] for item in held_rows]
    full_eval = [stats[(eps, "full")]["mse_eval"] for eps in eps_list]
    if best_full["eps"] == eps_list[-1]:
        print("完整白化在 64 条上的最低点落在网格右端，样本外曲线也要看它有没有回头。")
    if _turns_up(full_eval):
        turn = eps_list[int(np.argmin(full_eval))]
        print(f"完整白化的样本外 MSE 先降后升，最低在 ε={turn:.0e}。")
    elif full_eval[-1] < full_eval[-2]:
        print("完整白化的样本外 MSE 到网格右端还在降。")
    if _turns_up(full_held):
        turn = eps_list[int(np.argmin(full_held))]
        print(f"64 条上完整白化先降后升，最低在 ε={turn:.0e}。")
    elif full_held[-1] < full_held[-2]:
        print("64 条上完整白化到网格右端还在降。")
    profile = floor_report(calib, 1e-1)
    print(
        f"ε=1e-1：ε/λmax={profile['eps_over_lam']:.4f}，弱方向占迹 {profile['weak_trace_share']:.4f}，"
        f"ε 是弱特征值均值的 {profile['eps_over_weak_mean']:.1f} 倍，抬平后条件数 {profile['cond_floored']:.1f}，"
        f"仍高于 ε 的特征值 {profile['n_above_eps']} 个。"
    )
    base = stats[(eps_list[0], "full")]
    print(
        f"默认 ε 的完整白化：rho 中位 {base['rho_median']:.3f}，最大 {base['rho_max']:.3f}，"
        f"按残差能量加权 {_fmt(base['rho_weighted'], base['rho_weighted_std'])}。"
    )
    return dict(stats=stats, raw_eval=raw_eval, raw_eval_std=raw_eval_std, best_block=best_block, best_full=best_full, profile=profile)


def main() -> int:
    cols = 64
    weight = small_row_weights(128, cols, np.random.default_rng(11))
    eps_list = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0)
    seeds = range(5)
    print("numpy 种子，和 C++ 的 0.0216 / 0.0230 不是同一套实现。8 次迭代，5 个种子，标准差除以 4。")
    orthogonal, _ = np.linalg.qr(np.random.default_rng(31).standard_normal((cols, cols)))
    scales = latent_scales(cols, 1)
    run_case(
        "秩 1。权重种子 11，正交阵 31，校准 101，评估 202。",
        weight,
        scaled_pair(256, cols, scales, orthogonal, np.random.default_rng(101)),
        scaled_pair(256, cols, scales, orthogonal, np.random.default_rng(202)),
        eps_list,
        seeds,
    )
    for n_latent, q_seed in ((4, 45), (6, 47), (8, 49)):
        graded, _ = np.linalg.qr(np.random.default_rng(q_seed).standard_normal((cols, cols)))
        graded_scales = latent_scales(cols, n_latent)
        run_case(
            f"{n_latent} 个潜变量，尺度从 8 逐级减半，其余 0.05。正交阵种子 {q_seed}，校准 111，评估 222。",
            weight,
            scaled_pair(256, cols, graded_scales, graded, np.random.default_rng(111)),
            scaled_pair(256, cols, graded_scales, graded, np.random.default_rng(222)),
            eps_list,
            seeds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
