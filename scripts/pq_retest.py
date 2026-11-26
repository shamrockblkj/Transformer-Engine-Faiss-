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
                rows.append(dict(eps=eps, seed=s, path=name,
                    mse_cal=out_mse(W, Wh, X_cal), mse_eval=out_mse(W, Wh, X_eval),
                    weak_share_eval=wk / tot,           # 只取弱子空间那部分，不含它和强方向的交叉
                    n_weak=int(weak.sum()), rho_median=float(np.median(rho)) if weak.any() else np.nan))
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
        stats[key] = dict(
            mse_cal=cal_mean, mse_cal_std=cal_std, mse_eval=eval_mean, mse_eval_std=eval_std,
            weak_share=weak_mean, weak_share_std=weak_std,
            rho_median=float(group[0]["rho_median"]), n_weak=int(group[0]["n_weak"]),
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
    best = min(chosen, key=lambda item: item["full"])
    return chosen, best


def rank1_pair(tokens, cols, orthogonal, rng):
    scales = np.full(cols, 0.05)
    scales[0] = 8.0
    latent = rng.standard_normal((tokens, cols)) * scales
    return latent @ orthogonal.T


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


def classify(stats, eps_list, raw_eval):
    base = eps_list[0]
    full_in = stats[(base, "full")]["mse_cal"]
    block_in = stats[(base, "block")]["mse_cal"]
    full_in_std = stats[(base, "full")]["mse_cal_std"]
    block_in_std = stats[(base, "block")]["mse_cal_std"]
    full_out = stats[(base, "full")]["mse_eval"]
    block_out = stats[(base, "block")]["mse_eval"]
    full_out_std = stats[(base, "full")]["mse_eval_std"]
    block_out_std = stats[(base, "block")]["mse_eval_std"]
    in_gap = full_in - block_in
    out_gap = full_out - block_out
    in_worse = in_gap > max(full_in_std, block_in_std)
    out_worse = out_gap > max(full_out_std, block_out_std)
    gaps = [stats[(eps, "full")]["mse_eval"] - stats[(eps, "block")]["mse_eval"] for eps in eps_list]
    narrowed = gaps[-1] < 0.5 * gaps[0] or min(gaps) <= 0.0
    flat = (max(gaps) - min(gaps)) <= max(full_out_std, block_out_std)
    large = eps_list[-1]
    full_large = stats[(large, "full")]["mse_eval"]
    near_raw = abs(full_large - raw_eval) <= 0.1 * max(raw_eval, 1e-12)
    if not out_worse:
        cause = "样本外差距没有超过种子标准差，不能判成样本外放大。"
    elif (not in_worse) and narrowed:
        cause = "样本外放大，问题在 Σ 估计。"
    elif in_worse and flat:
        cause = "PQ 码结构：秩 1 时两种做法都只是在量化一个标量。"
    elif (not in_worse) and out_worse and flat:
        cause = "另有原因，先看 weak_share_eval 再定。"
    else:
        cause = "三行对照没有同时对上，保留各项数字，不单押一种原因。"
    return dict(
        in_gap=in_gap, out_gap=out_gap, in_worse=in_worse, out_worse=out_worse,
        gaps=gaps, narrowed=narrowed, flat=flat, near_raw=near_raw, cause=cause,
    )


def main() -> int:
    cols = 64
    weight_rng = np.random.default_rng(11)
    weight = small_row_weights(128, cols, weight_rng)
    orthogonal, _ = np.linalg.qr(np.random.default_rng(31).standard_normal((cols, cols)))
    calib = rank1_pair(256, cols, orthogonal, np.random.default_rng(101))
    eval_act = rank1_pair(256, cols, orthogonal, np.random.default_rng(202))
    eps_list = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
    seeds = range(5)
    rows = oos_check(weight, calib, eval_act, seeds=seeds, eps_list=eps_list)
    stats = summarize(rows)
    raw = raw_rows(weight, calib, eval_act, seeds=seeds)
    raw_cal, raw_cal_std = _mean_std([item["mse_cal"] for item in raw])
    raw_eval, raw_eval_std = _mean_std([item["mse_eval"] for item in raw])
    print("秩 1，numpy 种子。校准 256、评估 256。k-means 8 次迭代、5 个种子。标准差除以 4。")
    print("这一表用全部 256 条校准估计 Σ，评估列只作诊断，不拿来选 ε。")
    print(f"{'ε':>8} {'路径':<6} {'样本内':>18} {'样本外':>18} {'弱方向占比':>18} {'rho中位':>8} {'弱方向数':>8}")
    for eps in eps_list:
        for path in ("block", "full"):
            item = stats[(eps, path)]
            print(
                f"{eps:8.0e} {path:<6} {_fmt(item['mse_cal'], item['mse_cal_std']):>18} "
                f"{_fmt(item['mse_eval'], item['mse_eval_std']):>18} "
                f"{_fmt(item['weak_share'], item['weak_share_std']):>18} {item['rho_median']:8.3f} {item['n_weak']:8d}"
            )
    print(f"raw 不白化  样本内 {_fmt(raw_cal, raw_cal_std)}  样本外 {_fmt(raw_eval, raw_eval_std)}")
    held_rows, best = select_eps(weight, calib, eps_list, seeds=seeds)
    print("ε 在校准集内部选：前 192 条估计 Σ，后 64 条看完整白化的 MSE。评估集没有进入这一步。")
    for item in held_rows:
        print(f"  ε={item['eps']:.0e}  64 条上块白化 {_fmt(item['block'], item['block_std'])}  完整白化 {_fmt(item['full'], item['full_std'])}")
    print(f"64 条上完整白化最低的是 ε={best['eps']:.0e}。")
    verdict = classify(stats, eps_list, raw_eval)
    print(
        f"默认 ε={eps_list[0]:.0e}：样本内完整减块 {verdict['in_gap']:+.4f}，样本外 {verdict['out_gap']:+.4f}。"
        f"样本外各 ε 的差距为 {', '.join(f'{gap:+.4f}' for gap in verdict['gaps'])}。"
    )
    base = stats[(eps_list[0], "full")]
    block_base = stats[(eps_list[0], "block")]
    print(
        f"默认 ε 的弱方向占比：完整白化 {base['weak_share']:.3f}，块白化 {block_base['weak_share']:.3f}。"
        f"rho 中位 {base['rho_median']:.3f}，弱方向 {base['n_weak']} 个。"
    )
    print(f"判定：{verdict['cause']}")
    if verdict["near_raw"]:
        print(f"ε={eps_list[-1]:.0e} 时完整白化的样本外 MSE 接近 raw。这一档是把白化关掉了，不能当成度量本身起了作用。")
    else:
        print(f"ε={eps_list[-1]:.0e} 时完整白化样本外 {_fmt(stats[(eps_list[-1], 'full')]['mse_eval'], stats[(eps_list[-1], 'full')]['mse_eval_std'])}，raw 是 {_fmt(raw_eval, raw_eval_std)}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
