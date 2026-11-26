#!/usr/bin/env python3
"""白化往返一致性检查：不经量化，走一遍 行RMS→子空间白化→反白化，W 应恢复到浮点误差级别。

口径：Σ = E[xxᵀ]（不减均值），每个子空间一个 d_sub×d_sub 块；前向 W' = W·Σ^{1/2}，反向 ×Σ^{-1/2}。
把 W、X 换成实际用的权重和激活即可。weight_rel_ok 小于 1e-10 才算通过。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def sym_pow(S, p, eps):
    w, V = np.linalg.eigh(S + eps * np.eye(len(S)))
    return (V * w**p) @ V.T


def roundtrip_check(W, X, d_sub=4, eps=1e-6, use_row_rms=True):
    n_rows, D = W.shape
    assert D % d_sub == 0
    blocks = [slice(i, i + d_sub) for i in range(0, D, d_sub)]
    r = np.sqrt((W**2).mean(1, keepdims=True)) if use_row_rms else np.ones((n_rows, 1))
    Wr = W / r
    Sh, Sih, Wp = [], [], np.empty_like(W)
    for b in blocks:
        S = X[:, b].T @ X[:, b] / len(X)
        Sh.append(sym_pow(S, 0.5, eps))
        Sih.append(sym_pow(S, -0.5, eps))
        Wp[:, b] = Wr[:, b] @ Sh[-1]  # 前向：W' = W Σ^{1/2}

    def unwhiten(Wq, mats):
        out = np.empty_like(Wq)
        for b, M in zip(blocks, mats):
            out[:, b] = Wq[:, b] @ M
        return out * r  # 乘回行尺度

    W_ok, W_bad = unwhiten(Wp, Sih), unwhiten(Wp, Sh)  # 正确方向 / 反方向（对照）
    rel = lambda A: np.linalg.norm(A - W) / np.linalg.norm(W)
    out_mse = lambda A: np.mean((X @ A.T - X @ W.T) ** 2)
    return dict(
        weight_rel_ok=rel(W_ok),
        weight_rel_wrong=rel(W_bad),
        out_mse_ok=out_mse(W_ok),
        out_mse_wrong=out_mse(W_bad),
        max_block_cond=max(np.linalg.cond(s @ s) for s in Sh),
    )


def load_matrix(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32, count=2)
    rows, cols = int(raw[0]), int(raw[1])
    data = np.fromfile(path, dtype=np.float64, offset=8)
    if data.size != rows * cols:
        raise SystemExit(f"{path} 的长度和形状 {rows}×{cols} 不一致")
    return data.reshape(rows, cols)


def raw_sigma_cond(X: np.ndarray, d_sub: int = 4) -> float:
    worst = 0.0
    for start in range(0, X.shape[1], d_sub):
        block = X[:, start : start + d_sub]
        sigma = block.T @ block / len(X)
        worst = max(worst, float(np.linalg.cond(sigma)))
    return worst


def report(title: str, result: dict) -> None:
    print(title)
    for key, value in result.items():
        print(f"  {key:18s} {value:.3e}")
    if not result["weight_rel_ok"] < 1e-10:
        raise SystemExit("往返未恢复：检查白化/反白化方向或行尺度顺序")


def demo() -> dict:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((128, 64)) * rng.lognormal(0, 1, (128, 1))
    activation = rng.standard_normal((32, 64)) * rng.lognormal(0, 1, 64)
    return roundtrip_check(weight, activation)


def main() -> int:
    parser = argparse.ArgumentParser(description="白化往返一致性检查，不经量化。")
    parser.add_argument("--w", type=Path, help="128×64 自检的权重，W.bin")
    parser.add_argument("--x", type=Path, help="128×64 自检的激活，X.bin")
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    report("随机数据，行 RMS 开，eps=1e-6", demo())
    if args.w is None or args.x is None:
        print("未提供实际 W、X。通过项是随机数据。")
        return 0

    weight = load_matrix(args.w)
    activation = load_matrix(args.x)
    print(f"实际自检矩阵 {weight.shape[0]}×{weight.shape[1]}，激活 {activation.shape[0]} 条。不减均值。")
    print(f"不加 ε 时，最差 4×4 块的条件数 {raw_sigma_cond(activation):.3e}")
    report(
        f"实际 W/X，行 RMS 开，eps={args.eps:g}",
        roundtrip_check(weight, activation, eps=args.eps, use_row_rms=True),
    )
    report(
        f"实际 W/X，行 RMS 关，eps={args.eps:g}",
        roundtrip_check(weight, activation, eps=args.eps, use_row_rms=False),
    )
    print("weight_rel_ok 小于 1e-10，方向通过。方向写反的误差大几个数量级，只作对照。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
