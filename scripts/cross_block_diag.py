#!/usr/bin/env python3
"""三步检查：Σ 有没有跨块结构，块白化后还在不在，PQ 残差自己相不相关。

交叉项小，只说明各块残差没有叠加。不能据此说块对角近似没有损失。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from whiten_roundtrip_check_v2 import block_stats, load_matrix


def block_whiteners(activation: np.ndarray, d_sub: int, eps_rel: float):
    _blocks, forward, inverse, _conds = block_stats(activation, d_sub, eps_rel)
    return forward, inverse, _conds


def cross_block_diag(activation: np.ndarray, delta: np.ndarray, d_sub: int = 4, eps_rel: float = 1e-6):
    samples = np.asarray(activation, dtype=np.float64)
    residual = np.asarray(delta, dtype=np.float64)
    second = samples.T @ samples / len(samples)
    width = len(second)
    mask = np.kron(np.eye(width // d_sub), np.ones((d_sub, d_sub)))
    off_energy = np.linalg.norm(second * (1 - mask)) / np.linalg.norm(second)
    _forward, inverse, _conds = block_whiteners(samples, d_sub, eps_rel)
    whitener = np.zeros_like(second)
    for index, block in enumerate(inverse):
        start = index * d_sub
        whitener[start : start + d_sub, start : start + d_sub] = block
    canonical = whitener @ second @ whitener
    max_canon = float(np.abs(canonical * (1 - mask)).max())
    correlation = np.corrcoef(residual.T)
    resid_cross = float(np.abs(correlation * (1 - mask)).mean())
    top_corr, top_share, top_quad = direction_block_corr(second, residual, d_sub)
    return float(off_energy), max_canon, resid_cross, top_corr, top_share, top_quad


def direction_block_corr(second: np.ndarray, residual: np.ndarray, d_sub: int) -> tuple[float, float, float]:
    """把各块残差投到 Σ 的最大特征向量上。Pearson 相关会按块标准化，二次型才是交叉项本身。"""
    moment = 0.5 * (second + second.T)
    evals, evecs = np.linalg.eigh(moment)
    top = int(np.argmax(evals))
    width = moment.shape[0]
    groups = width // d_sub
    direction = evecs[:, top].reshape(groups, d_sub)
    projected = np.einsum("ibd,bd->ib", residual.reshape(residual.shape[0], groups, d_sub), direction)
    within = np.sum(projected * projected, axis=1)
    total = np.sum(projected, axis=1) ** 2
    quad = float((total - within).sum() / total.sum()) if float(total.sum()) > 0.0 else float("nan")
    std = projected.std(axis=0)
    keep = std > 1e-8
    if int(keep.sum()) < 2:
        return float("nan"), float(evals[top] / np.trace(moment)), quad
    corr = np.corrcoef(projected[:, keep].T)
    off = ~np.eye(corr.shape[0], dtype=bool)
    return float(corr[off].mean()), float(evals[top] / np.trace(moment)), quad


def read_case(directory: Path) -> tuple[float, float, float, float, float, float]:
    weight = load_matrix(directory / "W.bin")
    activation = load_matrix(directory / "X.bin")
    recon = load_matrix(directory / "recon.bin")
    return cross_block_diag(activation, weight - recon)


def main() -> int:
    parser = argparse.ArgumentParser(description="对 pq_layer --dump-cross 写出的目录做三步检查。")
    parser.add_argument("root", type=Path, help="含 axis、one、many 子目录，或直接含 W.bin 的一组")
    args = parser.parse_args()
    cases = ["axis", "many", "one"]
    if (args.root / "W.bin").is_file():
        names = [args.root.name]
        dirs = [args.root]
    else:
        names = cases
        dirs = [args.root / name for name in names]
    print(f"{'数据':<8} {'①跨块能量':>12} {'②最大典型相关':>14} {'③旧':>8} {'③相关':>8} {'③二次型':>8} {'主方向能量':>10}")
    for name, directory in zip(names, dirs):
        off_energy, max_canon, resid_cross, top_corr, top_share, top_quad = read_case(directory)
        print(
            f"{name:<8} {off_energy:12.3f} {max_canon:14.3f} {resid_cross:8.3f} {top_corr:8.3f} {top_quad:8.3f} {top_share:10.3f}"
        )
        if not np.isfinite([off_energy, max_canon, resid_cross, top_corr, top_share, top_quad]).all():
            print("三步检查出现非有限数", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
