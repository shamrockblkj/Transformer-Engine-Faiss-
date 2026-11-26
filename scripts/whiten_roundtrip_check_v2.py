#!/usr/bin/env python3
"""白化往返一致性检查 v2

不经量化，走一遍 行RMS → 子空间白化 → 反白化，W 应恢复到浮点误差级别。
口径：Σ_b = E[x_b x_bᵀ]（不减均值），每个子空间一个 d_sub×d_sub 块；
前向 W'_b = (W/r)_b · Σ_b^{1/2}，反向 W_b = W'_b · Σ_b^{-1/2} · r。

两种用法：
  1) 只给 W、X：用本文件的参考实现自检（验证口径和代数）。
  2) 另外传 whiten_fn / unwhiten_fn：检查实际代码的往返。0.4642 要靠这一条。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PQ_LAYER = ROOT / "pq_layer"


def sym_pow(S, p, eps_rel=1e-6):
    """对称 PSD 矩阵的 p 次幂。eps 按 trace/d 相对缩放，负特征值截断。"""
    S = 0.5 * (S + S.T)
    d = len(S)
    eps = eps_rel * np.trace(S) / d
    w, V = np.linalg.eigh(S + eps * np.eye(d))
    w = np.clip(w, eps, None)
    return (V * w**p) @ V.T, w


def block_stats(X, d_sub, eps_rel):
    D = X.shape[1]
    assert D % d_sub == 0, f"D={D} 不能被 d_sub={d_sub} 整除"
    blocks = [slice(i, i + d_sub) for i in range(0, D, d_sub)]
    Sh, Sih, conds = [], [], []
    for b in blocks:
        S = X[:, b].T @ X[:, b] / len(X)
        h, w = sym_pow(S, 0.5, eps_rel)
        ih, _ = sym_pow(S, -0.5, eps_rel)
        Sh.append(h)
        Sih.append(ih)
        conds.append(w.max() / w.min())
    return blocks, Sh, Sih, np.array(conds)


def row_scale(W, use_row_rms):
    if not use_row_rms:
        return np.ones((W.shape[0], 1), W.dtype)
    r = np.sqrt((W**2).mean(1, keepdims=True))
    return np.where(r > 0, r, 1.0)  # 全零行不除零


def ref_whiten(W, r, blocks, Sh):
    Wr = W / r
    return np.concatenate([Wr[:, b] @ M for b, M in zip(blocks, Sh)], axis=1)


def ref_unwhiten(Wq, r, blocks, mats):
    return np.concatenate([Wq[:, b] @ M for b, M in zip(blocks, mats)], axis=1) * r


def roundtrip_check(W, X, d_sub=4, eps_rel=1e-6, use_row_rms=True, whiten_fn=None, unwhiten_fn=None, tol=None):
    W = np.asarray(W)
    X = np.asarray(X)
    assert W.shape[1] == X.shape[1], "W 列数必须等于激活维度"
    if tol is None:
        tol = 1e-10 if W.dtype == np.float64 else 1e-4  # float32 放宽
    blocks, Sh, Sih, conds = block_stats(X.astype(np.float64), d_sub, eps_rel)
    r = row_scale(W, use_row_rms)
    # 每块 Σ^{1/2}·Σ^{-1/2} 应为单位阵
    eye_err = max(np.abs(h @ ih - np.eye(d_sub)).max() for h, ih in zip(Sh, Sih))
    if whiten_fn is None:  # 参考实现
        Wp = ref_whiten(W, r, blocks, Sh)
        W_back = ref_unwhiten(Wp, r, blocks, Sih)
    else:  # 实际代码：whiten_fn(W, X) -> (Wp, state); unwhiten_fn(Wp, state) -> W
        Wp, state = whiten_fn(W, X)
        W_back = unwhiten_fn(Wp, state)
    W_wrong = ref_unwhiten(ref_whiten(W, r, blocks, Sh), r, blocks, Sh)  # 方向写反的对照
    nW = np.linalg.norm(W)
    rel = lambda A: np.linalg.norm(A - W) / nW
    Y = X @ W.T
    out_rel = lambda A: np.linalg.norm(X @ A.T - Y) / np.linalg.norm(Y)
    res = dict(
        weight_rel=rel(W_back),
        out_rel=out_rel(W_back),
        weight_rel_wrong_dir=rel(W_wrong),
        eye_err=eye_err,
        cond_max=conds.max(),
        cond_median=np.median(conds),
        n_samples=len(X),
        d_sub=d_sub,
        eps_rel=eps_rel,
        tol=tol,
    )
    res["passed"] = bool(res["weight_rel"] < tol and eye_err < 1e3 * tol)
    return res


def report(res):
    for k, v in res.items():
        print(f"{k:22s} {v:.3e}" if isinstance(v, float) else f"{k:22s} {v}")
    if res["n_samples"] < 10 * res["d_sub"]:
        print("警告：样本数 < 10×d_sub，Σ 估计不稳定")
    if res["cond_max"] > 1e3:
        print("警告：存在条件数 >1e3 的子空间块，量化误差会在弱方向被 Σ^{-1/2} 放大")


def save_matrix(path: Path, array: np.ndarray) -> None:
    values = np.ascontiguousarray(array, dtype=np.float64)
    header = np.array(values.shape, dtype=np.int32)
    with path.open("wb") as handle:
        header.tofile(handle)
        values.tofile(handle)


def load_matrix(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int32, count=2)
    rows, cols = int(raw[0]), int(raw[1])
    data = np.fromfile(path, dtype=np.float64, offset=8)
    if data.size != rows * cols:
        raise SystemExit(f"{path} 的长度和形状 {rows}×{cols} 不一致")
    return data.reshape(rows, cols)


def cpp_pair(row_rms: bool):
    """返回实际 pq_layer 的 whiten_fn / unwhiten_fn。白化走 prepare_work，反白化走 restore_work。"""
    mode = "rms" if row_rms else "no"

    def whiten_fn(weight, activation):
        directory = Path(tempfile.mkdtemp(prefix="whiten-"))
        save_matrix(directory / "W.bin", weight)
        save_matrix(directory / "X.bin", activation)
        subprocess.run(
            [str(PQ_LAYER), "--whiten", str(directory / "W.bin"), str(directory / "X.bin"), str(directory), mode],
            check=True,
        )
        return load_matrix(directory / "Wp.bin"), directory

    def unwhiten_fn(coded, directory):
        save_matrix(directory / "Wp_in.bin", coded)
        subprocess.run(
            [
                str(PQ_LAYER),
                "--unwhiten",
                str(directory / "Wp_in.bin"),
                str(directory / "state.bin"),
                str(directory / "Wback.bin"),
            ],
            check=True,
        )
        return load_matrix(directory / "Wback.bin")

    return whiten_fn, unwhiten_fn


def self_test() -> None:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((128, 64)) * rng.lognormal(0, 1, (128, 1))
    weight[5] = 0.0  # 测全零行
    activation = rng.standard_normal((32, 64)) * rng.lognormal(0, 1, 64)
    print("== 参考实现 float64 ==")
    res = roundtrip_check(weight, activation)
    report(res)
    assert res["passed"]
    print("\n== 参考实现 float32 ==")
    res = roundtrip_check(weight.astype(np.float32), activation.astype(np.float32))
    report(res)
    assert res["passed"]
    print("\n== 模拟一个方向写反的用户实现（应失败）==")

    def bad_whiten(matrix, samples):
        blocks, _sh, sih, _conds = block_stats(samples, 4, 1e-6)
        scale = row_scale(matrix, True)
        return ref_whiten(matrix, scale, blocks, sih), (scale, blocks, sih)  # 前向错用 Σ^{-1/2}

    def bad_unwhiten(coded, state):
        scale, blocks, sih = state
        return ref_unwhiten(coded, scale, blocks, sih)

    res = roundtrip_check(weight, activation, whiten_fn=bad_whiten, unwhiten_fn=bad_unwhiten)
    report(res)
    assert not res["passed"]
    print("\n全部自测通过")


def main() -> int:
    parser = argparse.ArgumentParser(description="白化往返检查 v2。不带参数时只跑参考实现自测。")
    parser.add_argument("--w", type=Path, help="128×64 自检的 W.bin")
    parser.add_argument("--x", type=Path, help="128×64 自检的 X.bin")
    args = parser.parse_args()
    self_test()
    if args.w is None or args.x is None:
        return 0
    if not PQ_LAYER.is_file():
        raise SystemExit("先编译 pq_layer，实际往返要调用它的 prepare_work / restore_work")
    weight = load_matrix(args.w)
    activation = load_matrix(args.x)
    print("\n== 实际代码，子空间白化，不开行 RMS（0.4642 那条）==")
    whiten_fn, unwhiten_fn = cpp_pair(False)
    res = roundtrip_check(weight, activation, whiten_fn=whiten_fn, unwhiten_fn=unwhiten_fn, use_row_rms=False)
    report(res)
    if not res["passed"]:
        raise SystemExit("实际代码往返未通过，0.4642 不能当作原始空间的输出误差")
    print("\n== 实际代码，行 RMS 加子空间白化 ==")
    whiten_fn, unwhiten_fn = cpp_pair(True)
    both = roundtrip_check(weight, activation, whiten_fn=whiten_fn, unwhiten_fn=unwhiten_fn, use_row_rms=True)
    report(both)
    if not both["passed"]:
        raise SystemExit("行 RMS 加白化的实际往返未通过")
    print("\n实际代码 passed 为 True。0.4642 可以当作原始空间的输出误差。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
