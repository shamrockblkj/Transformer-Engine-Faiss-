#!/usr/bin/env python3
"""W3 离线 PQ 的数学预处理：列缩放、分块 FWHT 旋转、按行 RMS（B2.3）、子空间 4×4 二阶矩白化（B2.5）。

全部在 NumPy / FP32 或 FP64 中完成。不含 Faiss 调用，也不含任何下载或登录逻辑。
旋转与列缩放是本周固定前置（四组对照不把它们列入消融）；行 RMS 与 Σ 加权才是四组因子。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

DSUB_DEFAULT = 4


def _require_2d(name: str, x: np.ndarray) -> None:
    if x.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，实际 shape={x.shape}")


def fwht_unnormalized(x: np.ndarray) -> np.ndarray:
    """沿最后一维做未归一化 FWHT。该维长度必须是 2 的幂。"""
    x_arr = np.asarray(x)
    n = x_arr.shape[-1]
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"FWHT 最后一维必须是 2 的幂，得到 {n}")
    orig = x_arr.shape
    y = np.array(x_arr, dtype=np.float64, copy=True, order="C").reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a = y[:, :, 0, :].copy()
        b = y[:, :, 1, :].copy()
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        y = y.reshape(-1, n)
        h *= 2
    return y.reshape(orig)


def fwht_normalized(x: np.ndarray) -> np.ndarray:
    """归一化 FWHT：H x / sqrt(n)，沿最后一维。两次应用等于恒等（数值误差内）。"""
    n = x.shape[-1]
    return fwht_unnormalized(x) / np.sqrt(n)


def make_sign_vector(dim: int, seed: int) -> np.ndarray:
    """确定性随机 ±1 对角（方案 2.2 的 D）。"""
    rng = np.random.RandomState(int(seed))
    # RandomState.choice 可完全复现；值为 ±1 的 float64。
    return rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=int(dim))


@dataclass(frozen=True)
class RotationSpec:
    """右乘正交旋转 R = H D / sqrt(d_block)。

    gate/up：整段 4096 维 FWHT，共用同一组符号（同输入、同旋转）。
    down：默认 7 段独立 2048 维分块 FWHT（14336 = 7 × 2048）。
    """

    dim: int
    signs: np.ndarray
    block_size: int

    def __post_init__(self) -> None:
        if self.signs.shape != (self.dim,):
            raise ValueError(f"符号向量长度 {self.signs.shape} 与 dim={self.dim} 不一致")
        if self.dim % self.block_size != 0:
            raise ValueError(f"dim={self.dim} 不能被 block_size={self.block_size} 整除")
        if self.block_size & (self.block_size - 1) != 0:
            raise ValueError(f"block_size 必须是 2 的幂，得到 {self.block_size}")


def build_rotation_spec(dim: int, seed: int, block_size: Optional[int] = None) -> RotationSpec:
    block = int(block_size) if block_size is not None else int(dim)
    return RotationSpec(dim=int(dim), signs=make_sign_vector(int(dim), seed), block_size=block)


def _apply_fwht_blocks(x: np.ndarray, spec: RotationSpec, inverse: bool = False) -> np.ndarray:
    """沿最后一维按 block_size 分段应用 R 或 R^T。

    行向量约定：R = (H / sqrt(d_block)) D，故
    - 正向（inverse=False）：先归一化 FWHT，再乘符号，等价于右乘 R；
    - 反向（inverse=True）：先乘符号，再归一化 FWHT，等价于右乘 R^T。
    因 R 正交，W @ x（行形式 x @ W.T）等于 (x @ R) @ (W @ R).T。
    """
    _require_2d("旋转输入", x)
    if x.shape[1] != spec.dim:
        raise ValueError(f"最后一维 {x.shape[1]} 与旋转 dim={spec.dim} 不一致")
    n_blocks = spec.dim // spec.block_size
    x64 = np.asarray(x, dtype=np.float64)
    if inverse:
        x64 = x64 * spec.signs[None, :]
        blocks = x64.reshape(x64.shape[0], n_blocks, spec.block_size)
        return fwht_normalized(blocks).reshape(x64.shape[0], spec.dim)
    blocks = x64.reshape(x64.shape[0], n_blocks, spec.block_size)
    rotated = fwht_normalized(blocks)
    return rotated.reshape(x64.shape[0], spec.dim) * spec.signs[None, :]


def rotate_weight_rows(weight: np.ndarray, spec: RotationSpec) -> np.ndarray:
    """W_rot = W @ R，R = H D / sqrt(d_block)。W 形状 [N, D]（行 = 输出通道）。"""
    return _apply_fwht_blocks(weight, spec, inverse=False)


def rotate_activations(x: np.ndarray, spec: RotationSpec) -> np.ndarray:
    """行形式的旋转域激活：x_rot_row = x_row @ R，对应列向量 x_rot = R^T x。

    验证：x @ W.T == x_rot @ W_rot.T（见 self-check）。
    """
    return _apply_fwht_blocks(x, spec, inverse=False)


def unrotate_weight_rows(weight_rot: np.ndarray, spec: RotationSpec) -> np.ndarray:
    """W_s = W_rot @ R^T，回到列缩放域。不能对正向再套一次（那是右乘 R 而不是 R^T）。"""
    return _apply_fwht_blocks(weight_rot, spec, inverse=True)


# ---------------------------------------------------------------------------
# 列缩放（方案 2.1，本周固定前置，不做四组消融）
# ---------------------------------------------------------------------------


def channel_rms_from_sumsq(sum_sq: np.ndarray, n_tokens: int, alpha: float = 1.0) -> np.ndarray:
    if n_tokens <= 0:
        raise ValueError("校准 token 数必须为正")
    rms = np.sqrt(np.maximum(np.asarray(sum_sq, dtype=np.float64) / float(n_tokens), 0.0))
    if alpha != 1.0:
        rms = np.power(rms, float(alpha))
    return rms


def finalize_column_scale(
    raw_s: np.ndarray,
    median_normalize: bool = True,
    clamp_min: float = 0.05,
    clamp_max: float = 20.0,
) -> np.ndarray:
    """按通道中位数归一化后再双侧钳位。见方案 2.1 安全钳位规则。

    本周单层实验不把 s^{-1} 折进 RMSNorm gamma，只在离线显式应用；
    「折叠后 |gamma|≤64」的契约检查留到后续，不在本脚本静默改写 s。
    """
    s = np.asarray(raw_s, dtype=np.float64).copy()
    if s.ndim != 1:
        raise ValueError(f"列尺度必须是一维，得到 shape={s.shape}")
    if median_normalize:
        med = float(np.median(s))
        if not np.isfinite(med) or med <= 0.0:
            raise ValueError(f"列尺度中位数非法：{med}")
        s = s / med
    if clamp_min <= 0.0 or clamp_max <= clamp_min:
        raise ValueError(f"钳位范围非法：[{clamp_min}, {clamp_max}]")
    return np.clip(s, clamp_min, clamp_max)


def apply_column_scale_weight(weight: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """W_s = W @ diag(s)，即每列乘 s_j。"""
    _require_2d("权重", weight)
    if scale.shape != (weight.shape[1],):
        raise ValueError(f"列尺度长度 {scale.shape} 与权重列数 {weight.shape[1]} 不一致")
    return np.asarray(weight, dtype=np.float64) * np.asarray(scale, dtype=np.float64)[None, :]


def undo_column_scale_weight(weight_s: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """W = W_s / s（按列）。"""
    _require_2d("权重", weight_s)
    s = np.asarray(scale, dtype=np.float64)
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("列尺度必须为正有限值")
    return np.asarray(weight_s, dtype=np.float64) / s[None, :]


def apply_column_scale_activations(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """x_s = x / s。"""
    _require_2d("激活", x)
    s = np.asarray(scale, dtype=np.float64)
    if s.shape != (x.shape[1],):
        raise ValueError(f"列尺度长度 {s.shape} 与激活宽度 {x.shape[1]} 不一致")
    if np.any(s <= 0.0) or not np.all(np.isfinite(s)):
        raise ValueError("列尺度必须为正有限值")
    return np.asarray(x, dtype=np.float64) / s[None, :]


# ---------------------------------------------------------------------------
# B2.3 按行 RMS 归一化
# ---------------------------------------------------------------------------


def row_rms(weight: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    _require_2d("权重", weight)
    # 不把 eps 加进均方内部，避免普遍抬高尺度；仅用下限保护除零。
    rms = np.sqrt(np.mean(np.square(np.asarray(weight, dtype=np.float64)), axis=1))
    return np.maximum(rms, float(eps))


def store_row_scale_f16(rms: np.ndarray) -> np.ndarray:
    """部署形态：FP16 行尺度。训练与重建都使用 cast 回 FP32 后的值，避免训练/解码不一致。"""
    return np.asarray(rms, dtype=np.float32).astype(np.float16)


def apply_row_rms(weight: np.ndarray, scale_f16: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """每行除以已存储的 FP16 尺度（升回 FP32）。"""
    _require_2d("权重", weight)
    scale = np.maximum(np.asarray(scale_f16, dtype=np.float32).astype(np.float64), float(eps))
    if scale.shape != (weight.shape[0],):
        raise ValueError(f"行尺度长度 {scale.shape} 与行数 {weight.shape[0]} 不一致")
    return np.asarray(weight, dtype=np.float64) / scale[:, None]


def undo_row_rms(weight_norm: np.ndarray, scale_f16: np.ndarray) -> np.ndarray:
    scale = np.asarray(scale_f16, dtype=np.float32).astype(np.float64)
    return np.asarray(weight_norm, dtype=np.float64) * scale[:, None]


# ---------------------------------------------------------------------------
# B2.5 子空间 4×4 激活二阶矩加权（先白化再交给 Faiss）
# ---------------------------------------------------------------------------


def n_subspaces(dim: int, dsub: int = DSUB_DEFAULT) -> int:
    if dsub <= 0 or dim % dsub != 0:
        raise ValueError(f"dim={dim} 不能被 dsub={dsub} 整除")
    return dim // dsub


def init_sigma_accum(dim: int, dsub: int = DSUB_DEFAULT) -> dict:
    m = n_subspaces(dim, dsub)
    return {
        "dsub": int(dsub),
        "dim": int(dim),
        "sum_xx": np.zeros((m, dsub, dsub), dtype=np.float64),
        "n_tokens": 0,
    }


def accumulate_sigma(accum: dict, x_rot: np.ndarray) -> None:
    """用旋转域激活累加每个 4 维子空间的二阶矩 Σ_m = E[x x^T]（不加中心化，见 B2.5）。"""
    _require_2d("旋转域激活", x_rot)
    dsub = int(accum["dsub"])
    if x_rot.shape[1] != accum["dim"]:
        raise ValueError(f"激活宽度 {x_rot.shape[1]} 与累加器 dim={accum['dim']} 不一致")
    x64 = np.asarray(x_rot, dtype=np.float64)
    m = accum["sum_xx"].shape[0]
    # 按子空间块计算 X^T X，避免构造巨大的 D×D 矩阵。
    for i in range(m):
        sl = slice(i * dsub, (i + 1) * dsub)
        block = x64[:, sl]
        accum["sum_xx"][i] += block.T @ block
    accum["n_tokens"] += int(x64.shape[0])


def finalize_sigma(accum: dict, eig_floor: float = 1e-8) -> dict:
    """由累加器得到 Σ、Σ^{1/2}、Σ^{-1/2}。特征值下限防止死子空间导致不可逆。"""
    if accum["n_tokens"] <= 0:
        raise ValueError("尚未累加任何校准 token，无法估计 Σ（待准备校准语料）")
    mean_xx = accum["sum_xx"] / float(accum["n_tokens"])
    dsub = int(accum["dsub"])
    m = mean_xx.shape[0]
    sqrt_s = np.zeros_like(mean_xx)
    inv_sqrt_s = np.zeros_like(mean_xx)
    eigs = np.zeros((m, dsub), dtype=np.float64)
    floored = 0
    floor = float(eig_floor)
    if floor <= 0.0:
        raise ValueError(f"eig_floor 必须为正，得到 {eig_floor}")
    for i in range(m):
        # 对称化，消除累加次序带来的轻微非对称。
        sym = 0.5 * (mean_xx[i] + mean_xx[i].T)
        evals, evecs = np.linalg.eigh(sym)
        floored += int(np.count_nonzero(evals < floor))
        evals = np.maximum(evals, floor)
        eigs[i] = evals
        sqrt_e = np.sqrt(evals)
        sqrt_s[i] = (evecs * sqrt_e[None, :]) @ evecs.T
        inv_sqrt_s[i] = (evecs * (1.0 / sqrt_e)[None, :]) @ evecs.T
    return {
        "sigma": mean_xx,
        "sqrt": sqrt_s,
        "inv_sqrt": inv_sqrt_s,
        "eigenvalues": eigs,
        "n_tokens": int(accum["n_tokens"]),
        "dsub": dsub,
        "n_floored_eigs": floored,
        "eig_floor": floor,
    }


def _apply_block_transform(rows: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """对每个 4 维子空间做右乘：row_m @ mat_m。mats: [M, dsub, dsub]。"""
    _require_2d("权重子向量", rows)
    dsub = mats.shape[1]
    if rows.shape[1] != mats.shape[0] * dsub:
        raise ValueError(
            f"行宽 {rows.shape[1]} 与子空间 mats {mats.shape} 不匹配（dsub={dsub}）"
        )
    n, dim = rows.shape
    m = mats.shape[0]
    x = np.asarray(rows, dtype=np.float64).reshape(n, m, dsub)
    # out[n, m, dsub] = x[n, m, k] * mats[m, k, dsub]
    out = np.einsum("nmk,mkd->nmd", x, mats, optimize=True)
    return out.reshape(n, dim)


def whiten_weight_rows(weight: np.ndarray, sigma_factors: dict) -> np.ndarray:
    """权重子向量右乘 Σ_m^{1/2}，再交给 Faiss（B2.5）。Σ^{1/2} 取主平方根，对称。"""
    return _apply_block_transform(weight, sigma_factors["sqrt"])


def unwhiten_weight_rows(weight_white: np.ndarray, sigma_factors: dict) -> np.ndarray:
    """白化域向量或质心右乘 Σ_m^{-1/2}，变回旋转（及可选行归一化）域。"""
    return _apply_block_transform(weight_white, sigma_factors["inv_sqrt"])


def unwhiten_centroids(centroids: np.ndarray, sigma_factors: dict) -> np.ndarray:
    """centroids: [M, ksub, dsub]；每个子空间的 256 个中心右乘 Σ_m^{-1/2}。"""
    if centroids.ndim != 3:
        raise ValueError(f"质心必须是 [M, ksub, dsub]，得到 {centroids.shape}")
    m, ksub, dsub = centroids.shape
    mats = sigma_factors["inv_sqrt"]
    if mats.shape != (m, dsub, dsub):
        raise ValueError(f"inv_sqrt 形状 {mats.shape} 与质心 {centroids.shape} 不匹配")
    c = np.asarray(centroids, dtype=np.float64)
    return np.einsum("mkd,mde->mke", c, mats, optimize=True)


def reconstruct_from_codes(codes: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """用码字与质心重建行向量。不调用 Faiss search / decode。

    codes: [N, M] uint8；centroids: [M, ksub, dsub]（已在目标域，例如反白化后）。
    返回 [N, M*dsub]。
    """
    if codes.ndim != 2 or centroids.ndim != 3:
        raise ValueError("codes 为 [N,M]，centroids 为 [M,ksub,dsub]")
    n, m = codes.shape
    if centroids.shape[0] != m:
        raise ValueError(f"质心 M={centroids.shape[0]} 与码字 M={m} 不一致")
    ksub, dsub = centroids.shape[1], centroids.shape[2]
    if codes.dtype != np.uint8:
        raise ValueError(f"码字必须是 uint8，得到 {codes.dtype}")
    if int(codes.max(initial=0)) >= ksub:
        raise ValueError(f"码字最大值 {int(codes.max())} 超出 ksub={ksub}")
    out = np.empty((n, m * dsub), dtype=np.float64)
    c = np.asarray(centroids, dtype=np.float64)
    for i in range(m):
        out[:, i * dsub : (i + 1) * dsub] = c[i, codes[:, i], :]
    return out


# ---------------------------------------------------------------------------
# 把重建结果从「Faiss 输入域」还原到旋转域 / 原始权重域
# ---------------------------------------------------------------------------


@dataclass
class PreprocessBundle:
    """某个 (layer, role) 的固定前置 + 可选对照状态（不含 PQ 本身）。"""

    column_scale: Optional[np.ndarray]  # [D]
    rotation: Optional[RotationSpec]
    row_scale_f16: Optional[np.ndarray]  # [N]
    sigma_factors: Optional[dict]
    weight_rot: np.ndarray  # 列缩放+旋转后、行 RMS 之前，作为四组对照的公共参考域


def to_faiss_matrix(weight_rot: np.ndarray, use_row_rms: bool, use_sigma: bool, bundle_state: dict) -> np.ndarray:
    """从旋转域权重得到交给 Faiss 的矩阵（FP64，调用方再转 float32）。"""
    w = weight_rot
    if use_row_rms:
        w = apply_row_rms(w, bundle_state["row_scale_f16"], eps=bundle_state.get("row_rms_eps", 1e-12))
    if use_sigma:
        w = whiten_weight_rows(w, bundle_state["sigma_factors"])
    return w


def reconstructed_in_rot_domain(
    codes: np.ndarray,
    centroids_faiss: np.ndarray,
    use_row_rms: bool,
    use_sigma: bool,
    bundle_state: dict,
) -> np.ndarray:
    """把 Faiss 域的质心还原到旋转域（乘回行尺度、反白化）。"""
    centroids = np.asarray(centroids_faiss, dtype=np.float64)
    if use_sigma:
        centroids = unwhiten_centroids(centroids, bundle_state["sigma_factors"])
    recon = reconstruct_from_codes(codes, centroids)
    if use_row_rms:
        recon = undo_row_rms(recon, bundle_state["row_scale_f16"])
    return recon


def rot_domain_to_original(
    weight_rot: np.ndarray,
    column_scale: Optional[np.ndarray],
    rotation: Optional[RotationSpec],
) -> np.ndarray:
    """旋转域 → 列缩放域 → 原始权重域。"""
    w = np.asarray(weight_rot, dtype=np.float64)
    if rotation is not None:
        w = unrotate_weight_rows(w, rotation)
    if column_scale is not None:
        w = undo_column_scale_weight(w, column_scale)
    return w
