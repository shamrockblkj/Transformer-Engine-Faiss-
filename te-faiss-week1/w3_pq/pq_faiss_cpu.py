#!/usr/bin/env python3
"""CPU Faiss ProductQuantizer 封装：只暴露 train / compute_codes，禁止 search / IndexIVFPQ / GpuIndex。"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

import numpy as np

# 只允许的公开方法名（本模块对外）。
ALLOWED_OPS = ("train", "compute_codes", "centroids")


def _fail(msg: str, code: int = 2) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def import_faiss_cpu(forbid_gpu: bool = True, expected_version_prefix: str = "1.16"):
    try:
        import faiss  # type: ignore
    except ImportError as exc:
        _fail(
            "无法导入 faiss。请在 conda 环境 torch2.8_cuda12.8 中使用已锁定的 faiss-cpu 1.16.0。"
            f" 原始错误：{exc}"
        )

    # 版本核对：优先发行包名 faiss-cpu，避免误装 faiss-gpu。
    ver = getattr(faiss, "__version__", None)
    dist_name = None
    dist_ver = None
    try:
        from importlib.metadata import PackageNotFoundError, version

        for name in ("faiss-cpu", "faiss_cpu", "faiss"):
            try:
                dist_ver = version(name)
                dist_name = name
                break
            except PackageNotFoundError:
                continue
    except Exception:
        pass

    resolved = dist_ver or ver
    if expected_version_prefix and resolved is not None:
        if not str(resolved).startswith(expected_version_prefix):
            _fail(
                f"Faiss 版本为 {resolved}（发行包 {dist_name}），与锁定的 faiss-cpu "
                f"{expected_version_prefix}.x 不符。拒绝继续，以免 silently 换栈。"
            )
    if forbid_gpu and dist_name in {"faiss-gpu", "faiss-gpu-cuvs"}:
        _fail(f"检测到 GPU Faiss 发行包 {dist_name}。本实验只允许 faiss-cpu，拒绝继续。")

    # 即使环境里混入了 GPU 符号，本模块也绝不实例化它们。
    return faiss, resolved, dist_name


def _set_if_has(obj: Any, name: str, value: Any) -> bool:
    if obj is None or not hasattr(obj, name):
        return False
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        return False


class CpuProductQuantizer:
    """faiss.ProductQuantizer 的薄封装。

    不提供 search、不构造任何 Index。重建走 pq_math.reconstruct_from_codes。
    """

    def __init__(
        self,
        d: int,
        m: int,
        nbits: int,
        seed: int,
        niter: int = 20,
        nredo: int = 1,
        max_points_per_centroid: int = 256,
        verbose: bool = True,
        faiss_mod=None,
    ) -> None:
        if nbits != 8:
            _fail(f"本周锁定 nbits=8（每子量化器 256 中心），得到 nbits={nbits}")
        if d <= 0 or m <= 0 or d % m != 0:
            _fail(f"ProductQuantizer 要求 d 能被 M 整除，得到 d={d}, M={m}")
        self.d = int(d)
        self.M = int(m)
        self.nbits = int(nbits)
        self.dsub = self.d // self.M
        self.ksub = 1 << self.nbits
        self.seed = int(seed)

        faiss = faiss_mod
        if faiss is None:
            faiss, _, _ = import_faiss_cpu()
        self._faiss = faiss

        if not hasattr(faiss, "ProductQuantizer"):
            _fail("该 Faiss 构建不含 ProductQuantizer")

        self._pq = faiss.ProductQuantizer(self.d, self.M, self.nbits)

        # 尽量把聚类随机源与迭代次数钉死，便于复现。
        cp = getattr(self._pq, "cp", None)
        _set_if_has(cp, "seed", int(seed))
        _set_if_has(cp, "niter", int(niter))
        _set_if_has(cp, "nredo", int(nredo))
        _set_if_has(cp, "max_points_per_centroid", int(max_points_per_centroid))
        _set_if_has(self._pq, "verbose", bool(verbose))
        # 部分绑定在模块级提供 srand。
        if hasattr(faiss, "srand"):
            try:
                faiss.srand(int(seed))
            except Exception:
                pass

    def train(self, x: np.ndarray) -> None:
        x32 = _as_train_matrix(x, self.d)
        if x32.shape[0] < self.ksub:
            _fail(
                f"训练行数 {x32.shape[0]} 少于每子空间中心数 {self.ksub}，"
                "Faiss k-means 无法稳定进行。"
            )
        # 只调用 train。
        self._pq.train(x32)

    def compute_codes(self, x: np.ndarray) -> np.ndarray:
        x32 = _as_train_matrix(x, self.d)
        codes = self._pq.compute_codes(x32)
        codes = np.asarray(codes)
        if codes.dtype != np.uint8:
            codes = codes.astype(np.uint8, copy=False)
        if codes.ndim != 2 or codes.shape != (x32.shape[0], self.M):
            _fail(f"compute_codes 返回形状异常：{codes.shape}，期望 {(x32.shape[0], self.M)}")
        return np.ascontiguousarray(codes)

    def centroids(self) -> np.ndarray:
        """返回 [M, ksub, dsub] float64 副本。"""
        pq = self._pq
        raw = None
        if hasattr(pq, "centroids"):
            cent = pq.centroids
            # swig 的 float vector
            try:
                raw = self._faiss.vector_to_array(cent)
            except Exception:
                raw = np.asarray(cent)
        if raw is None:
            _fail("无法读取 ProductQuantizer.centroids")
        arr = np.asarray(raw, dtype=np.float64).reshape(-1)
        expected = self.M * self.ksub * self.dsub
        if arr.size != expected:
            _fail(f"质心长度 {arr.size} 与 M*ksub*dsub={expected} 不符")
        return arr.reshape(self.M, self.ksub, self.dsub).copy()


def _as_train_matrix(x: np.ndarray, d: int) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2 or arr.shape[1] != d:
        _fail(f"Faiss 输入必须是 [N, {d}]，得到 {arr.shape}")
    if not np.all(np.isfinite(arr)):
        _fail("Faiss 输入含 NaN/Inf，停止（可能是列尺度或白化数值问题）")
    return np.ascontiguousarray(arr, dtype=np.float32)


def configure_threads(n_threads: Optional[int]) -> None:
    """在 import faiss 之前由主程序调用更稳妥；此处再设一次 OpenMP 线程数。"""
    if n_threads is None:
        return
    n = int(n_threads)
    if n <= 0:
        return
    os.environ.setdefault("OMP_NUM_THREADS", str(n))
    try:
        import faiss  # type: ignore

        if hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(n)
    except Exception:
        pass


def assert_no_forbidden_symbols_used() -> None:
    """文档性断言：本模块源码不得出现被禁 API 的调用（静态保险）。"""
    here = os.path.abspath(__file__)
    with open(here, "r", encoding="utf-8") as f:
        src = f.read()
    # 去掉本函数所在的检查字符串后，不应再出现调用。
    # 允许在注释或本列表里提到名字，因此只查典型调用形态。
    forbidden_calls = (
        ".search(",
        "IndexIVFPQ(",
        "GpuIndex(",
        "GpuIndexFlat",
        "index_cpu_to_gpu",
        "StandardGpuResources(",
    )
    for token in forbidden_calls:
        # 本列表本身含这些子串；只统计出现次数是否超出列表定义。
        if src.count(token) > 1:
            _fail(f"pq_faiss_cpu.py 中出现被禁调用形态 {token}，请删除。")
