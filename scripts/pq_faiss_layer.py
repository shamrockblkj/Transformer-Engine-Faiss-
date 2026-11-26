#!/usr/bin/env python3
"""用官方 Faiss ProductQuantizer 对 Llama 3 8B 的一层 FFN 做全尺寸离线 PQ。

测量主机执行。云端没有权重，也没有 GPU。云端只跑 --self-check。
自检使用缩小随机矩阵，不能写成全尺寸结论，也不能判定 S3。

示例（测量主机）：

  python3 -m pip install -r requirements-pq.txt
  python3 scripts/pq_faiss_layer.py \\
      --weights /路径/Meta-Llama-3-8B \\
      --activations-dir /路径/校准激活 \\
      --layers 0 \\
      --out out/pq_layer0

权重目录是 Hugging Face safetensors，跳过路径中含 original 的文件。
默认只跑第 0 层的 gate_proj、up_proj、down_proj。最多两层，例如 --layers 0,1。
两层是分开训练的两个码本，不会把行拼在一起。

激活文件（float32 或 float16，C 顺序 npy）：
  layer0.gate_up.npy    形状 [token 数, 4096]，gate 和 up 共用
  layer0.down.npy       形状 [token 数, 14336]

四组都会写入报告：原样、行 RMS、子空间加权、两者都用。
缩小自检里「两者都用」的小行相对误差是 0.2260，高于原样 0.1979。
全尺寸四组都要原样留下，不能只写下降的组。这一步不能判定 S3。

本脚本不做通道缩放，不做 FWHT，不做每 4 层共享码本，不算词表 KL。
行尺度在本次用 FP32 乘回，不含方案里的 FP16 行尺度舍入。
1.2 ms 与本脚本无关。
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

DSUB = 4
NBITS = 8
FAISS_VERSION = "1.16.0"
KSUB = 256
MIN_POINTS_DEFAULT = 39
RECOMMENDED_ROWS = KSUB * MIN_POINTS_DEFAULT  # 9984
SHAPES = {
    "gate": (14336, 4096),
    "up": (14336, 4096),
    "down": (4096, 14336),
}
CASES = (
    ("原样", False, False),
    ("行 RMS", True, False),
    ("子空间加权", False, True),
    ("两者都用", True, True),
)
NOTE = (
    "四组都保留。缩小自检里「两者都用」的小行相对误差是 0.2260，高于原样 0.1979；"
    "全尺寸不能只写下降的组。本步不能判定 S3。"
    "未做 FWHT，未做通道缩放，不是每 4 层共享码本，没有计算词表 KL。"
    "行尺度按 FP32 乘回。1.2 ms 与本实验无关。"
)


def fail(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def require_faiss():
    try:
        import faiss
    except ImportError:
        fail("没有安装 faiss。请执行：python3 -m pip install -r requirements-pq.txt")
    if getattr(faiss, "__version__", "") != FAISS_VERSION:
        fail(f"需要 faiss-cpu {FAISS_VERSION}，当前是 {getattr(faiss, '__version__', '未知')}。")
    return faiss


def parse_layers(text: str) -> list[int]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    try:
        layers = [int(part) for part in parts]
    except ValueError:
        fail("层号要是整数，例如 0 或 0,1")
    if not layers or any(layer < 0 or layer > 31 for layer in layers):
        fail("层号要在 0 到 31，例如 0 或 0,1")
    if len(layers) > 2:
        fail("这一步最多两层")
    if len(set(layers)) != len(layers):
        fail("层号重复")
    return layers


def parse_roles(text: str) -> list[str]:
    roles = [part.strip() for part in text.split(",") if part.strip()]
    allowed = ("gate", "up", "down")
    if not roles or any(role not in allowed for role in roles):
        fail("角色只能是 gate、up、down")
    if len(set(roles)) != len(roles):
        fail("角色重复")
    return roles


def tensor_name(layer: int, role: str) -> str:
    return f"model.layers.{layer}.mlp.{role}_proj.weight"


def activation_file(directory: Path, layer: int, role: str) -> Path:
    if role in ("gate", "up"):
        return directory / f"layer{layer}.gate_up.npy"
    return directory / f"layer{layer}.down.npy"


def read_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            fail(f"{path} 不是 safetensors")
        (nbytes,) = struct.unpack("<Q", prefix)
        if nbytes <= 0 or nbytes > 100_000_000:
            fail(f"{path} 的 safetensors 头长度异常")
        payload = handle.read(nbytes)
        if len(payload) != nbytes:
            fail(f"{path} 的 safetensors 头被截断")
    try:
        header = json.loads(payload)
    except json.JSONDecodeError:
        fail(f"{path} 的 safetensors 头不是 JSON")
    return header, 8 + nbytes


def _decode_tensor(raw: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    if dtype == "F32":
        values = np.frombuffer(raw, dtype="<f4")
        if values.size != count:
            fail("F32 字节数和形状不一致")
        return np.array(values.reshape(shape), dtype=np.float32, copy=True)
    if dtype == "F16":
        values = np.frombuffer(raw, dtype="<f2")
        if values.size != count:
            fail("F16 字节数和形状不一致")
        return np.array(values.reshape(shape), dtype=np.float32, copy=True)
    if dtype == "BF16":
        values = np.frombuffer(raw, dtype="<u2")
        if values.size != count:
            fail("BF16 字节数和形状不一致")
        bits = values.astype(np.uint32) << 16
        return np.array(bits.view(np.float32).reshape(shape), dtype=np.float32, copy=True)
    fail(f"不支持的权重 dtype {dtype}，只接受 F32、F16、BF16")
    raise AssertionError


def tensor_meta(path: Path, name: str) -> tuple[str, tuple[int, ...]]:
    header, _ = read_header(path)
    if name not in header or "data_offsets" not in header[name]:
        fail(f"{path} 里没有 {name}")
    meta = header[name]
    return str(meta["dtype"]), tuple(int(dim) for dim in meta["shape"])


def read_tensor(path: Path, name: str) -> tuple[np.ndarray, str]:
    header, data_start = read_header(path)
    if name not in header or "data_offsets" not in header[name]:
        fail(f"{path} 里没有 {name}")
    meta = header[name]
    dtype = str(meta["dtype"])
    shape = tuple(int(dim) for dim in meta["shape"])
    start, end = meta["data_offsets"]
    with path.open("rb") as handle:
        handle.seek(data_start + int(start))
        raw = handle.read(int(end) - int(start))
    array = _decode_tensor(raw, dtype, shape)
    if array.ndim != 2:
        fail(f"{name} 不是二维矩阵")
    if not np.isfinite(array).all():
        fail(f"{name} 含有非有限数")
    return array, dtype


def _reject_original(root: Path, relative: Path) -> None:
    if "original" in relative.parts:
        fail(f"已排除 original/ 下的权重：{relative}")


def _inside(root: Path, path: Path) -> None:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        fail(f"分片路径跳出了权重目录：{path}")


def find_shard(weights_dir: Path, name: str) -> Path:
    root = weights_dir.resolve()
    if root.name == "original" or "original" in root.parts:
        fail("权重目录不能是 original/")
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError):
            fail(f"无法读取 {index}")
        if name not in weight_map:
            fail(f"索引里没有 {name}。需要 HF 名称 model.layers.N.mlp.{{gate,up,down}}_proj.weight")
        relative = Path(weight_map[name])
        _reject_original(root, relative)
        shard = root / relative
        _inside(root, shard)
        if not shard.is_file():
            fail(f"索引指向的分片不存在：{shard}")
        return shard
    matches: list[Path] = []
    for candidate in sorted(root.glob("*.safetensors")):
        if "original" in candidate.parts:
            continue
        header, _ = read_header(candidate)
        if name in header and "data_offsets" in header[name]:
            matches.append(candidate)
    if not matches:
        fail(f"在 {root} 的 safetensors 里找不到 {name}，并且没有使用 original/")
    if len(matches) > 1:
        fail(f"{name} 出现在多个分片里：{', '.join(path.name for path in matches)}")
    return matches[0]


def require_llama_shape(role: str, shape: tuple[int, ...]) -> None:
    expected = SHAPES[role]
    if tuple(shape) != expected:
        fail(
            f"{role}_proj 的形状是 {tuple(shape)}，Llama 3 8B 这一层应为 {expected}。"
            "拒绝把这份权重写成全尺寸结果。",
            code=3,
        )


def load_activation(path: Path, in_features: int) -> np.ndarray:
    if not path.is_file():
        fail(f"缺少激活文件 {path}")
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        fail(f"{path} 必须是二维 npy，形状 [token 数, {in_features}]")
    if array.shape[1] != in_features:
        fail(f"{path} 的列数是 {array.shape[1]}，这一层需要 {in_features}")
    if array.shape[0] < 1:
        fail(f"{path} 没有 token")
    if array.dtype not in (np.float32, np.float16):
        fail(f"{path} 的 dtype 是 {array.dtype}，只接受 float32 或 float16")
    if not array.flags.c_contiguous:
        fail(f"{path} 不是 C 顺序，请另存为 C 顺序 npy")
    return array


EPS_REL = 1e-6
EIGEN_FLOOR = 1e-8


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _median(values: list[float]) -> float | None:
    finite = [value for value in values if np.isfinite(value)]
    if not finite:
        return None
    return float(np.median(np.asarray(finite, dtype=np.float64)))


def block_diagnostics(activation: np.ndarray, eps_rel: float = EPS_REL, eigen_floor: float = EIGEN_FLOOR) -> dict:
    """每块记录谱。条件数用未抬高的特征值。减均值只诊断，不进入白化。"""
    values = np.asarray(activation, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] % DSUB != 0:
        fail("诊断激活必须是二维，且列数能被 4 整除")
    n_tokens, width = values.shape
    groups = width // DSUB
    sigma = np.zeros((groups, DSUB, DSUB), dtype=np.float64)
    mean = np.zeros((groups, DSUB), dtype=np.float64)
    step = max(1, min(4096, (32 * 1024 * 1024) // max(width * 8, 1)))
    for start in range(0, n_tokens, step):
        block = np.asarray(values[start : start + step], dtype=np.float64)
        if not np.isfinite(block).all():
            fail("激活里有非有限数")
        view = block.reshape(block.shape[0], groups, DSUB)
        sigma += np.einsum("tmi,tmj->mij", view, view, optimize=True)
        mean += view.sum(axis=0)
    sigma /= float(n_tokens)
    sigma = 0.5 * (sigma + np.swapaxes(sigma, -1, -2))
    mean /= float(n_tokens)
    centered = sigma - mean[:, :, None] * mean[:, None, :]
    centered = 0.5 * (centered + np.swapaxes(centered, -1, -2))

    blocks = []
    for index in range(groups):
        raw = np.linalg.eigvalsh(sigma[index])
        centered_evals = np.linalg.eigvalsh(centered[index])
        lambda_max = float(raw[-1])
        lambda_min = float(raw[0])
        trace = float(np.trace(sigma[index]))
        eps = float(eps_rel * trace / DSUB)
        blocks.append(
            {
                "block": index,
                "lambda_max": lambda_max,
                "lambda_min": lambda_min,
                "condition": _finite_ratio(lambda_max, lambda_min),
                "eps": eps,
                "eps_over_lambda_min": _finite_ratio(eps, lambda_min),
                "truncated": bool(lambda_min < eps),
                "eigen_floor": float(eigen_floor),
                "floor_over_lambda_min": _finite_ratio(eigen_floor, lambda_min),
                "truncated_by_floor": bool(lambda_min < eigen_floor),
                "condition_centered": _finite_ratio(float(centered_evals[-1]), float(centered_evals[0])),
            }
        )
    conditions = [item["condition"] for item in blocks if item["condition"] is not None]
    centered_conditions = [item["condition_centered"] for item in blocks if item["condition_centered"] is not None]
    ratios = [item["eps_over_lambda_min"] for item in blocks if item["eps_over_lambda_min"] is not None]
    return {
        "centering": "白化仍用 E[xx^T]，不减均值。condition_centered 只诊断。",
        "eps_definition": "eps = 1e-6 * tr(Σ) / 4",
        "eigen_floor": eigen_floor,
        "eigen_floor_role": "当前白化代码把小于 1e-8 的特征值抬到 1e-8。这和相对 eps 是两件事。",
        "calibration_tokens": int(n_tokens),
        "blocks": blocks,
        "summary": {
            "blocks": groups,
            "condition_max": max(conditions) if conditions else None,
            "condition_median": _median(conditions),
            "condition_min": min(conditions) if conditions else None,
            "blocks_condition_above_1e3": sum(1 for value in conditions if value > 1e3),
            "condition_centered_max": max(centered_conditions) if centered_conditions else None,
            "condition_centered_median": _median(centered_conditions),
            "condition_centered_min": min(centered_conditions) if centered_conditions else None,
            "eps_over_lambda_min_max": max(ratios) if ratios else None,
            "eps_over_lambda_min_median": _median(ratios),
            "truncated_blocks": sum(1 for item in blocks if item["truncated"]),
            "truncated_by_floor_blocks": sum(1 for item in blocks if item["truncated_by_floor"]),
        },
    }


def second_moment(activation: np.ndarray) -> np.ndarray:
    n_tokens, width = activation.shape
    groups = width // DSUB
    acc = np.zeros((groups, DSUB, DSUB), dtype=np.float64)
    step = max(1, min(4096, (32 * 1024 * 1024) // max(width * 8, 1)))
    for start in range(0, n_tokens, step):
        block = np.asarray(activation[start : start + step], dtype=np.float64)
        if not np.isfinite(block).all():
            fail("激活里有非有限数")
        view = block.reshape(block.shape[0], groups, DSUB)
        acc += np.einsum("tmi,tmj->mij", view, view, optimize=True)
    acc /= float(n_tokens)
    return 0.5 * (acc + np.swapaxes(acc, -1, -2))


def whitening_factors(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    evals, evecs = np.linalg.eigh(sigma)
    evals = np.maximum(evals, 1e-8)
    sqrt_e = np.sqrt(evals)
    return evecs, sqrt_e, 1.0 / sqrt_e


def diagonal_factors(sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只用 Σ 的对角元做逐通道缩放，不做块内旋转。"""
    groups = sigma.shape[0]
    eye = np.zeros((groups, DSUB, DSUB), dtype=np.float64)
    index = np.arange(DSUB)
    eye[:, index, index] = 1.0
    diag = np.maximum(np.diagonal(sigma, axis1=-2, axis2=-1), EIGEN_FLOOR)
    sqrt_e = np.sqrt(diag)
    return eye, sqrt_e, 1.0 / sqrt_e


def apply_subspace(matrix: np.ndarray, evecs: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    groups = cols // DSUB
    values = np.asarray(matrix, dtype=np.float64).reshape(rows, groups, DSUB)
    rotated = np.einsum("nmi,mij->nmj", values, evecs)
    rotated *= scales[None, :, :]
    restored = np.einsum("nmj,mij->nmi", rotated, evecs)
    return np.ascontiguousarray(restored.reshape(rows, cols))


def row_rms(weight: np.ndarray) -> np.ndarray:
    squared = np.asarray(weight, dtype=np.float64) ** 2
    scale = np.sqrt(np.mean(squared, axis=1, keepdims=True))
    return np.maximum(scale, 1e-8)


def mse_indices(n_tokens: int, mse_tokens: int) -> np.ndarray:
    if mse_tokens >= n_tokens:
        return np.arange(n_tokens, dtype=np.int64)
    picked = np.rint(np.linspace(0, n_tokens - 1, mse_tokens)).astype(np.int64)
    return np.unique(picked)


def small_row_relative(weight: np.ndarray, hat: np.ndarray) -> tuple[float, int]:
    norms = np.linalg.norm(weight, axis=1)
    order = np.argsort(norms, kind="mergesort")
    take = max(weight.shape[0] // 4, 1)
    chosen = order[:take]
    err = np.linalg.norm(weight[chosen] - hat[chosen], axis=1)
    rel = err / np.maximum(norms[chosen], 1e-8)
    return float(np.mean(rel)), int(take)


def output_error_split(weight: np.ndarray, hat: np.ndarray, activation: np.ndarray, indices: np.ndarray) -> dict:
    """把输出误差拆成块内部分和跨块交叉项。交叉项可以为负。"""
    delta = np.asarray(weight - hat, dtype=np.float64)
    rows, cols = delta.shape
    groups = cols // DSUB
    delta_view = np.ascontiguousarray(delta.reshape(rows, groups, DSUB))
    within = 0.0
    total = 0.0
    count = 0
    bytes_per = rows * groups * 8
    step = max(1, min(32, (256 * 1024 * 1024) // max(bytes_per, 1)))
    for cursor in range(0, indices.size, step):
        chosen = indices[cursor : cursor + step]
        block = np.asarray(activation[chosen], dtype=np.float64).reshape(-1, groups, DSUB)
        dots = np.einsum("bgi,rgi->brg", block, delta_view, optimize=True)
        total += float(np.sum(np.square(dots.sum(axis=2)), dtype=np.float64))
        within += float(np.sum(np.square(dots), dtype=np.float64))
        count += int(dots.shape[0] * dots.shape[1])
    if count == 0:
        fail("没有可用于误差分解的 token")
    within_mse = within / count
    cross_mse = (total - within) / count
    output = within_mse + cross_mse
    share = (lambda part: None if output == 0.0 else part / output)
    return {
        "within_mse": within_mse,
        "cross_mse": cross_mse,
        "within_share": share(within_mse),
        "cross_share": share(cross_mse),
    }


def output_mse(weight: np.ndarray, hat: np.ndarray, activation: np.ndarray, indices: np.ndarray) -> float:
    delta = np.ascontiguousarray(weight - hat, dtype=np.float32)
    acc = 0.0
    count = 0
    step = max(1, min(64, (16 * 1024 * 1024) // max(weight.shape[0] * 4, 1)))
    for cursor in range(0, indices.size, step):
        chosen = indices[cursor : cursor + step]
        block = np.asarray(activation[chosen], dtype=np.float32)
        diff = block @ delta.T
        acc += float(np.sum(np.square(diff, dtype=np.float64)))
        count += diff.size
    if count == 0:
        fail("没有可用于输出 MSE 的 token")
    return acc / count


def faiss_reconstruct(work: np.ndarray, niter: int, seed: int, min_points: int | None):
    faiss = require_faiss()
    rows, width = work.shape
    if width % DSUB != 0:
        fail("列数不能被 4 整除")
    groups = width // DSUB
    pq = faiss.ProductQuantizer(width, groups, NBITS)
    pq.cp.niter = int(niter)
    pq.cp.seed = int(seed)
    pq.cp.verbose = False
    pq.train_type = faiss.ProductQuantizer.Train_default
    if min_points is not None:
        pq.cp.min_points_per_centroid = int(min_points)
    train_x = np.ascontiguousarray(work, dtype=np.float32)
    pq.train(train_x)
    codes = pq.compute_codes(train_x)
    decoded = pq.decode(codes)
    if codes.dtype != np.uint8 or codes.shape != (rows, groups):
        fail("compute_codes 的形状或类型不对")
    if decoded.shape != train_x.shape:
        fail("重建矩阵的形状不对")
    return codes, decoded, int(pq.cp.min_points_per_centroid)


def prepare_work(weight, use_rms, use_whiten, rms, evecs, sqrt_scales):
    work = np.array(weight, dtype=np.float32, copy=True)
    if use_rms:
        work /= rms.astype(np.float32)
    if use_whiten:
        work = np.ascontiguousarray(apply_subspace(work, evecs, sqrt_scales), dtype=np.float32)
    return work


def restore(decoded, use_rms, use_whiten, rms, evecs, inv_scales):
    hat = np.asarray(decoded, dtype=np.float64)
    if use_whiten:
        hat = apply_subspace(hat, evecs, inv_scales)
    if use_rms:
        hat = hat * rms
    return hat


def quantize_matrix(weight, activation, *, niter, seed, mse_tokens, min_points, full_size: bool):
    if weight.shape[1] != activation.shape[1]:
        fail("权重列数和激活列数不一致")
    if weight.shape[1] % DSUB != 0:
        fail("列数不能被 4 整除")
    sigma = second_moment(activation)
    spectrum = block_diagnostics(activation)
    evecs, sqrt_scales, inv_scales = whitening_factors(sigma)
    diag_evecs, diag_sqrt, diag_inv = diagonal_factors(sigma)
    rms = row_rms(weight)
    indices = mse_indices(activation.shape[0], mse_tokens)
    reports = []
    min_used = MIN_POINTS_DEFAULT
    jobs = [(name, use_rms, use_whiten, False) for name, use_rms, use_whiten in CASES]
    jobs.append(("对角白化", False, True, True))
    for name, use_rms, use_whiten, use_diagonal in jobs:
        factors = (diag_evecs, diag_sqrt, diag_inv) if use_diagonal else (evecs, sqrt_scales, inv_scales)
        print(f"  训练 {name}，niter={niter}，seed={seed}", flush=True)
        started = time.perf_counter()
        work = prepare_work(
            weight,
            use_rms,
            use_whiten,
            rms,
            factors[0],
            factors[1],
        )
        _codes, decoded, min_used = faiss_reconstruct(work, niter, seed, min_points)
        hat = restore(decoded, use_rms, use_whiten, rms, factors[0], factors[2])
        if not np.isfinite(hat).all():
            fail(f"{name} 的重建含有非有限数")
        rel, take = small_row_relative(weight, hat)
        mse = output_mse(weight, hat, activation, indices)
        split = output_error_split(weight, hat, activation, indices)
        elapsed = time.perf_counter() - started
        reports.append(
            {
                "name": name,
                "row_rms": use_rms,
                "subspace_weighting": use_whiten,
                "small_row_relative": rel,
                "small_row_count": take,
                "output_mse": mse,
                "within_share": split["within_share"],
                "cross_share": split["cross_share"],
                "diagonal_whiten": use_diagonal,
                "train_seconds": elapsed,
                "full_size": full_size,
            }
        )
        print(f"  {name}  小行相对误差 {rel:.4f}  输出 MSE {mse:.4f}", flush=True)
    spectrum["eval_tokens"] = int(indices.size)
    spectrum["eval_is_subset_of_calibration"] = True
    return reports, int(min_used), int(indices.size), spectrum


def sample_note(rows: int, min_points: int) -> str | None:
    recommended = KSUB * min_points
    if rows >= recommended:
        return None
    return (
        f"行数 {rows} 低于 Faiss 默认建议 {recommended}。"
        "单层 down_proj 是 4096 行，两层若分开训练仍然各是 4096 行。"
        "本脚本不把多层拼成一个码本，也不自动加到第三层。"
    )


def write_report(path: Path, payload: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    text = render_report(payload)
    (path / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (path / "report.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    print(f"报告已写入 {path / 'report.json'}", flush=True)


def render_report(payload: dict) -> str:
    lines = [
        payload["banner"],
        NOTE,
        f"faiss-cpu {payload['faiss']}",
        f"k-means niter {payload['kmeans_niter']}，seed {payload['seed']}",
        f"判定 S3：否",
        "",
    ]
    for block in payload["matrices"]:
        lines.append(
            f"层 {block['layer']}  {block['role']}_proj  {block['rows']}×{block['cols']}  "
            f"子空间 {block['subspaces']}  磁盘 dtype {block['disk_dtype']}"
        )
        lines.append(
            f"校准样本 {block['sigma_tokens']}，评估样本 {block['mse_tokens']}"
            + ("，评估集是校准集的子集" if block.get("eval_is_subset_of_calibration") else "")
        )
        summary = block.get("spectrum_summary")
        if summary:
            def num(key: str) -> str:
                value = summary.get(key)
                return "无" if value is None else f"{value:.3e}"

            lines.append(
                "每块谱：条件数最大 "
                f"{num('condition_max')}，中位 {num('condition_median')}，最小 {num('condition_min')}；"
                f"减均值后中位 {num('condition_centered_median')}（只诊断）；"
                f"eps/λmin 最大 {num('eps_over_lambda_min_max')}，中位 {num('eps_over_lambda_min_median')}；"
                f"相对 eps 截断 {summary['truncated_blocks']} 块，"
                f"绝对下限 1e-8 截断 {summary['truncated_by_floor_blocks']} 块。"
            )
        if block.get("sample_note"):
            lines.append(block["sample_note"])
        lines.append(f"{'路径':<12} {'小行相对误差':>14} {'输出 MSE':>14}")
        for item in block["cases"]:
            lines.append(
                f"{item['name']:<12} {item['small_row_relative']:14.4f} {item['output_mse']:14.4f}"
            )
            if item.get("within_share") is not None:
                lines.append(
                    f"  块内 {item['within_share']:.1%}，交叉项 {item['cross_share']:.1%}"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_real(args) -> int:
    if args.weights is None or args.activations_dir is None:
        fail("全尺寸运行需要 --weights 和 --activations-dir。云端没有权重时只跑 --self-check。")
    weights_dir = args.weights.expanduser()
    activations_dir = args.activations_dir.expanduser()
    if not weights_dir.is_dir():
        fail(f"权重目录不存在：{weights_dir}")
    if not activations_dir.is_dir():
        fail(f"激活目录不存在：{activations_dir}")
    layers = parse_layers(args.layers)
    roles = parse_roles(args.roles)
    located = []
    for layer in layers:
        for role in roles:
            name = tensor_name(layer, role)
            shard = find_shard(weights_dir, name)
            dtype, shape = tensor_meta(shard, name)
            require_llama_shape(role, shape)
            act_path = activation_file(activations_dir, layer, role)
            located.append((layer, role, name, shard, dtype, shape, act_path))
    print("权重和激活的形状符合 Llama 3 8B 单层 FFN。", flush=True)
    for layer, role, name, shard, dtype, shape, act_path in located:
        print(f"  层 {layer} {role}_proj {shape} {dtype} <- {shard.name}，激活 {act_path.name}", flush=True)
    if args.check_inputs:
        print("只检查了输入，没有训练。", flush=True)
        return 0
    faiss = require_faiss()
    if args.mse_tokens <= 0:
        fail("--mse-tokens 要是正整数。默认 256。")
    if args.mse_tokens > 4096:
        print("输出 MSE 的 token 数超过 4096，CPU 上会很慢。", file=sys.stderr)
    if args.kmeans_niter < 1:
        fail("--kmeans-niter 至少为 1")
    matrices = []
    loaded_acts: dict[Path, np.ndarray] = {}
    for layer, role, name, shard, dtype, shape, act_path in located:
        print(f"读取 {name}", flush=True)
        weight, disk_dtype = read_tensor(shard, name)
        require_llama_shape(role, tuple(weight.shape))
        if act_path not in loaded_acts:
            loaded_acts[act_path] = load_activation(act_path, weight.shape[1])
        activation = loaded_acts[act_path]
        print(
            f"层 {layer} {role}_proj，{weight.shape[0]}×{weight.shape[1]}，四组开始。"
            f"Faiss {faiss.__version__}。",
            flush=True,
        )
        cases, min_used, used_mse, spectrum = quantize_matrix(
            weight,
            activation,
            niter=args.kmeans_niter,
            seed=args.seed,
            mse_tokens=args.mse_tokens,
            min_points=None,
            full_size=True,
        )
        matrices.append(
            {
                "layer": layer,
                "role": role,
                "tensor": name,
                "rows": int(weight.shape[0]),
                "cols": int(weight.shape[1]),
                "subspaces": int(weight.shape[1] // DSUB),
                "disk_dtype": disk_dtype,
                "sigma_tokens": int(activation.shape[0]),
                "mse_tokens": used_mse,
                "eval_is_subset_of_calibration": True,
                "min_points_per_centroid": min_used,
                "sample_note": sample_note(int(weight.shape[0]), min_used),
                "spectrum_summary": spectrum["summary"],
                "blocks": spectrum["blocks"],
                "cases": cases,
            }
        )
    payload = {
        "scale": "llama3_8b_ffn_layer",
        "full_size": True,
        "judges_s3": False,
        "banner": "全尺寸离线 PQ。Llama 3 8B 的 FFN 层，Faiss ProductQuantizer.train 与 compute_codes。",
        "faiss": faiss.__version__,
        "kmeans_niter": args.kmeans_niter,
        "seed": args.seed,
        "weights": str(weights_dir),
        "activations_dir": str(activations_dir),
        "complete_layer": roles == ["gate", "up", "down"],
        "matrices": matrices,
        "note": NOTE,
    }
    write_report(args.out, payload)
    return 0


def _write_safetensors(path: Path, arrays: dict[str, tuple[np.ndarray, str]]) -> None:
    blobs = []
    header = {}
    cursor = 0
    for name, (array, dtype) in arrays.items():
        if dtype == "F32":
            raw = np.ascontiguousarray(array, dtype="<f4").tobytes()
        elif dtype == "F16":
            raw = np.ascontiguousarray(array, dtype="<f2").tobytes()
        elif dtype == "BF16":
            bits = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32) >> 16
            raw = bits.astype("<u2").tobytes()
        else:
            fail(f"自检不写 {dtype}")
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [cursor, cursor + len(raw)]}
        blobs.append(raw)
        cursor += len(raw)
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"".join(blobs))


def _assert_close(actual, expected, what: str) -> None:
    if not np.allclose(actual, expected, rtol=1e-5, atol=1e-5):
        fail(f"接口自检失败：{what}")


def run_self_check(args) -> int:
    faiss = require_faiss()
    print(
        "这是 faiss-cpu 接口自检，使用缩小随机矩阵，不是 Llama 3 8B 全尺寸，不能写成全尺寸结论，也不能判定 S3。",
        flush=True,
    )
    weight = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    rms = row_rms(weight)
    _assert_close(rms[0, 0], np.sqrt(12.5), "行 RMS")
    tokens = np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    sigma = second_moment(tokens)
    _assert_close(sigma[0], np.diag([1.0, 0.0, 0.0, 0.0]), "二阶矩")
    known = block_diagnostics(tokens)
    known_block = known["blocks"][0]
    _assert_close(known_block["lambda_max"], 1.0, "λmax")
    _assert_close(known_block["eps"], 2.5e-7, "相对 eps")
    if known_block["condition"] is not None or known_block["condition_centered"] is not None:
        fail("常数激活减均值后不应再有条件数；不减均值的 λmin 为 0")
    if not known_block["truncated"] or not known_block["truncated_by_floor"]:
        fail("λmin 为 0 时应记为截断")
    evecs, sqrt_e, inv_e = whitening_factors(np.stack([np.diag([4.0, 1.0, 1.0, 1.0])]))
    sample = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    roundtrip = apply_subspace(apply_subspace(sample, evecs, sqrt_e), evecs, inv_e)
    _assert_close(roundtrip, sample, "Σ 的平方根与逆")

    scratch = args.out / "self_check_tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    name = tensor_name(0, "gate")
    f32 = np.arange(8, dtype=np.float32).reshape(2, 4)
    f16 = np.array([[0.5, -2.0], [1.0, 3.5]], dtype=np.float16)
    bf16 = np.array([[1.0, -2.0], [0.5, 3.5]], dtype=np.float32)
    _write_safetensors(scratch / "f32.safetensors", {"f32": (f32, "F32")})
    _write_safetensors(scratch / "f16.safetensors", {"f16": (f16, "F16")})
    _write_safetensors(scratch / "bf16.safetensors", {"bf16": (bf16, "BF16")})
    got32, _ = read_tensor(scratch / "f32.safetensors", "f32")
    got16, _ = read_tensor(scratch / "f16.safetensors", "f16")
    gotbf, _ = read_tensor(scratch / "bf16.safetensors", "bf16")
    _assert_close(got32, f32, "F32 读写")
    _assert_close(got16, f16.astype(np.float32), "F16 读写")
    _assert_close(gotbf, bf16, "BF16 读写")
    try:
        from safetensors.numpy import load_file
    except ImportError:
        load_file = None
    if load_file is not None:
        official32 = load_file(str(scratch / "f32.safetensors"))
        official16 = load_file(str(scratch / "f16.safetensors"))
        _assert_close(official32["f32"], f32, "官方 safetensors 对读 F32")
        _assert_close(official16["f16"].astype(np.float32), f16.astype(np.float32), "官方 safetensors 对读 F16")
    _write_safetensors(scratch / "tiny.safetensors", {name: (np.zeros((8, 8), dtype=np.float32), "F32")})
    try:
        dtype, shape = tensor_meta(scratch / "tiny.safetensors", name)
        require_llama_shape("gate", shape)
    except SystemExit as exc:
        if exc.code != 3:
            raise
    else:
        fail("小矩阵应被拒绝，不能当成 Llama 3 8B 全尺寸")
    del dtype

    rng = np.random.default_rng(11)
    synthetic_w = rng.normal(size=(512, 16)).astype(np.float32)
    synthetic_x = rng.normal(size=(64, 16)).astype(np.float32)
    synthetic_x[:, 0::4] *= 8.0
    cases, _min_used, used, spectrum = quantize_matrix(
        synthetic_w,
        synthetic_x,
        niter=2,
        seed=7,
        mse_tokens=32,
        min_points=1,
        full_size=False,
    )
    names = [item["name"] for item in cases]
    if names[:4] != [name for name, _rms, _whiten in CASES] or "对角白化" not in names:
        fail("四组没有写全，或缺少对角白化")
    if any(item.get("within_share") is None or item.get("cross_share") is None for item in cases):
        fail("误差分解没有写下块内部分和交叉项")
    if any(not np.isfinite(item["small_row_relative"]) or not np.isfinite(item["output_mse"]) for item in cases):
        fail("自检指标不是有限数")
    if used < 1:
        fail("自检没有使用 token")
    if spectrum["summary"]["blocks"] != 4 or "eps_over_lambda_min" not in spectrum["blocks"][0]:
        fail("复测记录缺每块的 eps/λmin")
    if spectrum["eval_tokens"] != used or spectrum["calibration_tokens"] != 64:
        fail("校准样本数和评估样本数没有分开记")
    payload = {
        "scale": "interface_self_check",
        "full_size": False,
        "judges_s3": False,
        "banner": "faiss-cpu 接口自检。缩小随机矩阵，不是 Llama 3 8B 全尺寸，不能判定 S3。",
        "faiss": faiss.__version__,
        "kmeans_niter": 2,
        "seed": 7,
        "weights": None,
        "activations_dir": None,
        "complete_layer": False,
        "matrices": [
            {
                "layer": None,
                "role": "synthetic",
                "tensor": None,
                "rows": 512,
                "cols": 16,
                "subspaces": 4,
                "disk_dtype": "F32",
                "sigma_tokens": 64,
                "mse_tokens": used,
                "eval_is_subset_of_calibration": True,
                "min_points_per_centroid": 1,
                "spectrum_summary": spectrum["summary"],
                "blocks": spectrum["blocks"],
                "sample_note": "接口自检把 min_points_per_centroid 设为 1，全尺寸运行不会这样做。",
                "cases": cases,
            }
        ],
        "note": NOTE,
    }
    if payload["full_size"] or payload["judges_s3"]:
        fail("自检被标成了全尺寸或 S3")
    write_report(args.out, payload)
    print(
        "接口自检结束。上面的数字来自缩小随机矩阵，不是 Llama 3 8B 全尺寸，不能判定 S3。",
        flush=True,
    )
    return 0


def _load_diagnose_matrix(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        array = np.load(path)
        return np.asarray(array)
    raw = np.fromfile(path, dtype=np.int32, count=2)
    if raw.size != 2:
        fail(f"无法读取 {path}")
    rows, cols = int(raw[0]), int(raw[1])
    data = np.fromfile(path, dtype=np.float64, offset=8)
    if data.size != rows * cols:
        fail(f"{path} 的长度和形状 {rows}×{cols} 不一致")
    return data.reshape(rows, cols)


def _print_spectrum(spectrum: dict) -> None:
    summary = spectrum["summary"]

    def num(key: str) -> str:
        value = summary.get(key)
        return "无" if value is None else f"{value:.3e}"

    print(spectrum["centering"])
    print(f"eps 定义：{spectrum['eps_definition']}。{spectrum['eigen_floor_role']}")
    print(f"校准样本 {spectrum['calibration_tokens']}")
    print(
        f"条件数最大 {num('condition_max')}，中位 {num('condition_median')}，最小 {num('condition_min')}，"
        f"大于 1e3 的块 {summary['blocks_condition_above_1e3']} / {summary['blocks']}"
    )
    print(
        f"减均值后条件数最大 {num('condition_centered_max')}，中位 {num('condition_centered_median')}，"
        f"最小 {num('condition_centered_min')}。这一列不改白化。"
    )
    print(
        f"eps/λmin 最大 {num('eps_over_lambda_min_max')}，中位 {num('eps_over_lambda_min_median')}。"
        f"相对 eps 截断 {summary['truncated_blocks']} 块，绝对下限截断 {summary['truncated_by_floor_blocks']} 块。"
    )
    if summary["blocks"] <= 32:
        print(f"{'块':>4} {'λmax':>12} {'λmin':>12} {'条件数':>12} {'eps':>12} {'eps/λmin':>12} {'截断':>4} {'减均值条件数':>12}")
        for item in spectrum["blocks"]:
            def cell(value: float | None) -> str:
                return "无" if value is None else f"{value:.3e}"

            print(
                f"{item['block']:4d} {cell(item['lambda_max']):>12} {cell(item['lambda_min']):>12} "
                f"{cell(item['condition']):>12} {cell(item['eps']):>12} {cell(item['eps_over_lambda_min']):>12} "
                f"{'是' if item['truncated'] else '否':>4} {cell(item['condition_centered']):>12}"
            )


def run_diagnose(path: Path, out: Path | None) -> int:
    spectrum = block_diagnostics(_load_diagnose_matrix(path))
    _print_spectrum(spectrum)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(spectrum, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"每块记录已写入 {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pq_faiss_layer.py",
        description="用 Faiss ProductQuantizer 对 Llama 3 8B 的一层 FFN 做全尺寸离线 PQ。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--weights", type=Path, help="HF safetensors 目录，排除 original/")
    parser.add_argument("--activations-dir", type=Path, help="校准激活目录")
    parser.add_argument("--layers", default="0", help="层号，默认 0。最多两层，例如 0,1")
    parser.add_argument("--roles", default="gate,up,down", help="默认 gate,up,down")
    parser.add_argument("--out", type=Path, default=Path("out/pq_faiss_layer"))
    parser.add_argument("--mse-tokens", type=int, default=256, help="输出 MSE 使用的 token 数，默认 256")
    parser.add_argument("--kmeans-niter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--check-inputs", action="store_true", help="只核对形状，不训练")
    parser.add_argument("--self-check", action="store_true", help="云端接口自检，不是全尺寸结论")
    parser.add_argument("--diagnose", type=Path, help="只打印激活的每块谱，不训练。接受 npy 或 pq_layer 的 bin")
    parser.add_argument("--diagnose-out", type=Path, help="把 --diagnose 的每块记录写成 json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.diagnose is not None:
        return run_diagnose(args.diagnose, args.diagnose_out)
    if args.self_check:
        if args.out == Path("out/pq_faiss_layer"):
            args.out = Path("out/pq_faiss_self_check")
        print("自检不读取 --weights。没有使用 Llama 3 8B 权重。", flush=True)
        return run_self_check(args)
    return run_real(args)


if __name__ == "__main__":
    raise SystemExit(main())
