#!/usr/bin/env python3
"""第 1 周 W3：在 1–2 层真实 FFN 上做离线 PQ 四组对照。

只在目标主机上由开发总管执行。本脚本：
- 只用 faiss.ProductQuantizer.train / compute_codes；
- 不调用 search()，不建 IndexIVFPQ / GpuIndex；
- 不下载权重或校准语料，不索要 token；
- 不安装 AWQ/GPTQ/AQLM。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

# 保证可从任意 cwd 导入同目录模块
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Σ 分组的线性/FFN MSE 若用校准激活计算，必须原样写入日志、元数据与报告。
SIGMA_OUTPUT_MSE_CAVEAT = "偏乐观，不得当契约数"

import numpy as np

from pq_io import (
    check_frozen_env,
    check_gpu_policy,
    dump_json,
    enforce_offline_hub,
    expected_weight_shape,
    fail,
    linear_weight_name,
    list_groups,
    load_hf_config,
    load_json,
    load_linear_weight,
    pq_m_for_dim,
    resolve_model_dir,
    warn,
    weight_map,
)
from pq_math import (
    apply_column_scale_activations,
    apply_column_scale_weight,
    apply_row_rms,
    build_rotation_spec,
    channel_rms_from_sumsq,
    finalize_column_scale,
    finalize_sigma,
    init_sigma_accum,
    accumulate_sigma,
    reconstructed_in_rot_domain,
    rot_domain_to_original,
    rotate_activations,
    rotate_weight_rows,
    row_rms,
    store_row_scale_f16,
    to_faiss_matrix,
    unrotate_weight_rows,
    undo_column_scale_weight,
)
from pq_metrics import mse_report, per_row_relative_error, matvec_rows, shift_kl_from_logits
from pq_calib import (
    capture_ffn_inputs,
    ffn_forward_numpy,
    load_causal_lm,
    load_texts,
    mlp_modules,
    role_activation_key,
    tokenize_texts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="W3 单层离线 PQ 四组对照（只写结果，不编造数字）")
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "default.json")
    p.add_argument(
        "--groups",
        type=str,
        default="all",
        help="对照 id，逗号分隔：plain,row_rms,sigma,row_rms_sigma，或 all",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="覆盖配置中的 device，正式路径仅允许 cuda:0；禁止 cpu",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只检查模型/语料路径与配置，不训练、不编码，也不做 GPU/冻结环境硬检",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="用合成小矩阵检查数学可逆性（可选极小 Faiss 冒烟）；不做 GPU/冻结环境硬检",
    )
    p.add_argument("--skip-kl", action="store_true", help="跳过单层 KL（仍可算重建与输出 MSE）")
    p.add_argument(
        "--weights-only",
        action="store_true",
        help="退化通路：不读校准激活，关闭列缩放与 Σ 加权，只报旋转域重建 MSE",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="覆盖结果目录")
    return p.parse_args()


def selected_group_ids(raw: str) -> Optional[list[str]]:
    text = (raw or "all").strip()
    if text.lower() == "all" or text == "":
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def prepare_output_dirs(cfg: dict, override: Optional[Path]) -> tuple[Path, Path]:
    if override is not None:
        results = override
    else:
        results = Path(cfg.get("output", {}).get("results_dir") or "results")
        if not results.is_absolute():
            results = ROOT / results
    logs = Path(cfg.get("output", {}).get("logs_dir") or "logs")
    if not logs.is_absolute():
        logs = ROOT / logs
    results.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return results, logs


def run_self_check() -> int:
    """不依赖真实权重。失败即非零退出。"""
    from pq_math import (
        fwht_normalized,
        make_sign_vector,
        unwhiten_weight_rows,
        whiten_weight_rows,
        apply_row_rms,
        undo_row_rms,
    )

    errors: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        if cond:
            print(f"  PASS  {name}")
        else:
            errors.append(f"{name}: {detail}")
            print(f"  FAIL  {name}  {detail}")

    rng = np.random.RandomState(20261126)

    x = rng.randn(8, 16)
    twice = fwht_normalized(fwht_normalized(x))
    check("FWHT 两次回到自身", np.allclose(twice, x, atol=1e-10, rtol=1e-10), f"max_abs={np.max(np.abs(twice-x))}")

    # 整段旋转：W @ x == W_rot @ x_rot
    w = rng.randn(5, 16)
    spec = build_rotation_spec(16, seed=11, block_size=16)
    w_rot = rotate_weight_rows(w, spec)
    x_in = rng.randn(7, 16)
    x_rot = rotate_activations(x_in, spec)
    y = x_in @ w.T
    y2 = x_rot @ w_rot.T
    check("旋转后点积守恒 (16)", np.allclose(y, y2, atol=1e-10, rtol=1e-10), f"max_abs={np.max(np.abs(y-y2))}")
    w_back = unrotate_weight_rows(w_rot, spec)
    check("旋转可逆 (16)", np.allclose(w_back, w, atol=1e-10, rtol=1e-10), f"max_abs={np.max(np.abs(w_back-w))}")

    # down 式分块：7 × 16
    dim = 7 * 16
    w = rng.randn(4, dim)
    spec = build_rotation_spec(dim, seed=22, block_size=16)
    x_in = rng.randn(6, dim)
    w_rot = rotate_weight_rows(w, spec)
    x_rot = rotate_activations(x_in, spec)
    y = x_in @ w.T
    y2 = x_rot @ w_rot.T
    check("分块 FWHT 点积守恒 (7×16)", np.allclose(y, y2, atol=1e-10, rtol=1e-10), f"max_abs={np.max(np.abs(y-y2))}")
    w_back = unrotate_weight_rows(w_rot, spec)
    check("分块 FWHT 可逆 (7×16)", np.allclose(w_back, w, atol=1e-10, rtol=1e-10), f"max_abs={np.max(np.abs(w_back-w))}")

    # 列缩放
    s = np.linspace(0.2, 3.0, 16)
    w = rng.randn(3, 16)
    x_in = rng.randn(5, 16)
    w_s = apply_column_scale_weight(w, s)
    x_s = apply_column_scale_activations(x_in, s)
    check("列缩放点积守恒", np.allclose(x_in @ w.T, x_s @ w_s.T, atol=1e-10, rtol=1e-10))
    check("列缩放可逆", np.allclose(undo_column_scale_weight(w_s, s), w, atol=1e-12, rtol=1e-12))

    # 行 RMS + FP16 存储
    w = rng.randn(9, 16) * np.linspace(0.1, 5.0, 9)[:, None]
    scales = store_row_scale_f16(row_rms(w))
    w_n = apply_row_rms(w, scales)
    w_r = undo_row_rms(w_n, scales)
    check("行 RMS 按存储尺度可逆", np.allclose(w_r, w, atol=1e-3, rtol=1e-3), f"max_abs={np.max(np.abs(w_r-w))}")
    # 归一化后每行 RMS 约为 1（相对 FP16 精度）
    r = row_rms(w_n)
    check("行 RMS 归一化后接近 1", np.allclose(r, 1.0, atol=5e-3, rtol=5e-3), f"rms={r}")

    # 白化可逆 + 马氏距离恒等
    # 构造 SPD Σ
    a = rng.randn(4, 4)
    sigma = a @ a.T + 0.2 * np.eye(4)
    accum = init_sigma_accum(8, dsub=4)  # two subspaces, fill both with same family
    # 直接注入 finalize 所需字段
    factors = finalize_sigma(
        {
            "dsub": 4,
            "dim": 8,
            "sum_xx": np.stack([sigma * 10.0, sigma * 10.0], axis=0),
            "n_tokens": 10,
        },
        eig_floor=1e-8,
    )
    w = rng.randn(6, 8)
    w_white = whiten_weight_rows(w, factors)
    w_back = unwhiten_weight_rows(w_white, factors)
    check("Σ^{1/2} / Σ^{-1/2} 可逆", np.allclose(w_back, w, atol=1e-9, rtol=1e-9), f"max_abs={np.max(np.abs(w_back-w))}")
    dw = rng.randn(4)
    mahal = float(dw @ sigma @ dw)
    # factors['sqrt'][0] should be sigma^{1/2} (n_tokens cancelled)
    white = dw @ factors["sqrt"][0]
    euc = float(white @ white)
    # finalize used mean_xx = sum/n = sigma, good
    check("马氏距离等于白化后 L2", np.allclose(mahal, euc, atol=1e-9, rtol=1e-9), f"{mahal} vs {euc}")

    # 符号向量确定性
    check("符号种子可复现", np.array_equal(make_sign_vector(32, 20261126), make_sign_vector(32, 20261126)))

    # 列缩放 + 旋转往返应回到原始权重（与训练路径 rot_domain_to_original 一致）
    s = finalize_column_scale(np.linspace(0.01, 4.0, 16), median_normalize=True, clamp_min=0.05, clamp_max=20.0)
    w = rng.randn(5, 16)
    spec = build_rotation_spec(16, seed=33, block_size=16)
    w_rot = rotate_weight_rows(apply_column_scale_weight(w, s), spec)
    w_back = rot_domain_to_original(w_rot, s, spec)
    check("列缩放+旋转往返", np.allclose(w_back, w, atol=1e-10, rtol=1e-10), f"max_abs={np.max(np.abs(w_back-w))}")

    # Faiss 冒烟：本盒若未装 faiss-cpu 则跳过，不算失败。目标主机上应能跑通。
    try:
        import faiss as _faiss_mod  # type: ignore  # noqa: F401
        faiss_available = True
    except ImportError:
        faiss_available = False
    if not faiss_available:
        print("  SKIP  Faiss 冒烟（当前进程没有 faiss-cpu；目标主机锁定环境应已安装 1.16.0）")
    else:
        try:
            from pq_faiss_cpu import CpuProductQuantizer, import_faiss_cpu, assert_no_forbidden_symbols_used

            assert_no_forbidden_symbols_used()
            faiss, ver, dist = import_faiss_cpu(forbid_gpu=True, expected_version_prefix="1.16")
            print(f"  INFO  导入 Faiss version={ver} dist={dist}")
            x = np.ascontiguousarray(rng.randn(64, 8).astype(np.float32))
            pq = CpuProductQuantizer(d=8, m=2, nbits=8, seed=1, niter=5, nredo=1, verbose=False, faiss_mod=faiss)
            pq.train(x)
            codes = pq.compute_codes(x)
            cents = pq.centroids()
            check(
                "Faiss train/compute_codes 形状",
                codes.shape == (64, 2) and cents.shape == (2, 256, 4),
                f"{codes.shape} {cents.shape}",
            )
            check("封装不含 search 方法", not hasattr(pq, "search"))
        except SystemExit as exc:
            errors.append(f"Faiss 冒烟失败: {exc}")
            print(f"  FAIL  Faiss 冒烟  {exc}")
        except Exception as exc:
            errors.append(f"Faiss 冒烟异常: {exc}")
            print(f"  FAIL  Faiss 冒烟  {exc}")

    if errors:
        print("SELF-CHECK 失败：")
        for e in errors:
            print(" -", e)
        return 1
    print("SELF-CHECK 全部通过。")
    return 0


def _slice_enc(enc, sl):
    return {k: v[sl] for k, v in enc.items() if hasattr(v, "__getitem__")}


def capture_batched(model, enc, layers, device, capture_logits, batch_size):
    n = int(enc["input_ids"].shape[0])
    bs = int(batch_size) if batch_size else n
    acts_all = {i: {"gate_up": [], "down": []} for i in layers}
    logits_parts = []
    mask_parts = []
    for start in range(0, n, bs):
        sl = slice(start, min(start + bs, n))
        sub = _slice_enc(enc, sl)
        print(f"INFO: 前向批次 {start}:{min(start+bs, n)} / {n}")
        acts, logits, mask = capture_ffn_inputs(model, sub, layers, device, capture_logits)
        for layer in layers:
            acts_all[layer]["gate_up"].append(acts[layer]["gate_up"])
            acts_all[layer]["down"].append(acts[layer]["down"])
        if capture_logits:
            logits_parts.append(logits)
        mask_parts.append(mask)
    merged = {}
    for layer in layers:
        merged[layer] = {
            "gate_up": np.concatenate(acts_all[layer]["gate_up"], axis=0),
            "down": np.concatenate(acts_all[layer]["down"], axis=0),
        }
    logits = None
    if capture_logits:
        import torch

        logits = torch.cat(logits_parts, dim=0)
    mask = np.concatenate(mask_parts, axis=0)
    return merged, logits, mask


def rotation_for_role(role: str, dim: int, rot_cfg: dict):
    if not rot_cfg.get("enabled", True):
        return None
    seeds = rot_cfg.get("sign_seeds") or {}
    if role in ("gate", "up"):
        seed = int(seeds.get("gate_up", 20261126))
        block = dim
    elif role == "down":
        seed = int(seeds.get("down", 20261127))
        block = int(rot_cfg.get("down_block_size") or 2048)
    else:
        fail(f"未知角色 {role}")
    return build_rotation_spec(dim, seed=seed, block_size=block)


def compute_column_scale(x: np.ndarray, col_cfg: dict) -> np.ndarray:
    sum_sq = np.sum(np.square(np.asarray(x, dtype=np.float64)), axis=0)
    raw = channel_rms_from_sumsq(sum_sq, int(x.shape[0]), alpha=float(col_cfg.get("alpha", 1.0)))
    return finalize_column_scale(
        raw,
        median_normalize=bool(col_cfg.get("median_normalize", True)),
        clamp_min=float(col_cfg.get("clamp_min", 0.05)),
        clamp_max=float(col_cfg.get("clamp_max", 20.0)),
    )


def maybe_require_paths(cfg: dict, model_dir_ok: bool, calib_ok: bool, eval_ok: bool, weights_only: bool) -> int:
    print(f"DRY-RUN 模型目录: {'OK' if model_dir_ok else '缺失'}")
    print(f"DRY-RUN 校准语料: {'OK' if calib_ok else '缺失（待准备）'}")
    print(f"DRY-RUN 评测语料: {'OK' if eval_ok else '缺失（待准备）'}")
    if not model_dir_ok:
        return 2
    if weights_only:
        return 0
    if not calib_ok:
        print("DRY-RUN: 校准语料未就绪。正式跑四组对照前必须本地放置，脚本不会下载。")
        return 3
    return 0


def train_one_matrix(
    weight: np.ndarray,
    x_in: Optional[np.ndarray],
    role: str,
    layer: int,
    group: dict,
    cfg: dict,
    faiss_mod,
    artifacts_dir: Path,
    weights_only: bool,
) -> dict:
    from pq_faiss_cpu import CpuProductQuantizer, configure_threads

    col_cfg = cfg.get("column_scale") or {}
    rot_cfg = cfg.get("rotation") or {}
    pq_cfg = cfg.get("pq") or {}
    row_cfg = cfg.get("row_rms") or {}
    sig_cfg = cfg.get("sigma_weight") or {}
    dsub = int(pq_cfg.get("dsub", 4))
    n, d = weight.shape
    use_row = bool(group["row_rms"])
    use_sig = bool(group["sigma_weight"])

    if use_sig and (x_in is None or weights_only):
        fail(f"分组 {group['id']} 需要子空间 Σ 加权，但没有校准激活（待准备）")
    if col_cfg.get("enabled", True) and not weights_only and x_in is None:
        fail("已启用列缩放但没有校准激活（待准备）")

    column_scale = None
    if col_cfg.get("enabled", True) and not weights_only:
        if x_in.shape[1] != d:
            fail(f"层 {layer} {role} 激活宽 {x_in.shape[1]} 与权重列 {d} 不一致")
        column_scale = compute_column_scale(x_in, col_cfg)
        w_s = apply_column_scale_weight(weight, column_scale)
        x_s = apply_column_scale_activations(x_in, column_scale)
    else:
        w_s = np.asarray(weight, dtype=np.float64)
        x_s = None if x_in is None else np.asarray(x_in, dtype=np.float64)

    rotation = rotation_for_role(role, d, rot_cfg) if rot_cfg.get("enabled", True) else None
    w_rot = rotate_weight_rows(w_s, rotation) if rotation is not None else w_s
    x_rot = None
    if x_s is not None and rotation is not None:
        x_rot = rotate_activations(x_s, rotation)
    elif x_s is not None:
        x_rot = x_s

    bundle_state = {"row_rms_eps": float(row_cfg.get("eps", 1e-12))}
    if use_row:
        scales = store_row_scale_f16(row_rms(w_rot, eps=float(row_cfg.get("eps", 1e-12))))
        bundle_state["row_scale_f16"] = scales
    else:
        scales = None

    sigma_factors = None
    if use_sig:
        accum = init_sigma_accum(d, dsub=dsub)
        accumulate_sigma(accum, x_rot)
        sigma_factors = finalize_sigma(accum, eig_floor=float(sig_cfg.get("eig_floor", 1e-8)))
        bundle_state["sigma_factors"] = sigma_factors
        print(
            f"INFO: 层 {layer} {role} 分组 {group['id']} Σ 估计 token={sigma_factors['n_tokens']}，"
            f"被下限截断的特征值个数={sigma_factors['n_floored_eigs']}"
        )

    faiss_mat = to_faiss_matrix(w_rot, use_row, use_sig, bundle_state)
    m = pq_m_for_dim(d, dsub)
    expected_m = {"gate": 1024, "up": 1024, "down": 3584}
    if d in (4096, 14336) and m != expected_m[role]:
        warn(f"层 {layer} {role} M={m}，官方 Llama 3 8B 期望 {expected_m[role]}")

    configure_threads(pq_cfg.get("omp_num_threads"))
    t0 = time.perf_counter()
    pq = CpuProductQuantizer(
        d=d,
        m=m,
        nbits=int(pq_cfg.get("nbits", 8)),
        seed=int(pq_cfg.get("seed", 1234)),
        niter=int(pq_cfg.get("niter", 20)),
        nredo=int(pq_cfg.get("nredo", 1)),
        max_points_per_centroid=int(pq_cfg.get("max_points_per_centroid", 256)),
        verbose=bool(pq_cfg.get("verbose", True)),
        faiss_mod=faiss_mod,
    )
    pq.train(faiss_mat)
    codes = pq.compute_codes(faiss_mat)
    centroids = pq.centroids()
    train_s = time.perf_counter() - t0
    print(f"INFO: 层 {layer} {role} 分组 {group['id']} Faiss train+codes 用时 {train_s:.1f}s，codes={codes.shape}")

    w_hat_rot = reconstructed_in_rot_domain(codes, centroids, use_row, use_sig, bundle_state)
    recon_rot = mse_report(w_rot, w_hat_rot)
    recon_rot.update(per_row_relative_error(w_rot, w_hat_rot))

    w_hat_orig = rot_domain_to_original(w_hat_rot, column_scale, rotation)
    w_orig = rot_domain_to_original(w_rot, column_scale, rotation)
    # 与原始输入权重对拍（应等于 weight，数值误差内）
    recon_orig = mse_report(weight, w_hat_orig)
    recon_orig.update(per_row_relative_error(weight, w_hat_orig))
    consistency = mse_report(weight, w_orig)
    if consistency["max_abs"] > 1e-5:
        warn(
            f"层 {layer} {role} 前置变换往返误差 max_abs={consistency['max_abs']:.3e}，"
            "请检查列缩放 / 旋转实现（不应在此量级）。"
        )

    linear_out = None
    linear_caveat = None
    if x_in is not None:
        y_ref = matvec_rows(weight, x_in)
        y_hat = matvec_rows(w_hat_orig, x_in)
        linear_out = mse_report(y_ref, y_hat)
        linear_out["activation_source"] = "calibration"
        if use_sig:
            linear_caveat = SIGMA_OUTPUT_MSE_CAVEAT
            linear_out["optimistic_not_for_contract"] = True
            linear_out["caveat"] = SIGMA_OUTPUT_MSE_CAVEAT
        else:
            linear_out["optimistic_not_for_contract"] = False
            linear_out["caveat"] = None
        extra = f"（{SIGMA_OUTPUT_MSE_CAVEAT}）" if use_sig else ""
        print(
            f"INFO: 层 {layer} {role} 分组 {group['id']} 线性输出 MSE={linear_out['mse']:.6e}  "
            f"来自校准激活{extra}；不预设门槛"
        )

    art = artifacts_dir / f"layer{layer}_{role}_{group['id']}"
    art.mkdir(parents=True, exist_ok=True)
    np.save(art / "codes_uint8.npy", codes)
    np.save(art / "centroids_faiss_fp64.npy", centroids)
    np.save(art / "weight_hat_orig_fp32.npy", w_hat_orig.astype(np.float32))
    if scales is not None:
        np.save(art / "row_scale_f16.npy", scales)
    if column_scale is not None:
        np.save(art / "column_scale_fp64.npy", column_scale)
    if sigma_factors is not None:
        np.save(art / "sigma_fp64.npy", sigma_factors["sigma"])
        dump_json(
            art / "sigma_meta.json",
            {
                "n_tokens": sigma_factors["n_tokens"],
                "n_floored_eigs": sigma_factors["n_floored_eigs"],
                "eig_floor": sigma_factors["eig_floor"],
            },
        )

    rec = {
        "layer": layer,
        "role": role,
        "group": group["id"],
        "row_rms": use_row,
        "sigma_weight": use_sig,
        "n_rows": n,
        "d": d,
        "M": m,
        "dsub": dsub,
        "train_seconds": train_s,
        "recon_rot": recon_rot,
        "recon_orig": recon_orig,
        "preproc_roundtrip_vs_raw": consistency,
        "linear_output": linear_out,
        "linear_output_caveat": linear_caveat,
        "sigma_n_floored_eigs": None if sigma_factors is None else sigma_factors["n_floored_eigs"],
        "weight_hat_orig": w_hat_orig,
        "artifact_dir": str(art),
    }
    return rec


def flatten_row(rec: dict) -> dict:
    row = {
        "layer": rec["layer"],
        "role": rec["role"],
        "group": rec["group"],
        "row_rms": rec["row_rms"],
        "sigma_weight": rec["sigma_weight"],
        "n_rows": rec["n_rows"],
        "d": rec["d"],
        "M": rec["M"],
        "train_seconds": rec["train_seconds"],
    }
    for prefix, blob in (
        ("recon_rot", rec["recon_rot"]),
        ("recon_orig", rec["recon_orig"]),
        ("linear_output", rec["linear_output"]),
    ):
        if blob is None:
            continue
        for k, v in blob.items():
            row[f"{prefix}_{k}"] = v
    row["sigma_n_floored_eigs"] = rec["sigma_n_floored_eigs"]
    row["linear_output_caveat"] = rec.get("linear_output_caveat")
    return row


def swap_linear_weight(module, new_np: np.ndarray):
    import torch

    orig = module.weight.data.clone()
    tensor = torch.from_numpy(np.ascontiguousarray(new_np, dtype=np.float32))
    tensor = tensor.to(device=module.weight.device, dtype=module.weight.dtype)
    if tensor.shape != module.weight.shape:
        fail(f"替换权重形状 {tuple(tensor.shape)} 与模块 {tuple(module.weight.shape)} 不一致")
    with torch.no_grad():
        module.weight.copy_(tensor)
    return orig


def restore_weight(module, orig) -> None:
    import torch

    with torch.no_grad():
        module.weight.copy_(orig)


def main() -> int:
    args = parse_args()
    # 尽早设线程，再 import faiss（self-check / 训练时）
    cfg = load_json(args.config.resolve())
    threads = (cfg.get("pq") or {}).get("omp_num_threads")
    if threads:
        os.environ.setdefault("OMP_NUM_THREADS", str(int(threads)))

    enforce_offline_hub()
    results_dir, _logs_dir = prepare_output_dirs(cfg, args.output_dir)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.self_check:
        return run_self_check()

    env_cfg = cfg.get("environment") or {}
    groups = list_groups(cfg, selected_group_ids(args.groups))
    layers = list(cfg.get("layers") or [0, 16])
    roles = list(cfg.get("roles") or ["gate", "up", "down"])
    for r in roles:
        if r not in ("gate", "up", "down"):
            fail(f"非法 roles：{r}")
    if not layers:
        fail("配置 layers 为空")

    device = args.device or (cfg.get("runtime") or {}).get("device") or "cuda:0"
    refuse = list(env_cfg.get("refuse_gpu_names") or ["Tesla P4", "Tesla-P4"])
    expect = str(env_cfg.get("expected_gpu_substr") or "4090")
    req_cc = env_cfg.get("required_compute_capability") or [8, 9]
    req_cc_t = (int(req_cc[0]), int(req_cc[1]))

    model_cfg = cfg.get("model") or {}
    model_path = Path(model_cfg.get("path") or "/data/models/Meta-Llama-3-8B")
    calib_cfg = cfg.get("calibration") or {}
    eval_cfg = cfg.get("evaluation") or {}
    calib_path = Path(calib_cfg.get("path") or "")
    eval_path = Path(eval_cfg.get("path") or "")
    eval_fallback = bool(eval_cfg.get("fallback_to_calib", False))

    model_ok = model_path.is_dir()
    calib_ok = calib_path.exists() if calib_path.name else False
    eval_ok = eval_path.exists() if eval_path.name else False

    print("==== W3 单层离线 PQ ====")
    print(f"配置: {args.config}")
    print(f"层: {layers}  角色: {roles}  分组: {[g['id'] for g in groups]}")
    print(f"模型: {model_path}  设备: {device}")
    print(f"校准: {calib_path}  评测: {eval_path}")
    print("说明: 本实验不能单独用来判定 S3（S3 要等同体积全模型对照）。本周不做全模型蒸馏。")

    if args.dry_run:
        if model_ok:
            try:
                hf = load_hf_config(resolve_model_dir(str(model_path)))
                print(
                    f"INFO: hidden={hf.get('hidden_size')} intermediate={hf.get('intermediate_size')} "
                    f"layers={hf.get('num_hidden_layers')}"
                )
            except SystemExit:
                raise
            except Exception as exc:
                warn(f"读取 config.json 失败：{exc}")
        return maybe_require_paths(cfg, model_ok, calib_ok, eval_ok, args.weights_only)

    # 正式路径硬检（不安装任何包）。--self-check / --dry-run 已在上文返回，不走这里。
    env_report = check_frozen_env(env_cfg)
    gpu_name = check_gpu_policy(
        refuse_names=refuse,
        expected_substr=expect,
        device=device,
        required_capability=req_cc_t,
    )

    from pq_faiss_cpu import import_faiss_cpu, configure_threads, assert_no_forbidden_symbols_used

    assert_no_forbidden_symbols_used()
    configure_threads(threads)
    faiss_mod, faiss_ver, faiss_dist = import_faiss_cpu(
        forbid_gpu=bool(env_cfg.get("forbid_gpu_faiss", True)),
        expected_version_prefix="1.16",
    )
    print(f"INFO: faiss {faiss_ver} ({faiss_dist})")

    model_dir = resolve_model_dir(str(model_path))
    hf_config = load_hf_config(model_dir)
    n_layers_model = int(hf_config.get("num_hidden_layers", 0))
    for layer in layers:
        if layer < 0 or (n_layers_model and layer >= n_layers_model):
            fail(f"层号 {layer} 越界（num_hidden_layers={n_layers_model}）")
    mapping = weight_map(model_dir)

    need_sigma = any(bool(g["sigma_weight"]) for g in groups)
    need_acts = (not args.weights_only) and (
        bool((cfg.get("column_scale") or {}).get("enabled", True))
        or need_sigma
        or bool(eval_cfg.get("compute_layer_output_mse", True))
        or bool(eval_cfg.get("compute_ffn_output_mse", True))
        or (bool(eval_cfg.get("compute_single_layer_kl", True)) and not args.skip_kl)
    )
    if args.weights_only and need_sigma:
        fail("--weights-only 与 Σ 加权分组不能同时使用。请用 --groups plain,row_rms")
    if (cfg.get("column_scale") or {}).get("enabled", True) and args.weights_only:
        warn("--weights-only 已关闭列缩放（无校准激活）。旋转仍按配置进行。")

    activations = None
    teacher_logits = None
    token_mask = None
    eval_pack = None  # (enc, texts_desc)
    model = None

    if need_acts:
        if not calib_ok:
            fail(
                f"缺少校准语料：{calib_path}。列缩放、Σ 加权、输出 MSE 都需要本地激活。\n"
                "请按 configs/default.json 中 calibration 准备 JSONL 后重跑。"
                "本脚本不会下载数据，也不会索要 token。状态：待准备。"
            )
        texts = load_texts(
            calib_path,
            text_key=str(calib_cfg.get("text_key", "text")),
            n_sequences=int(calib_cfg.get("n_sequences", 32)),
            seed=int(calib_cfg.get("seed", 42)),
            sample=True,
        )
        _, enc = tokenize_texts(model_dir, texts, max_length=int(calib_cfg.get("max_length", 512)))
        model = load_causal_lm(model_dir, str(model_cfg.get("torch_dtype", "bfloat16")), device)
        want_kl_here = False  # 教师 logits 在评测集上另采
        activations, _, _ = capture_batched(
            model,
            enc,
            layers,
            device,
            capture_logits=want_kl_here,
            batch_size=int(calib_cfg.get("batch_size", 4)),
        )

    # 加载评测集（KL / 可选持出输出 MSE 仍用校准激活作为主报告，评测集只跑 KL）
    do_kl = (
        (not args.skip_kl)
        and (not args.weights_only)
        and bool(eval_cfg.get("compute_single_layer_kl", True))
    )
    if do_kl:
        kl_path = eval_path if eval_ok else (calib_path if eval_fallback and calib_ok else None)
        if kl_path is None:
            warn(
                f"评测语料不存在：{eval_path}，且 fallback_to_calib="
                f"{eval_fallback}。单层 KL 本跑次将标为「待准备」，不中止其它指标。"
            )
            do_kl = False
        else:
            if (not eval_ok) and eval_fallback:
                warn("评测语料缺失，KL 回退到校准语料。评测与校准未隔离，报告必须注明。")
            ev_texts = load_texts(
                kl_path,
                text_key=str(eval_cfg.get("text_key") or calib_cfg.get("text_key", "text")),
                n_sequences=int(eval_cfg.get("n_sequences", 16)),
                seed=int(eval_cfg.get("seed", 43)),
                sample=True,
            )
            if model is None:
                model = load_causal_lm(model_dir, str(model_cfg.get("torch_dtype", "bfloat16")), device)
            _, ev_enc = tokenize_texts(model_dir, ev_texts, max_length=int(eval_cfg.get("max_length", 512)))
            print("INFO: 抽取教师 logits（BF16 全模型，评测集）")
            # 只需 logits，不必再存一份激活
            ev_bs = int(eval_cfg.get("batch_size", 4))
            import torch

            parts = []
            masks = []
            n = ev_enc["input_ids"].shape[0]
            for start in range(0, n, ev_bs):
                sl = slice(start, min(start + ev_bs, n))
                sub = _slice_enc(ev_enc, sl)
                with torch.no_grad():
                    out = model(
                        input_ids=sub["input_ids"].to(device),
                        attention_mask=sub["attention_mask"].to(device),
                    )
                parts.append(out.logits.detach().to(dtype=torch.float32, device="cpu"))
                masks.append(sub["attention_mask"].bool().cpu().numpy())
            teacher_logits = torch.cat(parts, dim=0)
            token_mask = np.concatenate(masks, axis=0)
            eval_pack = ev_enc

    # 训练与重建
    records = []
    hats = {}  # (layer, role, group_id) -> W_hat_orig
    for layer in layers:
        for role in roles:
            w = load_linear_weight(model_dir, layer, role, mapping)
            expect_shape = expected_weight_shape(hf_config, role)
            if tuple(w.shape) != expect_shape:
                fail(
                    f"{linear_weight_name(layer, role)} 形状 {tuple(w.shape)} 与 config.json 期望 {expect_shape} 不符"
                )
            x_in = None
            if activations is not None:
                x_in = activations[layer][role_activation_key(role)]
            for group in groups:
                print(f"==== 训练 layer={layer} role={role} group={group['id']} ====")
                rec = train_one_matrix(
                    weight=w,
                    x_in=x_in,
                    role=role,
                    layer=layer,
                    group=group,
                    cfg=cfg,
                    faiss_mod=faiss_mod,
                    artifacts_dir=results_dir / "artifacts",
                    weights_only=bool(args.weights_only),
                )
                hats[(layer, role, group["id"])] = rec.pop("weight_hat_orig")
                records.append(rec)

    # FFN 输出 MSE：同一层三个角色都用同一分组的重建权重
    ffn_rows = []
    if (
        activations is not None
        and bool(eval_cfg.get("compute_ffn_output_mse", True))
        and set(roles) >= {"gate", "up", "down"}
    ):
        for layer in layers:
            x = activations[layer]["gate_up"]
            w_g = load_linear_weight(model_dir, layer, "gate", mapping)
            w_u = load_linear_weight(model_dir, layer, "up", mapping)
            w_d = load_linear_weight(model_dir, layer, "down", mapping)
            y_ref = ffn_forward_numpy(x, w_g, w_u, w_d)
            for group in groups:
                y_hat = ffn_forward_numpy(
                    x,
                    hats[(layer, "gate", group["id"])],
                    hats[(layer, "up", group["id"])],
                    hats[(layer, "down", group["id"])],
                )
                report = mse_report(y_ref, y_hat)
                report.update({"layer": layer, "group": group["id"]})
                report["activation_source"] = "calibration"
                if bool(group["sigma_weight"]):
                    report["optimistic_not_for_contract"] = True
                    report["caveat"] = SIGMA_OUTPUT_MSE_CAVEAT
                    extra = f"（{SIGMA_OUTPUT_MSE_CAVEAT}）"
                else:
                    report["optimistic_not_for_contract"] = False
                    report["caveat"] = None
                    extra = ""
                ffn_rows.append(report)
                print(
                    f"INFO: 层 {layer} 分组 {group['id']} FFN 输出 MSE={report['mse']:.6e}  "
                    f"cosine={report['cosine']:.6f}，来自校准激活{extra}；不预设门槛"
                )

    # 单层 KL：每次只替换一层的所选 FFN 角色
    kl_rows = []
    if do_kl and eval_pack is not None and model is not None:
        import torch

        mods = mlp_modules(model, layers)
        temp = float(eval_cfg.get("kl_temperature", 1.0))
        ev_bs = int(eval_cfg.get("batch_size", 4))
        for layer in layers:
            for group in groups:
                saved = {}
                try:
                    for role in roles:
                        saved[role] = swap_linear_weight(mods[layer][role], hats[(layer, role, group["id"])])
                    parts = []
                    n = eval_pack["input_ids"].shape[0]
                    for start in range(0, n, ev_bs):
                        sl = slice(start, min(start + ev_bs, n))
                        sub = _slice_enc(eval_pack, sl)
                        with torch.no_grad():
                            out = model(
                                input_ids=sub["input_ids"].to(device),
                                attention_mask=sub["attention_mask"].to(device),
                            )
                        parts.append(out.logits.detach().to(dtype=torch.float32, device="cpu"))
                    student = torch.cat(parts, dim=0)
                    kl = shift_kl_from_logits(teacher_logits, student, token_mask, temperature=temp)
                    kl.update({"layer": layer, "group": group["id"], "roles_replaced": ",".join(roles)})
                    kl_rows.append(kl)
                    print(
                        f"INFO: 层 {layer} 分组 {group['id']} 单层 KL="
                        f"{kl['kl_nats_per_token']:.6e} nats/token（n={kl['n_tokens']}，待写入，不预设门槛）"
                    )
                finally:
                    for role, orig in saved.items():
                        restore_weight(mods[layer][role], orig)

    # 落盘
    csv_path = results_dir / f"pq_single_layer_results_{stamp}.csv"
    fieldnames: list[str] = []
    flat = [flatten_row(r) for r in records]
    for row in flat:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in flat:
            writer.writerow(row)

    ffn_path = results_dir / f"pq_ffn_output_{stamp}.json"
    kl_path_out = results_dir / f"pq_single_layer_kl_{stamp}.json"
    meta = {
        "stamp": stamp,
        "config": str(args.config.resolve()),
        "model_path": str(model_dir),
        "hf_hidden_size": hf_config.get("hidden_size"),
        "hf_intermediate_size": hf_config.get("intermediate_size"),
        "hf_num_hidden_layers": hf_config.get("num_hidden_layers"),
        "layers": layers,
        "roles": roles,
        "groups": [g["id"] for g in groups],
        "device": device,
        "gpu_name": gpu_name,
        "env": env_report,
        "faiss_version": faiss_ver,
        "faiss_dist": faiss_dist,
        "weights_only": bool(args.weights_only),
        "calib_path": str(calib_path) if calib_ok else None,
        "eval_path": str(eval_path) if eval_ok else None,
        "eval_fallback_used": bool(do_kl and (not eval_ok) and eval_fallback),
        "s3_note": "本实验不能单独用来判定 S3（S3 要等同体积全模型对照）。",
        "sigma_group_output_mse_caveat": (
            "Σ 分组（sigma / row_rms_sigma）的线性/FFN MSE 用校准激活计算，"
            f"{SIGMA_OUTPUT_MSE_CAVEAT}。"
        ),
        "metrics_csv": str(csv_path),
        "ffn_output_json": str(ffn_path),
        "kl_json": str(kl_path_out),
        "linear_records_without_weights": [
            {k: v for k, v in flatten_row(r).items()} for r in records
        ],
        "ffn_output": ffn_rows,
        "single_layer_kl": kl_rows,
        "pending": [],
    }
    if not do_kl:
        meta["pending"].append("单层 KL：本跑次未计算（评测语料待准备，或已 --skip-kl / --weights-only）")
    if args.weights_only:
        meta["pending"].append("列缩放、Σ 加权、输出 MSE / KL：--weights-only 退化通路，非正式四组对照")
    if activations is None:
        meta["pending"].append("线性输出 MSE / FFN 输出 MSE：无校准激活，待准备")
    if any(bool(g["sigma_weight"]) for g in groups) and activations is not None:
        meta["pending"].append(
            "Σ 分组（sigma / row_rms_sigma）的线性/FFN MSE 用校准激活计算，"
            f"{SIGMA_OUTPUT_MSE_CAVEAT}。"
        )

    dump_json(ffn_path, ffn_rows)
    dump_json(kl_path_out, kl_rows)
    dump_json(results_dir / f"run_meta_{stamp}.json", meta)
    # 稳定文件名的最新副本，便于报告引用
    dump_json(results_dir / "run_meta_latest.json", meta)
    print(f"INFO: 结果已写入 {csv_path}")
    print(f"INFO: 元数据 {results_dir / 'run_meta_latest.json'}")
    print(
        "INFO: Σ 分组（sigma / row_rms_sigma）的线性/FFN MSE 用校准激活计算，"
        f"{SIGMA_OUTPUT_MSE_CAVEAT}。"
    )
    print("INFO: 报告模板 pq_single_layer_report.md 中的数字列请用本次实测回填，禁止手写估计值。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
