#!/usr/bin/env python3
"""重建 MSE、线性/FFN 输出 MSE、单层 KL。所有数字必须由实测写入，本模块不提供默认「门槛值」。"""

from __future__ import annotations

from typing import Optional

import numpy as np


def mse_report(ref: np.ndarray, hat: np.ndarray) -> dict:
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(hat, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"MSE 形状不一致：{a.shape} vs {b.shape}")
    diff = a - b
    mse = float(np.mean(np.square(diff)))
    ref_energy = float(np.mean(np.square(a)))
    rel = float(mse / ref_energy) if ref_energy > 0.0 else float("nan")
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "relative_mse": rel,
        "max_abs": float(np.max(np.abs(diff))) if diff.size else float("nan"),
        "cosine": _cosine(a.reshape(-1), b.reshape(-1)),
        "n_elements": int(a.size),
    }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def per_row_relative_error(ref: np.ndarray, hat: np.ndarray) -> dict:
    a = np.asarray(ref, dtype=np.float64)
    b = np.asarray(hat, dtype=np.float64)
    num = np.sum(np.square(a - b), axis=1)
    den = np.sum(np.square(a), axis=1)
    rel = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    finite = rel[np.isfinite(rel)]
    if finite.size == 0:
        return {"row_rel_mse_mean": float("nan"), "row_rel_mse_median": float("nan")}
    return {
        "row_rel_mse_mean": float(np.mean(finite)),
        "row_rel_mse_median": float(np.median(finite)),
    }


def matvec_rows(weight: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y = x @ W.T，W 为 [out, in]，x 为 [T, in]。"""
    return np.asarray(x, dtype=np.float64) @ np.asarray(weight, dtype=np.float64).T


def shift_kl_from_logits(
    teacher_logits: "object",
    student_logits: "object",
    token_mask: np.ndarray,
    temperature: float = 1.0,
) -> dict:
    """因果 LM 的 next-token 单层 KL：对位置 t 用 logits[t] 预测 token t+1。

    单位：nats / token（自然对数）。mask 为 True 的位置是有效 token；
    参与平均的是「当前位置与下一位置都有效」的 t。
    teacher / student: [B, S, V] torch 或 numpy。
    """
    import torch
    import torch.nn.functional as F

    if temperature <= 0.0:
        raise ValueError("KL 温度必须为正")

    def _to_tensor(x):
        if isinstance(x, torch.Tensor):
            return x.to(dtype=torch.float32)
        return torch.from_numpy(np.asarray(x)).to(dtype=torch.float32)

    t_log = _to_tensor(teacher_logits)
    s_log = _to_tensor(student_logits)
    if t_log.shape != s_log.shape:
        raise ValueError(f"教师/学生 logits 形状不一致：{tuple(t_log.shape)} vs {tuple(s_log.shape)}")
    if t_log.dim() != 3:
        raise ValueError(f"logits 应为 [B,S,V]，得到 {tuple(t_log.shape)}")

    # 因果移位：位置 0..S-2 的分布对应目标 1..S-1
    t_use = t_log[:, :-1, :] / float(temperature)
    s_use = s_log[:, :-1, :] / float(temperature)
    mask = torch.from_numpy(np.asarray(token_mask, dtype=bool))
    if mask.shape != t_log.shape[:2]:
        raise ValueError(f"mask 形状 {tuple(mask.shape)} 与 logits {tuple(t_log.shape[:2])} 不一致")
    pair_mask = mask[:, :-1] & mask[:, 1:]

    log_p = F.log_softmax(t_use, dim=-1)
    log_q = F.log_softmax(s_use, dim=-1)
    # KL(P||Q) = sum P (log P - log Q)
    p = log_p.exp()
    kl_tok = torch.sum(p * (log_p - log_q), dim=-1)  # [B, S-1]
    valid = pair_mask.to(dtype=torch.bool)
    n = int(valid.sum().item())
    if n <= 0:
        return {
            "kl_nats_per_token": float("nan"),
            "n_tokens": 0,
            "temperature": float(temperature),
            "unit": "nats_per_token",
        }
    kl_mean = float(kl_tok[valid].mean().item())
    kl_median = float(kl_tok[valid].median().item())
    return {
        "kl_nats_per_token": kl_mean,
        "kl_nats_per_token_median": kl_median,
        "n_tokens": n,
        "temperature": float(temperature),
        "unit": "nats_per_token",
    }
