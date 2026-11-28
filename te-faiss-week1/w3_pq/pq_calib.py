#!/usr/bin/env python3
"""本地校准语料读取与 FFN 激活抽取。

语料必须已经在目标主机本地。缺失时给出清晰错误并标「待准备」。
绝不从网络拉取数据集，不索要或写入 token。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np

from pq_io import fail, warn


def load_texts(
    path: Path,
    text_key: str,
    n_sequences: Optional[int],
    seed: int,
    sample: bool = True,
) -> list[str]:
    if not path.exists():
        fail(
            f"校准/评测语料不存在：{path}\n"
            "  本实验不从网上拉取数据集，也不索要 Hugging Face token。\n"
            "  请在目标主机放入本地 JSONL（每行一个对象，默认字段 text），\n"
            "  或在配置中改 calibration.path / evaluation.path。在语料就绪前，\n"
            "  需要激活的列缩放、Σ 加权以及输出 MSE / 单层 KL 均标为「待准备」。"
        )
    texts = _read_all_texts(path, text_key)
    texts = [t.strip() for t in texts if t and t.strip()]
    if not texts:
        fail(f"{path} 中没有非空文本")
    if n_sequences is None or n_sequences <= 0 or not sample:
        chosen = texts if (n_sequences is None or n_sequences <= 0) else texts[: int(n_sequences)]
    elif n_sequences >= len(texts):
        if n_sequences > len(texts):
            warn(f"{path} 只有 {len(texts)} 条，少于请求的 {n_sequences} 条，将全部使用")
        chosen = texts
    else:
        rng = random.Random(int(seed))
        chosen = rng.sample(texts, int(n_sequences))
    print(f"INFO: 从 {path} 读取 {len(texts)} 条文本，使用 {len(chosen)} 条（seed={seed}）")
    return chosen


def _read_all_texts(path: Path, text_key: str) -> list[str]:
    if path.is_dir():
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".txt", ".jsonl", ".json"}
        )
        if not files:
            fail(f"校准目录 {path} 中没有 .txt / .jsonl / .json 文件")
        out: list[str] = []
        for p in files:
            out.extend(_read_all_texts(p, text_key))
        return out
    suf = path.suffix.lower()
    if suf == ".jsonl":
        return list(_iter_jsonl_texts(path, text_key))
    if suf == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            texts = []
            for i, obj in enumerate(payload):
                if isinstance(obj, str):
                    texts.append(obj)
                elif isinstance(obj, dict) and text_key in obj:
                    texts.append(str(obj[text_key]))
                else:
                    fail(f"{path} 第 {i} 项无法取到文本字段 {text_key}")
            return texts
        fail(f"{path} 需要 JSON 数组或改用 JSONL")
    if suf in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8").strip()
        return [text] if text else []
    fail(f"不支持的校准文件类型：{path}")
    return []


def _iter_jsonl_texts(path: Path, text_key: str):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{line_no} 不是合法 JSON：{exc}")
            if isinstance(obj, str):
                yield obj
            elif isinstance(obj, dict):
                if text_key not in obj:
                    fail(f"{path}:{line_no} 缺少字段「{text_key}」")
                text = obj[text_key]
                if not isinstance(text, str):
                    fail(f"{path}:{line_no} 字段 {text_key} 不是字符串")
                yield text
            else:
                fail(f"{path}:{line_no} 需要字符串或对象，得到 {type(obj).__name__}")


def tokenize_texts(model_dir: Path, texts: list[str], max_length: int):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        fail("未安装 transformers。请使用锁定环境中的 transformers 5.24.1。")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        # Llama 3 常用 eos 作为 pad；仅用于批内对齐，KL / 激活统计会 mask 掉 pad。
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=int(max_length),
        return_tensors="pt",
        add_special_tokens=True,
    )
    return tokenizer, enc


def _model_layers(model):
    root = model
    if hasattr(root, "model") and hasattr(root.model, "layers"):
        return root.model.layers
    if hasattr(root, "layers"):
        return root.layers
    fail("无法在模型上定位 layers（请确认是 Llama ForCausalLM）")


def mlp_modules(model, layers: list[int]) -> dict:
    blocks = _model_layers(model)
    out = {}
    n = len(blocks)
    for layer in layers:
        if layer < 0 or layer >= n:
            fail(f"层号 {layer} 越界，模型共 {n} 层")
        mlp = blocks[layer].mlp
        out[int(layer)] = {
            "gate": mlp.gate_proj,
            "up": mlp.up_proj,
            "down": mlp.down_proj,
            "mlp": mlp,
        }
    return out


def load_causal_lm(model_dir: Path, torch_dtype: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if torch_dtype not in dtype_map:
        fail(f"不支持的 torch_dtype：{torch_dtype}")
    print(
        f"INFO: 以 local_files_only 从 {model_dir} 加载模型，"
        f"dtype={torch_dtype}，device={device}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=dtype_map[torch_dtype],
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def capture_ffn_inputs(
    model,
    enc,
    layers: list[int],
    device: str,
    capture_logits: bool,
):
    """一次前向，抓取各层 gate/up 共享输入与 down 输入（仅有效 token），可选教师 logits。

    返回：
      activations[layer]["gate_up"]: float32 [T, hidden]
      activations[layer]["down"]:    float32 [T, intermediate]
      logits: float32 [B, S, V] 或 None
      token_mask: bool [B, S]
    """
    import torch

    mods = mlp_modules(model, layers)
    buckets: dict[int, dict[str, list]] = {i: {"gate_up": [], "down": []} for i in layers}
    hooks = []

    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    # [B, S]
    token_mask = attn.bool()

    def _valid_rows(hidden, batch_info=None):
        # hidden: [B, S, C] 或 [B*S, C]
        if hidden.dim() == 3:
            b, s, c = hidden.shape
            flat = hidden.reshape(b * s, c)
            mask = token_mask.reshape(b * s)
            return flat[mask]
        fail(f"意外的激活维数 {tuple(hidden.shape)}")

    def make_hook(layer: int, kind: str):
        def _hook(module, inputs, output=None):
            x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            rows = _valid_rows(x.detach()).to(dtype=torch.float32, device="cpu")
            buckets[layer][kind].append(rows.numpy())

        return _hook

    for layer, pack in mods.items():
        # gate 与 up 输入相同（post-attn RMSNorm 之后）；只挂 gate 避免加倍存储。
        hooks.append(pack["gate"].register_forward_pre_hook(make_hook(layer, "gate_up")))
        hooks.append(pack["down"].register_forward_pre_hook(make_hook(layer, "down")))

    logits = None
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
            if capture_logits:
                logits = out.logits.detach().to(dtype=torch.float32, device="cpu").contiguous()
    finally:
        for h in hooks:
            h.remove()

    activations = {}
    for layer in layers:
        gu = buckets[layer]["gate_up"]
        dn = buckets[layer]["down"]
        if not gu or not dn:
            fail(f"层 {layer} 未捕获到 FFN 激活；请检查 hook 是否挂到 LlamaMLP")
        activations[int(layer)] = {
            "gate_up": np.concatenate(gu, axis=0),
            "down": np.concatenate(dn, axis=0),
        }
        print(
            f"INFO: 层 {layer} 捕获 gate/up 输入 {activations[layer]['gate_up'].shape}，"
            f"down 输入 {activations[layer]['down'].shape}"
        )
    mask_cpu = token_mask.to(device="cpu").numpy()
    return activations, logits, mask_cpu


def role_activation_key(role: str) -> str:
    if role in ("gate", "up"):
        return "gate_up"
    if role == "down":
        return "down"
    fail(f"未知角色 {role}")


def silu(x: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    # 稳定 sigmoid
    out = np.empty_like(x64)
    pos = x64 >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x64[pos]))
    expx = np.exp(x64[~pos])
    out[~pos] = expx / (1.0 + expx)
    return x64 * out


def ffn_forward_numpy(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    """SwiGLU FFN：down(silu(gate(x)) * up(x))，权重均为 [out, in]，输入 [T, hidden]。"""
    x64 = np.asarray(x, dtype=np.float64)
    gate = x64 @ np.asarray(w_gate, dtype=np.float64).T
    up = x64 @ np.asarray(w_up, dtype=np.float64).T
    hidden = silu(gate) * up
    return hidden @ np.asarray(w_down, dtype=np.float64).T
