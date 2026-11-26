#!/usr/bin/env python3
"""从本地 Llama 3 8B 导出一层 FFN 的激活，供 pq_faiss_layer.py 使用。

不下载权重。云端没有模型，只跑 --self-check。自检写入缩小随机矩阵，不是全尺寸，也不能判定 S3。

测量主机：

  python3 scripts/export_layer_activations.py \\
      --weights /路径/Meta-Llama-3-8B \\
      --prompts /路径/prompt.txt \\
      --layers 0 \\
      --max-tokens 256 \\
      --out /路径/校准激活

写出：
  layer0.gate_up.npy    float32，C 顺序，[token 数, 4096]
  layer0.down.npy       float32，C 顺序，[token 数, 14336]

gate_up 是进入 MLP 的隐藏状态。down 是 down_proj 的输入，也就是 silu(gate) * up。
默认最多 256 个 token，不是方案里的 512×2048 全量校准。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

HIDDEN = 4096
INTERMEDIATE = 14336


def fail(message: str, code: int = 2) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def write_matrix(path: Path, array: np.ndarray) -> np.ndarray:
    matrix = np.ascontiguousarray(array, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        fail("激活必须是非空二维矩阵")
    if not np.isfinite(matrix).all():
        fail(f"{path.name} 含有非有限数")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, matrix)
    return matrix


def save_layer(out: Path, layer: int, gate_up: np.ndarray, down: np.ndarray, *, enforce_llama: bool) -> None:
    if enforce_llama:
        if gate_up.shape[1] != HIDDEN or down.shape[1] != INTERMEDIATE:
            fail(
                f"激活列数是 {gate_up.shape[1]} 和 {down.shape[1]}，"
                f"Llama 3 8B 需要 {HIDDEN} 和 {INTERMEDIATE}。拒绝写成全尺寸激活。"
            )
        if gate_up.shape[0] != down.shape[0]:
            fail("gate/up 与 down 的 token 数不一致")
    write_matrix(out / f"layer{layer}.gate_up.npy", gate_up)
    write_matrix(out / f"layer{layer}.down.npy", down)


def load_pq_module():
    path = Path(__file__).resolve().parent / "pq_faiss_layer.py"
    spec = importlib.util.spec_from_file_location("pq_faiss_layer", path)
    if spec is None or spec.loader is None:
        fail("找不到 pq_faiss_layer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_self_check(out: Path) -> int:
    print("这是激活文件格式自检，使用缩小随机矩阵，不是 Llama 3 8B 全尺寸，不能判定 S3。", flush=True)
    rng = np.random.default_rng(7)
    gate_up = rng.normal(size=(5, 8)).astype(np.float32)
    down = rng.normal(size=(5, 12)).astype(np.float32)
    save_layer(out, 0, gate_up, down, enforce_llama=False)
    module = load_pq_module()
    loaded_up = module.load_activation(out / "layer0.gate_up.npy", 8)
    loaded_down = module.load_activation(out / "layer0.down.npy", 12)
    if loaded_up.shape != (5, 8) or loaded_down.shape != (5, 12):
        fail("导出的 npy 不能被 PQ 脚本读回")
    if not np.allclose(loaded_up, gate_up) or not np.allclose(loaded_down, down):
        fail("读回的激活和写出的不一致")
    try:
        save_layer(out, 1, gate_up, down, enforce_llama=True)
    except SystemExit as exc:
        if exc.code != 2:
            raise
    else:
        fail("缩小矩阵不应被接受为 Llama 3 8B 全尺寸激活")
    print("格式自检通过。没有读取 Llama 权重。上面的数组不是全尺寸结果。", flush=True)
    return 0


def collect_layer(model, tokenizer, prompts: list[str], layer_index: int, max_tokens: int, device: str):
    import torch

    layer = model.model.layers[layer_index]
    gate_parts: list[np.ndarray] = []
    down_parts: list[np.ndarray] = []
    seen = 0

    def take(bucket: list[np.ndarray], tensor) -> None:
        nonlocal seen
        flat = tensor.detach().to("cpu", dtype=torch.float32).reshape(-1, tensor.shape[-1]).numpy()
        remain = max_tokens - seen
        if remain <= 0:
            return
        bucket.append(np.ascontiguousarray(flat[:remain]))

    def on_mlp(_module, inputs):
        take(gate_parts, inputs[0])

    def on_down(_module, inputs):
        nonlocal seen
        before = seen
        take(down_parts, inputs[0])
        if down_parts:
            seen = before + down_parts[-1].shape[0]

    hook_mlp = layer.mlp.register_forward_pre_hook(on_mlp)
    hook_down = layer.mlp.down_proj.register_forward_pre_hook(on_down)
    try:
        for text in prompts:
            if seen >= max_tokens:
                break
            encoded = tokenizer(text, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                model(**encoded, use_cache=False)
    finally:
        hook_mlp.remove()
        hook_down.remove()
    if not gate_parts or not down_parts:
        fail(f"第 {layer_index} 层没有采集到激活")
    return np.concatenate(gate_parts, axis=0), np.concatenate(down_parts, axis=0)


def export(args) -> int:
    if not args.weights.is_dir():
        fail(f"模型目录不存在：{args.weights}。不要下载权重。")
    if not args.prompts.is_file():
        fail(f"prompt 文件不存在：{args.prompts}")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        name = getattr(exc, "name", None) or "torch/transformers"
        fail(f"缺少依赖，未导出：{name}。测量主机再安装。不要在云端下载权重。")
    prompts = [line.strip() for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        fail("prompt 文件是空的")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"从本地目录导出，设备 {device}，最多 {args.max_tokens} 个 token。这不是 512×2048 的全量校准。", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.weights, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.weights,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device).eval()
    layers = [int(part) for part in args.layers.split(",") if part.strip()]
    if not layers or len(layers) > 2 or any(layer < 0 or layer > 31 for layer in layers):
        fail("层号要在 0 到 31，最多两层")
    for layer in layers:
        gate_up, down = collect_layer(model, tokenizer, prompts, layer, args.max_tokens, device)
        save_layer(args.out, layer, gate_up, down, enforce_llama=True)
        print(
            f"层 {layer}：gate_up {gate_up.shape}，down {down.shape}，已写入 {args.out}",
            flush=True,
        )
    print("导出完成。把这个目录传给 pq_faiss_layer.py 的 --activations-dir。本步不能判定 S3。", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出 Llama 3 8B 一层 FFN 的激活。不下载权重。")
    parser.add_argument("--self-check", action="store_true", help="只检查 npy 格式，不读模型")
    parser.add_argument("--weights", type=Path, help="本地 HF 模型目录")
    parser.add_argument("--prompts", type=Path, help="本地 prompt，一行一条")
    parser.add_argument("--layers", default="0")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("out/activations_self_check"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        if args.out == Path("out/activations_self_check"):
            args.out = Path("out/activations_self_check")
        print("自检不读取 --weights。", flush=True)
        return run_self_check(args.out)
    if args.weights is None or args.prompts is None:
        fail("导出需要 --weights 和 --prompts。云端没有权重时只跑 --self-check。")
    if args.max_tokens < 1:
        fail("--max-tokens 至少为 1")
    if args.max_tokens > 8192:
        print("token 数超过 8192，CPU 导出会很慢，而且仍不是方案里的全量校准。", file=sys.stderr)
    return export(args)


if __name__ == "__main__":
    raise SystemExit(main())
