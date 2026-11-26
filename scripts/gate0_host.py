#!/usr/bin/env python3
"""Gate 0 基线测量入口。只在测量主机执行。

测量主机是晨涧云 RTX 4090 48G（计算能力 8.9）。本脚本不下载权重，不安装依赖。
1.2 ms/token 是方案里 H100 的原文口径，不是这张卡的阈值。这里不判断通过或不通过。

用法：

  python3 scripts/gate0_host.py --list
  python3 scripts/gate0_host.py --check-env
  python3 scripts/gate0_host.py --execute --baseline bf16 \\
      --model-dir /路径/Meta-Llama-3-8B \\
      --prompts /路径/prompt.txt \\
      --out out/gate0

除 bf16 以外，还要 --quant-dir 指向测量主机上已经准备好的量化权重。
注意力 W4 + FFN 原方案还没有权重，不能测。
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

BASELINES = (
    ("bf16", "BF16", "质量与性能参照", True),
    ("te-fp8", "TE FP8", "稠密 FP8 上限参照，硬件支持时才测", True),
    ("awq", "AWQ 4-bit", "成熟低比特方案", True),
    ("gptq", "GPTQ 4-bit", "另一套成熟 4-bit 方案", True),
    ("gguf-q4km", "GGUF Q4_K_M", "低显存部署参照", True),
    ("gguf-q3k", "GGUF Q3_K", "更低比特的 GGUF 参照", True),
    ("aqlm-1x16", "AQLM 1×16", "2-bit 质量参照", True),
    ("aqlm-2x8", "AQLM 2×8", "LUT 查表路线的精度和速度参照", True),
    ("attn-w4", "注意力 W4 + FFN 原方案", "FFN 原方案还没有权重", False),
)
PREFILL_LEN = 2048
GEN_LEN = 128
NOTE = "这不是通过或不通过的结论。1.2 ms/token 是方案里 H100 的原文口径，不是本机阈值。"


def fail(message: str, code: int = 2) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)


def gpu_query() -> dict | None:
    if shutil.which("nvidia-smi") is None:
        return None
    command = [
        "nvidia-smi",
        "--query-gpu=name,compute_cap,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    line = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        return None
    return {"name": parts[0], "compute_capability": parts[1], "driver": parts[2]}


def require_gpu() -> dict:
    info = gpu_query()
    if info is None:
        fail("本机没有 GPU，Gate 0 不能在这里测量")
    return info


def print_list() -> int:
    print("Gate 0 待测矩阵。同一检查点、同一 tokenizer、同一 prompt 文件、同一硬件。")
    print("先 warmup，默认重复 3 次，报告中位数和最小/最大。吞吐和延迟都要。")
    print(f"prefill 与 decode 分开。批大小 1。prefill 长度 {PREFILL_LEN}。生成长度 {GEN_LEN}。do_sample=False。")
    print("记录 GPU 名称、计算能力、驱动、CUDA、框架版本、模型目录、量化工具版本。")
    print("不在这里判断达标。")
    print()
    for _key, title, purpose, runnable in BASELINES:
        state = "可在测量主机执行" if runnable else "未实现，不能测"
        print(f"- {title}：{purpose}。{state}。")
    print()
    print(NOTE)
    return 0


def print_env() -> int:
    info = require_gpu()
    print(f"GPU {info['name']}，计算能力 {info['compute_capability']}，驱动 {info['driver']}。")
    major, _, minor = info["compute_capability"].partition(".")
    if major == "8" and minor == "9":
        print("计算能力 8.9 是 Ada，例如 RTX 4090。")
    print(NOTE)
    return 0


def median_span(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def measure_bf16(model_dir: Path, prompts: Path, repeats: int, warmup: int) -> dict:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        fail(f"缺少依赖，未测量：{exc.name}。不要在这台机器上安装，交给测量主机。")
    if not torch.cuda.is_available():
        fail("本机没有 GPU，Gate 0 不能在这里测量")
    text = prompts.read_text(encoding="utf-8").strip()
    if not text:
        fail("prompt 文件是空的")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda().eval()
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"].cuda()
    if input_ids.shape[1] > PREFILL_LEN:
        input_ids = input_ids[:, :PREFILL_LEN]
    attention = torch.ones_like(input_ids)
    torch.cuda.reset_peak_memory_stats()

    def once(record: bool) -> tuple[float, float] | None:
        prefill_start = torch.cuda.Event(enable_timing=True)
        prefill_end = torch.cuda.Event(enable_timing=True)
        decode_end = torch.cuda.Event(enable_timing=True)
        with torch.inference_mode():
            prefill_start.record()
            out = model(input_ids=input_ids, attention_mask=attention, use_cache=True)
            prefill_end.record()
            past = out.past_key_values
            next_id = out.logits[:, -1:].argmax(dim=-1)
            for _ in range(GEN_LEN):
                step = model(input_ids=next_id, past_key_values=past, use_cache=True)
                past = step.past_key_values
                next_id = step.logits[:, -1:].argmax(dim=-1)
            decode_end.record()
        torch.cuda.synchronize()
        if not record:
            return None
        prefill_ms = prefill_start.elapsed_time(prefill_end)
        decode_ms = prefill_end.elapsed_time(decode_end)
        return prefill_ms, decode_ms

    for _ in range(warmup):
        once(False)
    prefill_ms: list[float] = []
    decode_ms: list[float] = []
    for _ in range(repeats):
        timing = once(True)
        if timing is None:
            fail("测量没有返回时间")
        prefill_ms.append(timing[0])
        decode_ms.append(timing[1])
    prompt_tokens = int(input_ids.shape[1])
    prefill_tok_s = [prompt_tokens / (ms / 1000.0) for ms in prefill_ms]
    decode_tok_s = [GEN_LEN / (ms / 1000.0) for ms in decode_ms]
    peak = int(torch.cuda.max_memory_allocated())
    return {
        "measured": True,
        "prompt_tokens": prompt_tokens,
        "prefill_ms": median_span(prefill_ms),
        "decode_ms": median_span(decode_ms),
        "prefill_tok_s": median_span(prefill_tok_s),
        "decode_tok_s": median_span(decode_tok_s),
        "peak_vram_bytes": peak,
        "framework": f"torch {torch.__version__}",
    }


def execute(args) -> int:
    if args.baseline is None or args.model_dir is None or args.prompts is None:
        fail("测量需要 --baseline、--model-dir 和 --prompts")
    known = {item[0] for item in BASELINES}
    if args.baseline not in known:
        fail("未知的 baseline")
    if args.baseline == "attn-w4":
        fail("注意力 W4 + FFN 原方案未实现，不能测")
    info = require_gpu()
    if args.baseline != "bf16" and args.quant_dir is None:
        fail("除 BF16 外需要 --quant-dir 指向测量主机上已有的量化权重。不要在脚本里下载。")
    if not args.model_dir.is_dir():
        fail(f"模型目录不存在：{args.model_dir}")
    if not args.prompts.is_file():
        fail(f"prompt 文件不存在：{args.prompts}")
    if args.baseline != "bf16":
        if args.quant_dir is None or not args.quant_dir.is_dir():
            fail("量化目录不存在。不要在脚本里下载。")
        fail(f"{args.baseline} 需要测量主机上的对应加载器。当前只实现了本地 BF16 测量，这一项未测。")
    result = measure_bf16(args.model_dir, args.prompts, args.repeats, args.warmup)
    payload = {
        "baseline": "bf16",
        "model_dir": str(args.model_dir),
        "quant_dir": None,
        "prompts": str(args.prompts),
        "gpu": info,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "prefill_len": PREFILL_LEN,
        "gen_len": GEN_LEN,
        "batch": 1,
        "do_sample": False,
        "measured": True,
        "note": NOTE,
        **result,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "report.txt").write_text(
        f"BF16 已测量。prefill tok/s 中位数 {result['prefill_tok_s']['median']:.3f}，"
        f"decode tok/s 中位数 {result['decode_tok_s']['median']:.3f}。\n{NOTE}\n",
        encoding="utf-8",
    )
    print((args.out / "report.txt").read_text(encoding="utf-8"), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gate 0 基线测量入口，供 RTX 4090 测量主机执行。")
    parser.add_argument("--list", action="store_true", help="打印待测矩阵，不测量")
    parser.add_argument("--check-env", action="store_true", help="查看本机 GPU")
    parser.add_argument("--execute", action="store_true", help="在有 GPU 和本地权重时才测量")
    parser.add_argument("--baseline", choices=[item[0] for item in BASELINES])
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--quant-dir", type=Path)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--out", type=Path, default=Path("out/gate0"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected = sum(bool(flag) for flag in (args.list, args.check_env, args.execute))
    if selected != 1:
        fail("请在 --list、--check-env、--execute 里选一个")
    if args.repeats < 3:
        fail("重复次数至少 3 次")
    if args.warmup < 0:
        fail("warmup 不能是负数")
    if args.list:
        return print_list()
    if args.check_env:
        return print_env()
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
