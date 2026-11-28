#!/usr/bin/env python3
"""Measure device-to-device copy bandwidth on the official 4090 24GB host.

Replaces the V2.8 planning value of 0.9 TB/s (section 1: ~89% of the public
1008 GB/s peak). This script does not treat 0.9 TB/s or 1008 GB/s as measured
results. Run only after the official 24GB hardware assert passes.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
from pathlib import Path


def die(msg: str) -> None:
    print(f"[bw][FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def maybe_assert() -> None:
    script = Path(__file__).resolve().parent / "assert_official_4090_24g.sh"
    if script.is_file():
        rc = subprocess.call(["bash", str(script)])
        if rc != 0:
            die("官方 24GB 硬件断言失败，拒绝测带宽。")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=1 << 30, help="copy size in bytes (default 1 GiB)")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    maybe_assert()

    try:
        import torch
    except ImportError:
        die("需要已激活的 torch2.8_cuda12.8 环境。")
    if not torch.cuda.is_available():
        die("CUDA 不可用。")

    nbytes = args.bytes
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    src.fill_(1)
    torch.cuda.synchronize()

    def one_copy() -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dst.copy_(src)
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end))  # milliseconds

    for _ in range(args.warmup):
        one_copy()
    times_ms = [one_copy() for _ in range(args.runs)]
    times_ms.sort()
    median_ms = statistics.median(times_ms)
    # Report both GiB/s (1024-based) and GB/s (decimal) plus TB/s decimal.
    gib = nbytes / (1024 ** 3)
    gb = nbytes / 1e9
    median_s = median_ms / 1e3
    gib_s = gib / median_s
    gb_s = gb / median_s
    tb_s = gb_s / 1e3
    print("device_to_device_copy")
    print(f"bytes={nbytes}")
    print(f"warmup={args.warmup} runs={args.runs}")
    print(f"times_ms={times_ms}")
    print(f"median_ms={median_ms:.4f}")
    print(f"range_ms={times_ms[0]:.4f}..{times_ms[-1]:.4f}")
    print(f"median_GiB_per_s={gib_s:.3f}")
    print(f"median_GB_per_s={gb_s:.3f}")
    print(f"median_TB_per_s={tb_s:.4f}")
    print("note: this measured median replaces the V2.8 planning 0.9 TB/s; do not quote 0.9 or the public 1008 GB/s peak as a host measurement.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
