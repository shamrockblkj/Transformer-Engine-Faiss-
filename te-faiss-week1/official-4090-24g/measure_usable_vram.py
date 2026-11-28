#!/usr/bin/env python3
"""Measure usable VRAM on the official 4090 24GB host.

Replaces the V2.8 planning assumption of ~23.5 GB remaining after driver /
display use (section 1, 待实测). Do not quote 23.5 GB as a measured result.
Reports nvidia-smi and torch values in MiB (and decimal GB only as a secondary
label). Run after the official 24GB hardware assert passes.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def die(msg: str) -> None:
    print(f"[vram][FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def smi(field: str) -> str:
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    )
    return proc.stdout.strip().splitlines()[0].strip()


def maybe_assert() -> None:
    script = Path(__file__).resolve().parent / "assert_official_4090_24g.sh"
    if script.is_file():
        rc = subprocess.call(["bash", str(script)])
        if rc != 0:
            die("官方 24GB 硬件断言失败，拒绝测可用显存。")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alloc-probe", action="store_true",
                        help="optional: binary-search max allocatable tensor (uses GPU memory)")
    args = parser.parse_args()
    maybe_assert()

    try:
        import torch
    except ImportError:
        die("需要已激活的 torch2.8_cuda12.8 环境。")
    if not torch.cuda.is_available():
        die("CUDA 不可用。")

    total_mib = int(float(smi("memory.total")))
    used_mib = int(float(smi("memory.used")))
    free_smi_mib = int(float(smi("memory.free")))
    if total_mib < 24000 or total_mib > 25000 or total_mib >= 40000:
        die(f"nvidia-smi memory.total={total_mib} MiB is not official 24GB class.")

    free_b, total_b = torch.cuda.mem_get_info(0)
    torch_free_mib = free_b // (1024 * 1024)
    torch_total_mib = total_b // (1024 * 1024)

    display_hint = "unknown"
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-accounted-apps=pid,used_memory", "--format=csv"],
            capture_output=True, text=True,
        )
        # Prefer a simple process list.
        q2 = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        text = q2.stdout
        if "Display" in text or "Xorg" in text or "gnome" in text.lower():
            display_hint = "possible_display_process_in_nvidia_smi"
        else:
            display_hint = "no_obvious_display_process_in_nvidia_smi"
    except Exception:
        display_hint = "nvidia_smi_process_query_failed"

    print("usable_vram")
    print(f"nvidia_smi_memory_total_MiB={total_mib}")
    print(f"nvidia_smi_memory_used_MiB={used_mib}")
    print(f"nvidia_smi_memory_free_MiB={free_smi_mib}")
    print(f"torch_mem_get_info_total_MiB={torch_total_mib}")
    print(f"torch_mem_get_info_free_MiB={torch_free_mib}")
    print(f"nvidia_smi_memory_total_decimal_GB={total_mib * 1024 * 1024 / 1e9:.3f}")
    print(f"torch_mem_get_info_free_decimal_GB={free_b / 1e9:.3f}")
    print(f"display_hint={display_hint}")
    print("note: ~23.5 GB in V2.8 section 1 is a planning assumption, not a measured host value.")

    if args.alloc_probe:
        lo, hi = 0, torch_free_mib
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                t = torch.empty((mid, 1024 * 1024), dtype=torch.uint8, device="cuda")
                del t
                torch.cuda.empty_cache()
                best = mid
                lo = mid + 1
            except RuntimeError:
                torch.cuda.empty_cache()
                hi = mid - 1
        print(f"max_allocatable_uint8_MiB={best}")
        print("note: alloc probe is a supplementary estimate, not a replacement for nvidia-smi total, and not official 24GB acceptance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
