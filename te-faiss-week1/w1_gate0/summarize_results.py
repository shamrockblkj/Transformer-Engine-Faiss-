#!/usr/bin/env python3
"""把已写出的 JSON 抄成 Markdown 草稿。缺项写「待实测」，绝不编造，绝不填写 H100 的 1.2 ms。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


def _load(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def _fmt_series(s: Optional[dict[str, Any]], unit: str = "") -> str:
    if not s or s.get("status") in (None, "待实测", "skipped") or s.get("n") in (0, None):
        return "待实测"
    med = s.get("median")
    lo, hi = s.get("min"), s.get("max")
    p50, p95 = s.get("p50"), s.get("p95")
    n = s.get("n")
    if med is None:
        return "待实测"
    u = f" {unit}" if unit else ""
    extra = ""
    if p50 is not None and p95 is not None:
        extra = f"；P50={p50:.4g}{u}，P95={p95:.4g}{u}"
    return f"中位数 {med:.4g}{u}（min {lo:.4g}–max {hi:.4g}，n={n}{extra}）"


def _fmt_num(v: Any) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.6g}"
    return "待实测"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.out)

    # 不扫描可选量化结果来「填」已测；没有 JSON 就待实测 / 待另批。
    lines: list[str] = []
    lines.append("# Gate 0 结果草稿（由 JSON 抄录，不是最终报告）")
    lines.append("")
    lines.append(f"- 来源目录：`{run_dir}`")
    lines.append("- 没有出现在 JSON 里的格子一律写「待实测」，禁止补编数字。")
    lines.append("- **S2 停线与 4090 TE FP8 基线阈值位留空。禁止把 H100 的 1.2 ms 写进本草稿。**")
    lines.append("- 目标任务：待补。")
    lines.append("")

    # 环境取第一份能读到的 env
    env = None
    for pat in ("perf_*.json", "quality_*.json"):
        for f in sorted(run_dir.glob(pat)):
            data = _load(f)
            if data and "env" in data:
                env = data["env"]
                break
        if env:
            break
    lines.append("## 环境抄录")
    lines.append("")
    if not env:
        lines.append("待实测（本目录还没有含 env 的 JSON）。")
    else:
        gpu = env.get("gpu") or {}
        pkgs = env.get("packages") or {}
        ckpt = env.get("checkpoint") or {}
        lines.append("| 项目 | 值 |")
        lines.append("| --- | --- |")
        lines.append(f"| 主机名 | {env.get('hostname', '待实测')} |")
        lines.append(
            f"| GPU | {gpu.get('gpu_name', '待实测')} / {gpu.get('gpu_memory_mib', '待实测')} MiB / sm {gpu.get('compute_capability', '待实测')} |"
        )
        lines.append(f"| 驱动 | {gpu.get('driver_version', '待实测')} |")
        cuda = (env.get("cuda_toolkit") or {}).get("nvcc_release", "待实测")
        lines.append(f"| CUDA 工具包 (nvcc) | {cuda} |")
        lines.append(f"| Conda | {env.get('conda_default_env', '待实测')} |")
        lines.append(f"| Python | {env.get('python', '待实测')} |")
        lines.append(f"| PyTorch | {pkgs.get('torch', '待实测')} / torch.cuda={pkgs.get('torch.version.cuda', '待实测')} |")
        lines.append(f"| Transformer Engine | {pkgs.get('transformer-engine', '待实测')} |")
        lines.append(
            f"| transformers / tokenizers / safetensors | {pkgs.get('transformers')} / {pkgs.get('tokenizers')} / {pkgs.get('safetensors')} |"
        )
        lines.append(f"| lm-eval / datasets | {pkgs.get('lm-eval')} / {pkgs.get('datasets')} |")
        lines.append(f"| huggingface_hub | {pkgs.get('huggingface_hub')} |")
        lines.append(f"| nvidia-cudnn-cu12 / cudnn.version() | {pkgs.get('nvidia-cudnn-cu12')} / {pkgs.get('torch.backends.cudnn.version')} |")
        lines.append(f"| 检查点 | {(ckpt.get('path') or '待实测')} |")
        lines.append(f"| config.json sha256 | {ckpt.get('config_sha256', '待实测')} |")
    lines.append("")

    lines.append("## 质量（对照同源 BF16；主指标 word_perplexity / MMLU acc）")
    lines.append("")
    lines.append("| 配置 | WikiText-2 word PPL | C4 word PPL | MMLU 5-shot acc | 目标任务 | 正式资格 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for cfg in ("bf16", "fp16", "te_fp8"):
        q = _load(run_dir / f"quality_{cfg}.json")
        if not q:
            # 默认矩阵里的空位
            if cfg == "fp16":
                continue
            lines.append(f"| {cfg} | 待实测 | 待实测 | 待实测 | 待补 | 待实测 |")
            continue
        s = q.get("summary") or {}
        lines.append(
            "| {cfg} | {w} | {c} | {m} | 待补 | {ok} |".format(
                cfg=q.get("config", cfg),
                w=_fmt_series(s.get("wikitext_word_perplexity")),
                c=_fmt_series(s.get("c4_word_perplexity")),
                m=_fmt_series(s.get("mmlu_acc")),
                ok="是" if q.get("formal_eligible") else "否（见 failures / limit / 次数）",
            )
        )
    lines.append("")
    lines.append("本轮环境默认未装、待另批的配置（AWQ / GPTQ / GGUF / AQLM / 注意力 W4+FFN）不在上表填数。")
    lines.append("")

    lines.append("## 性能主形状（B=1, L=2048）")
    lines.append("")
    lines.append("| 配置 | Prefill 吞吐 | Prefill 延迟 | Decode 吞吐 | Decode 逐 token 延迟 | FFN ms/token | 峰值显存 | 启动时间 | FP8 kernel |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cfg in ("bf16", "fp16", "te_fp8"):
        d = _load(run_dir / f"perf_{cfg}.json")
        if not d:
            if cfg == "fp16":
                continue
            lines.append(f"| {cfg} | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | 待实测 | {'不适用' if cfg=='bf16' else '待实测'} |")
            continue
        prim = d.get("primary") or {}
        fp8 = (d.get("fp8_detection") or {}).get("overall_verdict", "待实测")
        vram = (prim.get("peak_vram") or {}).get("torch_max_allocated_mib")
        start = d.get("startup_seconds_load_plus_first_forward")
        lines.append(
            "| {cfg} | {pt} | {pl} | {dt} | {dl} | {ffn} | {vram} | {st} | {fp8} |".format(
                cfg=d.get("config", cfg),
                pt=_fmt_series(prim.get("prefill_tokens_per_s"), "tok/s"),
                pl=_fmt_series(prim.get("prefill_latency_ms"), "ms"),
                dt=_fmt_series(prim.get("decode_tokens_per_s"), "tok/s"),
                dl=_fmt_series(prim.get("decode_per_token_latency_ms"), "ms"),
                ffn=_fmt_series(prim.get("ffn_per_token_ms"), "ms"),
                vram=_fmt_num(vram) + " MiB" if isinstance(vram, (int, float)) else "待实测",
                st=_fmt_num(start) + " s" if isinstance(start, (int, float)) else "待实测",
                fp8=fp8 if cfg == "te_fp8" or d.get("precision") == "te_fp8_delayed_scaling" else "不适用",
            )
        )
    lines.append("")
    lines.append("### S2 阈值位（留空）")
    lines.append("")
    lines.append("| 校准项 | 状态 | 数值 |")
    lines.append("| --- | --- | --- |")
    lines.append("| 4090 上 TE 逐张量 DelayedScaling 的 decode / FFN 基线延迟（中位数；prefill 与 decode 分列） | 待实测 | （留空） |")
    lines.append("| 由该同卡基线校准后的 S2 停线（FFN gather ms/token） | 待实测 | （留空，禁止填 1.2 或任何虚构值） |")
    lines.append("| 是否真正走到 TE FP8 kernel（而非静默回退） | 待实测 | （留空；上表仅抄录检测结论，不代替本行回填） |")
    lines.append("| 峰值显存、prefill / decode 吞吐 | 待实测 | （留空；上表抄录不等于已写入 VERSION_LOCK 第 7 节） |")
    lines.append("")
    lines.append("请把确认后的实测抄进 `gate0_baseline_report.md`。不要修改 `VERSION_LOCK.md` 的阈值位，除非开发总管明确要求回填。")
    lines.append("")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[gate0] 已写入 {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
