#!/usr/bin/env python3
"""Gate 0 编排：按配置依次拉起性能与质量子进程，失败即保留日志并返回非 0。"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from gate0_common import (
    SCRIPT_DIR,
    assert_config_allowed,
    die,
    load_protocol,
    load_run_configs,
    log,
    resolve_configs,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 0 编排器")
    p.add_argument("--configs", default="", help="逗号分隔，默认 bf16,te_fp8")
    p.add_argument("--run-dir", required=True, help="本轮日志与结果目录")
    p.add_argument("--allow-optional", action="store_true")
    p.add_argument("--skip-perf", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--skip-supplemental", action="store_true")
    p.add_argument("--quality-runs", type=int, default=None)
    p.add_argument("--decode-new-tokens", type=int, default=None)
    p.add_argument("--limit", type=float, default=None, help="仅调试质量任务；正式比分不要带")
    return p.parse_args()


def run_step(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"执行：{' '.join(cmd)}")
    log(f"日志：{log_path}")
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# cmd: {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(SCRIPT_DIR))
        fh.write(f"\n# exit_code: {proc.returncode}\n")
    return proc.returncode


def main() -> int:
    args = parse_args()
    protocol = load_protocol()
    run_cfgs = load_run_configs()
    names = resolve_configs(
        [x.strip() for x in args.configs.split(",") if x.strip()] or None,
        run_cfgs,
    )
    for n in names:
        assert_config_allowed(n, run_cfgs, args.allow_optional)

    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    failures: list[str] = []
    planned = []

    if args.skip_perf and args.skip_quality:
        die("不能同时 --skip-perf 与 --skip-quality。")

    manifest = {
        "run_dir": str(run_dir),
        "configs": names,
        "started_unix": time.time(),
        "timezone": "Asia/Shanghai",
        "skip_perf": args.skip_perf,
        "skip_quality": args.skip_quality,
        "allow_optional": args.allow_optional,
        "limit": args.limit,
        "steps": [],
    }

    for name in names:
        if not args.skip_perf:
            out = run_dir / f"perf_{name}.json"
            logp = run_dir / f"perf_{name}.log"
            cmd = [py, str(SCRIPT_DIR / "measure_perf.py"), "--config", name, "--out", str(out)]
            if args.allow_optional:
                cmd.append("--allow-optional")
            if args.skip_supplemental:
                cmd.append("--skip-supplemental")
            if args.decode_new_tokens is not None:
                cmd.extend(["--decode-new-tokens", str(args.decode_new_tokens)])
            rc = run_step(cmd, logp)
            planned.append({"config": name, "kind": "perf", "exit": rc, "out": str(out), "log": str(logp)})
            if rc != 0:
                failures.append(f"perf:{name}:exit={rc}")
                # 性能失败仍尝试质量？为了可复现诊断，同一配置的质量仍跑，但总退出码非 0。
        if not args.skip_quality:
            out = run_dir / f"quality_{name}.json"
            logp = run_dir / f"quality_{name}.log"
            cmd = [py, str(SCRIPT_DIR / "measure_quality.py"), "--config", name, "--out", str(out)]
            if args.allow_optional:
                cmd.append("--allow-optional")
            if args.quality_runs is not None:
                cmd.extend(["--quality-runs", str(args.quality_runs)])
            if args.limit is not None:
                cmd.extend(["--limit", str(args.limit)])
            rc = run_step(cmd, logp)
            planned.append({"config": name, "kind": "quality", "exit": rc, "out": str(out), "log": str(logp)})
            if rc != 0:
                failures.append(f"quality:{name}:exit={rc}")

    # 汇总草稿（只抄 JSON 里已有的数）
    summary_out = run_dir / "table_draft.md"
    rc = run_step(
        [py, str(SCRIPT_DIR / "summarize_results.py"), "--run-dir", str(run_dir), "--out", str(summary_out)],
        run_dir / "summarize.log",
    )
    planned.append({"kind": "summarize", "exit": rc, "out": str(summary_out)})
    if rc != 0:
        failures.append(f"summarize:exit={rc}")

    manifest["steps"] = planned
    manifest["finished_unix"] = time.time()
    manifest["failures"] = failures
    write_json(run_dir / "manifest.json", manifest)
    log(f"清单：{run_dir / 'manifest.json'}")
    if failures:
        log("本轮存在失败步骤：" + "; ".join(failures))
        return 1
    log("本轮子进程全部退出 0。请用实测 JSON 填写报告模板，阈值位仍留空。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        die("用户中断")
