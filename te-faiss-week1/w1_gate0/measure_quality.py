#!/usr/bin/env python3
"""Gate 0 质量测量：WikiText-2 PPL、C4 PPL、MMLU 5-shot。目标任务待补。

口径：eval_protocol.md 第 4 节。使用 lm-eval 0.4.13 官方任务配置，不另切未记录子集。
主指标 word_perplexity 与 MMLU 加权 acc。质量种子已冻结为 0。不编造分数。

撰写日期：2026-11-26（Asia/Shanghai）。
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Optional

from gate0_common import (
    assert_config_allowed,
    build_delayed_scaling_recipe,
    die,
    install_te_autocast_on_forward,
    load_hf_model,
    load_protocol,
    load_run_configs,
    load_tokenizer,
    log,
    replace_linear_with_te,
    resolve_configs,
    run_preflight,
    set_determinism,
    summarize_series,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 0 质量测量（只跑不编造）")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-optional", action="store_true")
    p.add_argument("--tasks", default="wikitext,c4,mmlu", help="逗号分隔，默认官方三项")
    p.add_argument("--quality-runs", type=int, default=None, help="默认 3，与协议一致")
    p.add_argument("--limit", type=float, default=None, help="仅供调试；正式比分不得带 limit")
    return p.parse_args()


def _filter_kwargs(fn, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    allowed = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in allowed}


def _extract_metrics(results: dict[str, Any], task: str, keys: list[str]) -> dict[str, Any]:
    block = (results.get("results") or {}).get(task) or {}
    found: dict[str, Any] = {"raw_keys": list(block.keys()), "values": {}}
    for k, v in block.items():
        found["values"][k] = v
    # 常见键名：word_perplexity,none / acc,none
    mapped: dict[str, Any] = {}
    for want in keys:
        mapped[want] = None
        for cand in (want, f"{want},none", f"{want},none,0"):
            if cand in block:
                mapped[want] = block[cand]
                break
        else:
            # 后缀模糊匹配
            for rk, rv in block.items():
                if rk.split(",")[0] == want and "stderr" not in rk and "alias" not in rk:
                    mapped[want] = rv
                    break
    found["mapped"] = mapped
    return found


def build_lm(model, tokenizer, protocol: dict[str, Any]):
    try:
        from lm_eval.models.huggingface import HFLM
    except Exception as exc:
        die(f"无法导入 lm_eval.models.huggingface.HFLM：{exc}。lm-eval 与 transformers 相容性见 VERSION_LOCK 待实测项。")

    kwargs = dict(
        pretrained=model,
        tokenizer=tokenizer,
        device="cuda:0",
        batch_size=int(protocol["quality"]["batch_size"]),
        trust_remote_code=False,
        dtype="bfloat16",
    )
    # 不另切窗口：不传死的 max_length，让 HFLM 用模型默认最大上下文，并在结果里记下来。
    kwargs = _filter_kwargs(HFLM.__init__, kwargs)
    try:
        lm = HFLM(**kwargs)
    except Exception as exc:
        die(f"构造 lm-eval HFLM 失败：{exc}")
    info = {
        "class": type(lm).__name__,
        "batch_size": getattr(lm, "batch_size", None),
        "max_length": getattr(lm, "max_length", None),
        "model_max_length_attr": None,
    }
    for attr in ("max_length", "_max_length", "max_gen_toks"):
        if hasattr(lm, attr):
            info[attr] = getattr(lm, attr)
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            info["model_max_position_embeddings"] = getattr(cfg, "max_position_embeddings", None)
    except Exception:
        pass
    return lm, info


def run_lm_eval(
    lm,
    tasks: list[str],
    *,
    num_fewshot: Optional[int],
    seed: int,
    limit: Optional[float],
) -> dict[str, Any]:
    try:
        from lm_eval import simple_evaluate
    except Exception as exc:
        die(f"无法导入 lm_eval.simple_evaluate：{exc}")

    kwargs: dict[str, Any] = dict(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        confirm_run_unsafe_code=False,
    )
    kwargs = _filter_kwargs(simple_evaluate, kwargs)
    # 部分版本用 --seed
    if "seed" in inspect.signature(simple_evaluate).parameters:
        kwargs.setdefault("seed", seed)
    log(f"lm-eval tasks={tasks} num_fewshot={num_fewshot} seed={seed} kwargs={list(kwargs)}")
    t0 = time.perf_counter()
    try:
        results = simple_evaluate(**kwargs)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "tasks": tasks,
            "elapsed_s": time.perf_counter() - t0,
            "kwargs_used": {k: (v if k != "model" else type(v).__name__) for k, v in kwargs.items()},
        }
    # simple_evaluate 可能返回很大的 samples；保留 metrics，samples 另存开关默认关掉以控制体积。
    slim = {
        "ok": True,
        "tasks": tasks,
        "elapsed_s": time.perf_counter() - t0,
        "results": results.get("results") if isinstance(results, dict) else None,
        "n-shot": results.get("n-shot") if isinstance(results, dict) else None,
        "config": results.get("config") if isinstance(results, dict) else None,
        "versions": results.get("versions") if isinstance(results, dict) else None,
        "kwargs_used": {k: (v if k != "model" else type(v).__name__) for k, v in kwargs.items()},
    }
    return slim


def main() -> int:
    args = parse_args()
    protocol = load_protocol()
    run_cfgs = load_run_configs()
    name = resolve_configs([args.config], run_cfgs)[0]
    meta = assert_config_allowed(name, run_cfgs, args.allow_optional)
    precision = meta["precision"]
    need_te = precision == "te_fp8_delayed_scaling"
    qcfg = protocol["quality"]
    seed = int(protocol["measurement"]["seed"])
    runs = int(args.quality_runs if args.quality_runs is not None else qcfg["quality_runs"])
    if runs < 1:
        die("--quality-runs 必须 >= 1")
    if args.limit is not None:
        log("警告：使用了 --limit，结果不得进入门禁（eval_protocol 要求正式比分用完整官方配置）。")

    requested = [t.strip() for t in args.tasks.split(",") if t.strip()]
    forbidden = set(qcfg["mmlu"]["forbidden_substitutes"])
    bad = [t for t in requested if t in forbidden]
    if bad:
        die(f"禁止用 {bad} 代替主报告 MMLU（eval_protocol 4.3）。")

    set_determinism(seed)
    snapshot = run_preflight(protocol, need_te=need_te)
    tokenizer = load_tokenizer(protocol)
    model, load_s = load_hf_model(protocol, precision)
    convert_info = None
    recipe_meta = None
    if need_te:
        recipe, recipe_meta = build_delayed_scaling_recipe(protocol)
        convert_info = replace_linear_with_te(model)
        if convert_info["replaced_count"] == 0:
            die("TE 路径没有替换任何 nn.Linear。拒绝把未转换模型的质量结果写成 TE FP8。")
        install_te_autocast_on_forward(model, recipe)

    lm, lm_info = build_lm(model, tokenizer, protocol)

    groups: list[tuple[str, list[str], Optional[int]]] = []
    if "wikitext" in requested:
        groups.append(("wikitext", [qcfg["wikitext"]["lm_eval_task"]], int(qcfg["wikitext"]["num_fewshot"])))
    if "c4" in requested:
        groups.append(("c4", [qcfg["c4"]["lm_eval_task"]], int(qcfg["c4"]["num_fewshot"])))
    if "mmlu" in requested:
        groups.append(("mmlu", [qcfg["mmlu"]["lm_eval_task"]], int(qcfg["mmlu"]["num_fewshot"])))
    extras = [t for t in requested if t not in {"wikitext", "c4", "mmlu"}]
    if extras:
        die(f"未识别的任务 {extras}。本脚本默认可跑 wikitext / c4 / mmlu；目标任务待补。")

    all_runs: dict[str, list[dict[str, Any]]] = {g[0]: [] for g in groups}
    failures: list[str] = []

    for r in range(runs):
        log(f"质量第 {r + 1}/{runs} 轮（正式种子冻结为 {seed}，每轮相同）")
        set_determinism(seed)  # eval_protocol 4.5：正式比分必须 --seed 0，不得改种子
        for label, tasks, nshot in groups:
            raw = run_lm_eval(lm, tasks, num_fewshot=nshot, seed=seed, limit=args.limit)
            rec: dict[str, Any] = {"run_index": r, "seed_used": seed, "raw_ok": raw.get("ok"), "elapsed_s": raw.get("elapsed_s")}
            if not raw.get("ok"):
                rec["error"] = raw.get("error")
                failures.append(f"{label} run {r}: {raw.get('error')}")
                all_runs[label].append(rec)
                continue
            rec["lm_eval"] = {
                "results": raw.get("results"),
                "n-shot": raw.get("n-shot"),
                "versions": raw.get("versions"),
                "config_excerpt": {
                    "keys": list((raw.get("config") or {}).keys()) if isinstance(raw.get("config"), dict) else None
                },
                "kwargs_used": raw.get("kwargs_used"),
            }
            if label in ("wikitext", "c4"):
                rec["metrics"] = _extract_metrics(
                    {"results": raw.get("results") or {}},
                    tasks[0],
                    list([qcfg["ppl_primary_metric"], *qcfg["ppl_secondary_metrics"]]),
                )
            elif label == "mmlu":
                rec["metrics"] = _extract_metrics({"results": raw.get("results") or {}}, "mmlu", ["acc"])
                rec["subgroups"] = {}
                for sg in qcfg["mmlu"]["subgroups"]:
                    rec["subgroups"][sg] = _extract_metrics({"results": raw.get("results") or {}}, sg, ["acc"])
            all_runs[label].append(rec)

    def _series_for(label: str, metric: str) -> dict[str, Any]:
        vals: list[float] = []
        for rec in all_runs.get(label) or []:
            mapped = ((rec.get("metrics") or {}).get("mapped") or {})
            v = mapped.get(metric)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return summarize_series(vals)

    summary = {
        "wikitext_word_perplexity": _series_for("wikitext", "word_perplexity") if "wikitext" in all_runs else {"status": "skipped"},
        "c4_word_perplexity": _series_for("c4", "word_perplexity") if "c4" in all_runs else {"status": "skipped"},
        "mmlu_acc": _series_for("mmlu", "acc") if "mmlu" in all_runs else {"status": "skipped"},
        "target_task": {"status": "待补", "value": None},
        "quality_runs_requested": runs,
        "seed_base": seed,
        "seed_scheme": "正式比分每轮均使用冻结种子 0（eval_protocol 4.5，开发总管 2026-11-26）。重复运行只观察实现噪声，不改种子。",
        "limit_used": args.limit,
        "limit_note": "只要带了 limit，结果不得进入门禁",
        "ppl_window_note": (
            "未另切滚动窗口，跟随 lm-eval / 模型默认最大上下文。"
            "实际 max_length 见 lm_info。若要与 QTIP 论文 8192 对照须单独列表，不得与默认窗口混比。"
        ),
        "c4_note": (
            "C4 必须走 lm-eval 0.4.13 官方任务分片；若未完成官方分片，结果不得进入门禁。"
        ),
    }

    out = {
        "kind": "gate0_quality",
        "config": name,
        "status_label": meta.get("status"),
        "precision": precision,
        "env": snapshot,
        "load_seconds": load_s,
        "te_conversion": convert_info,
        "te_recipe": recipe_meta,
        "lm_info": lm_info,
        "runs": all_runs,
        "summary": summary,
        "failures": failures,
        "formal_eligible": (args.limit is None) and (not failures) and (runs >= 3),
    }
    write_json(Path(args.out), out)
    log(f"已写入 {args.out}")
    if failures:
        log("存在失败任务，详见 JSON 的 failures。这些结果不得进入门禁。")
        return 2
    if runs < 3:
        log("重复次数少于 3，按协议不能作为正式中位数/波动。")
        return 2
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        die("用户中断")
