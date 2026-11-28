#!/usr/bin/env python3
"""Gate 0 性能测量：启动时间、prefill/decode 吞吐与延迟、峰值 VRAM、FFN 耗时。

口径：eval_protocol.md 第 5 节；主表已冻结为 B=1、L=2048、decode 256。warm-up 后再测，至少 3 次，
报告中位数与波动（含 P50/P95）。decode 先 batch=1。不编造数字。

撰写日期：2026-11-26（Asia/Shanghai）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

from gate0_common import (
    SCRIPT_DIR,
    assert_config_allowed,
    detect_fp8_after_forward,
    die,
    fp8_forward_context,
    install_te_autocast_on_forward,
    load_hf_model,
    load_protocol,
    load_run_configs,
    load_tokenizer,
    log,
    peak_vram_mib,
    profile_cuda_kernel_names,
    replace_linear_with_te,
    reset_peak_vram,
    resolve_configs,
    run_preflight,
    set_determinism,
    summarize_series,
    build_delayed_scaling_recipe,
    write_json,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gate 0 性能测量（只跑不编造）")
    p.add_argument("--config", required=True, help="单个配置名，如 bf16 或 te_fp8")
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    p.add_argument("--allow-optional", action="store_true")
    p.add_argument("--skip-supplemental", action="store_true", help="只跑主形状 L=2048")
    p.add_argument("--skip-fp8-probe", action="store_true")
    p.add_argument("--decode-new-tokens", type=int, default=None, help="覆盖脚本默认 decode 步数（仍不是冻结验收线）")
    return p.parse_args()


def _corpus_ids(tokenizer, length: int):
    import torch

    path = SCRIPT_DIR / "prompts" / "perf_corpus.txt"
    text = path.read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    if ids.numel() == 0:
        die("perf_corpus.txt 分词后为空。")
    reps = (length + int(ids.numel()) - 1) // int(ids.numel())
    packed = ids.repeat(reps)[:length]
    return packed.to(device="cuda:0").unsqueeze(0)


def _attach_ffn_timer(model):
    """用 CUDA event 累加名为 mlp 的模块耗时。失败则返回空记录，不编造。"""
    import torch

    state = {"ms": 0.0, "calls": 0, "hooks": [], "error": None}

    def _pre(_mod, _inp):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        state["pre"] = ev

    def _post(_mod, _inp, _out):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        ev.synchronize()
        pre = state.get("pre")
        if pre is not None:
            state["ms"] += float(pre.elapsed_time(ev))
            state["calls"] += 1

    try:
        for name, mod in model.named_modules():
            if name.endswith("mlp") or name.split(".")[-1] == "mlp":
                state["hooks"].append(mod.register_forward_pre_hook(_pre))
                state["hooks"].append(mod.register_forward_hook(_post))
    except Exception as exc:
        state["error"] = str(exc)
    return state


def _clear_hooks(state: dict[str, Any]) -> None:
    for h in state.get("hooks") or []:
        try:
            h.remove()
        except Exception:
            pass
    state["hooks"] = []


def _generate_greedy(model, input_ids, new_tokens: int, recipe, precision: str):
    import torch

    with torch.inference_mode():
        with fp8_forward_context(precision, recipe):
            out = model(input_ids=input_ids, use_cache=True)
            cache = out.past_key_values
            last = input_ids[:, -1:]
            # 用 logits 取 argmax，与 temperature=0 greedy 一致。
            nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            last = nxt
            for _ in range(new_tokens - 1):
                out = model(input_ids=last, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                last = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return cache


def measure_prefill(model, input_ids, recipe, precision: str) -> float:
    import torch

    torch.cuda.synchronize()
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    with torch.inference_mode():
        with fp8_forward_context(precision, recipe):
            _ = model(input_ids=input_ids, use_cache=True)
    ev_e.record()
    ev_e.synchronize()
    return float(ev_s.elapsed_time(ev_e))


def measure_decode_steps(
    model,
    input_ids,
    new_tokens: int,
    recipe,
    precision: str,
    ffn_state: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    import torch

    # 先做一次不计入的 prefill，拿到 cache，再逐 token 计时。
    with torch.inference_mode():
        with fp8_forward_context(precision, recipe):
            out = model(input_ids=input_ids, use_cache=True)
            cache = out.past_key_values
            last = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize()

    step_ms: list[float] = []
    ffn_ms: list[float] = []
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    for _ in range(new_tokens):
        if ffn_state is not None:
            ffn_state["ms"] = 0.0
            ffn_state["calls"] = 0
        ev_s.record()
        with torch.inference_mode():
            with fp8_forward_context(precision, recipe):
                out = model(input_ids=last, past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                last = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ev_e.record()
        ev_e.synchronize()
        step_ms.append(float(ev_s.elapsed_time(ev_e)))
        if ffn_state is not None:
            ffn_ms.append(float(ffn_state["ms"]))
    return {
        "per_token_ms": step_ms,
        "ffn_per_token_ms": ffn_ms,
        "total_decode_ms": sum(step_ms),
        "tokens": new_tokens,
    }


def run_one_shape(
    model,
    tokenizer,
    *,
    prefill_len: int,
    decode_tokens: int,
    warmup: int,
    timed: int,
    recipe,
    precision: str,
    collect_ffn: bool,
) -> dict[str, Any]:
    ids = _corpus_ids(tokenizer, prefill_len)
    if int(ids.shape[-1]) != prefill_len or int(ids.shape[0]) != 1:
        die(f"构造输入失败：得到 shape={tuple(ids.shape)}，期望 (1, {prefill_len})")

    # FFN hook 会在每层 mlp 上 synchronize，若挂在同一轮会污染端到端延迟。
    # 因此先无 hook 测 prefill/decode，再单独加 hook 采 FFN ms/token。
    log(f"warmup prefill L={prefill_len} × {warmup}")
    for _ in range(warmup):
        measure_prefill(model, ids, recipe, precision)
        if decode_tokens > 0:
            measure_decode_steps(model, ids, decode_tokens, recipe, precision, None)

    prefill_ms: list[float] = []
    decode_run_ms: list[float] = []
    decode_tok_s: list[float] = []
    all_step_ms: list[float] = []
    all_ffn_ms: list[float] = []
    reset_peak_vram()
    for i in range(timed):
        log(f"timed run {i + 1}/{timed}  prefill L={prefill_len}, decode={decode_tokens}")
        pm = measure_prefill(model, ids, recipe, precision)
        prefill_ms.append(pm)
        if decode_tokens > 0:
            dec = measure_decode_steps(model, ids, decode_tokens, recipe, precision, None)
            decode_run_ms.append(dec["total_decode_ms"])
            if dec["total_decode_ms"] > 0:
                decode_tok_s.append(dec["tokens"] / (dec["total_decode_ms"] / 1000.0))
            all_step_ms.extend(dec["per_token_ms"])
    vram = peak_vram_mib()

    ffn_hook_error = None
    if collect_ffn and decode_tokens > 0:
        ffn_state = _attach_ffn_timer(model)
        try:
            log(f"FFN-only timing pass × {timed}（不计入端到端延迟）")
            for i in range(timed):
                dec = measure_decode_steps(model, ids, decode_tokens, recipe, precision, ffn_state)
                all_ffn_ms.extend(dec["ffn_per_token_ms"])
            ffn_hook_error = ffn_state.get("error")
        finally:
            _clear_hooks(ffn_state)

    prefill_tok_s = [prefill_len / (ms / 1000.0) for ms in prefill_ms if ms > 0]
    result: dict[str, Any] = {
        "batch": 1,
        "prefill_length": prefill_len,
        "decode_new_tokens": decode_tokens,
        "warmup_runs": warmup,
        "timed_runs": timed,
        "prefill_latency_ms": summarize_series(prefill_ms),
        "prefill_tokens_per_s": summarize_series(prefill_tok_s),
        "decode_e2e_latency_ms": summarize_series(decode_run_ms) if decode_tokens else {"status": "skipped"},
        "decode_tokens_per_s": summarize_series(decode_tok_s) if decode_tokens else {"status": "skipped"},
        "decode_per_token_latency_ms": summarize_series(all_step_ms) if all_step_ms else {"status": "skipped"},
        "ffn_per_token_ms": summarize_series(all_ffn_ms) if all_ffn_ms else {"status": "待实测", "reason": "未采到 FFN hook 样本"},
        "peak_vram": vram,
        "sampling": "greedy / do_sample=False / temperature=0 / top_p 关闭（eval_protocol 5.2，已冻结）",
    }
    if ffn_hook_error:
        result["ffn_hook_error"] = ffn_hook_error
    return result


def main() -> int:
    args = parse_args()
    protocol = load_protocol()
    run_cfgs = load_run_configs()
    names = resolve_configs([args.config], run_cfgs)
    name = names[0]
    meta = assert_config_allowed(name, run_cfgs, args.allow_optional)
    precision = meta["precision"]
    need_te = precision == "te_fp8_delayed_scaling"

    meas = protocol["measurement"]
    seed = int(meas["seed"])
    set_determinism(seed)
    decode_tokens = int(args.decode_new_tokens or meas["decode_new_tokens"])
    frozen_decode = int(meas["decode_new_tokens"])
    if decode_tokens != frozen_decode:
        log(
            f"警告：decode 步数为 {decode_tokens}，已冻结主表是 {frozen_decode}。"
            "该结果不得作为主表正式比分，只能作补充。"
        )
    warmup = int(meas["warmup_runs"])
    timed = int(meas["timed_runs"])
    if timed < 3:
        die("协议要求每配置至少 3 次正式测量。")
    if int(meas["decode_batch_size"]) != 1:
        die("本周 decode 先 batch=1，协议被改成了其它值，拒绝继续。")

    snapshot = run_preflight(protocol, need_te=need_te)
    tokenizer = load_tokenizer(protocol)
    t_load0 = time.perf_counter()
    model, load_s = load_hf_model(protocol, precision)
    convert_info = None
    recipe = None
    recipe_meta = None
    if need_te:
        recipe, recipe_meta = build_delayed_scaling_recipe(protocol)
        convert_info = replace_linear_with_te(model)
        if convert_info["replaced_count"] == 0:
            die("TE 路径没有替换任何 nn.Linear。拒绝把未转换模型的结果写成 TE FP8。")
        install_te_autocast_on_forward(model, recipe)
    load_and_convert_s = time.perf_counter() - t_load0

    # 启动时间：加载（及转换）完成后，第一次 warmup prefill 结束。
    ids_boot = _corpus_ids(tokenizer, int(meas["primary_prefill_length"]))
    t_boot0 = time.perf_counter()
    measure_prefill(model, ids_boot, recipe, precision)
    first_forward_s = time.perf_counter() - t_boot0
    startup_s = load_and_convert_s + first_forward_s

    fp8_detect: dict[str, Any]
    if need_te:
        fp8_detect = detect_fp8_after_forward(model, recipe, run_probe=not args.skip_fp8_probe)

        def _one_decode():
            measure_decode_steps(model, ids_boot, 1, recipe, precision, None)

        fp8_detect["full_model_one_decode_profile"] = profile_cuda_kernel_names(_one_decode)
    else:
        fp8_detect = {"overall_verdict": "不适用（非 TE FP8 配置）"}

    primary = run_one_shape(
        model,
        tokenizer,
        prefill_len=int(meas["primary_prefill_length"]),
        decode_tokens=decode_tokens,
        warmup=warmup,
        timed=timed,
        recipe=recipe,
        precision=precision,
        collect_ffn=True,
    )

    supplemental: dict[str, Any] = {"status": "skipped"}
    if not args.skip_supplemental:
        supplemental = {"prefill_only": [], "short_prompt_decode": []}
        for L in meas.get("supplemental_prefill_lengths") or []:
            L = int(L)
            if L == int(meas["primary_prefill_length"]):
                continue
            log(f"可选补充：prefill L={L}（不替代主表）")
            supplemental["prefill_only"].append(
                run_one_shape(
                    model,
                    tokenizer,
                    prefill_len=L,
                    decode_tokens=0,
                    warmup=warmup,
                    timed=timed,
                    recipe=recipe,
                    precision=precision,
                    collect_ffn=False,
                )
            )
        for L in meas.get("supplemental_decode_prompt_lengths") or []:
            L = int(L)
            if L == int(meas["primary_prefill_length"]):
                continue
            log(f"可选补充：decode prompt L={L}（不替代主表）")
            supplemental["short_prompt_decode"].append(
                run_one_shape(
                    model,
                    tokenizer,
                    prefill_len=L,
                    decode_tokens=decode_tokens,
                    warmup=warmup,
                    timed=timed,
                    recipe=recipe,
                    precision=precision,
                    collect_ffn=True,
                )
            )

    out = {
        "kind": "gate0_perf",
        "config": name,
        "status_label": meta.get("status"),
        "precision": precision,
        "env": snapshot,
        "load_seconds": load_s,
        "load_and_convert_seconds": load_and_convert_s,
        "first_warmup_prefill_seconds": first_forward_s,
        "startup_seconds_load_plus_first_forward": startup_s,
        "te_conversion": convert_info,
        "te_recipe": recipe_meta,
        "fp8_detection": fp8_detect,
        "protocol_notes": {
            "primary_shape": "batch=1, prefill L=2048（eval_protocol 5.1 / V2.6 主形状）",
            "decode_new_tokens": decode_tokens,
            "decode_new_tokens_frozen": bool(meas.get("decode_new_tokens_frozen", True)),
            "decode_new_tokens_note": (
                "性能主表已冻结为 decode 256（eval_protocol 5.1，开发总管 2026-11-26）。"
                "报告须抄录实际步数；主表不得改用其它步数。"
            ),
            "sampling_frozen": bool(meas["sampling"].get("frozen", True)),
            "seed": seed,
            "seed_frozen": bool(meas.get("seed_frozen", True)),
            "s2_threshold": "留空，禁止填入 H100 的 1.2 ms",
        },
        "primary": primary,
        "supplemental_a7": supplemental,
    }
    out_path = Path(args.out)
    write_json(out_path, out)
    log(f"已写入 {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        die("用户中断")
