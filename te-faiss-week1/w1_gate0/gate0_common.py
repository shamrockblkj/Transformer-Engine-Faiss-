#!/usr/bin/env python3
"""Gate 0 共享逻辑：环境硬检查、检查点校验、TE FP8 转换与 kernel 检测。

只在目标主机上由 run_gate0.sh 调用。本模块不安装任何包，不下载模型权重，
也不登录任何主机。口径见同目录 configs/protocol.yaml 与
/workspace/te-faiss-week1/eval_protocol.md、VERSION_LOCK.md。

撰写日期：2026-11-26（Asia/Shanghai）。
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import socket
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = SCRIPT_DIR / "configs" / "protocol.yaml"
DEFAULT_RUN_CONFIGS = SCRIPT_DIR / "configs" / "run_configs.yaml"
WEEK1_DIR = SCRIPT_DIR.parent
VERSION_LOCK = WEEK1_DIR / "VERSION_LOCK.md"

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------


class Gate0Error(SystemExit):
    """可复现失败，退出码非 0。"""


def die(msg: str, code: int = 1) -> None:
    print(f"[gate0][FATAL] {msg}", file=sys.stderr, flush=True)
    raise Gate0Error(code)


def log(msg: str) -> None:
    print(f"[gate0] {msg}", flush=True)


def load_yaml(path: Path) -> dict[str, Any]:
    """读取协议 / 运行配置。优先 PyYAML；否则读同名 .json 旁路文件（VERSION_LOCK 未锁定 PyYAML）。"""
    path = Path(path)
    data = None
    try:
        import yaml  # type: ignore
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except ImportError:
        json_path = path.with_suffix(".json") if path.suffix in {".yaml", ".yml"} else path
        if path.suffix in {".yaml", ".yml"} and json_path.is_file():
            log(f"未安装 PyYAML，改读旁路 {json_path}（与 yaml 同步生成，不必额外安装）。")
            data = json.loads(json_path.read_text(encoding="utf-8"))
        else:
            die(
                f"当前环境没有 PyYAML，且找不到旁路 JSON：{path.with_suffix('.json')}。"
                "本脚本不会 pip install。"
            )
    if not isinstance(data, dict):
        die(f"配置根节点不是映射：{path}")
    return data


def sha256_file(path: Path, limit_bytes: Optional[int] = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        if limit_bytes is None:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        else:
            h.update(f.read(limit_bytes))
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def pkg_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def median(xs: list[float]) -> float:
    if not xs:
        raise ValueError("median() 收到空列表（不应在未测时调用）")
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return 0.5 * (s[mid - 1] + s[mid])


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        raise ValueError("percentile() 收到空列表（不应在未测时调用）")
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"百分位越界：{p}")
    s = sorted(xs)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    w = k - lo
    return float(s[lo] * (1.0 - w) + s[hi] * w)


def summarize_series(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {
            "n": 0,
            "median": None,
            "min": None,
            "max": None,
            "p50": None,
            "p95": None,
            "values": [],
            "status": "待实测",
        }
    return {
        "n": len(xs),
        "median": median(xs),
        "min": min(xs),
        "max": max(xs),
        "p50": percentile(xs, 50),
        "p95": percentile(xs, 95),
        "values": list(xs),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# 协议与配置
# ---------------------------------------------------------------------------


def load_protocol(path: Optional[Path] = None) -> dict[str, Any]:
    p = path or Path(os.environ.get("GATE0_PROTOCOL", DEFAULT_PROTOCOL))
    if not p.is_file():
        die(f"找不到协议文件：{p}")
    return load_yaml(p)


def load_run_configs(path: Optional[Path] = None) -> dict[str, Any]:
    p = path or Path(os.environ.get("GATE0_RUN_CONFIGS", DEFAULT_RUN_CONFIGS))
    if not p.is_file():
        die(f"找不到运行配置：{p}")
    return load_yaml(p)


def resolve_configs(names: Optional[list[str]], run_cfgs: dict[str, Any]) -> list[str]:
    known = run_cfgs["configs"]
    if not names:
        names = list(run_cfgs.get("default_configs") or ["bf16", "te_fp8"])
    out: list[str] = []
    for n in names:
        n = n.strip()
        if not n:
            continue
        if n not in known:
            die(f"未知配置「{n}」。已知：{', '.join(known)}")
        out.append(n)
    if not out:
        die("没有要跑的配置。")
    return out


def assert_config_allowed(name: str, run_cfgs: dict[str, Any], force_optional: bool) -> dict[str, Any]:
    meta = run_cfgs["configs"][name]
    default_on = bool(meta.get("enabled_by_default"))
    if default_on:
        return meta
    status = meta.get("status", "本轮环境默认未装，待另批")
    if not force_optional:
        die(
            f"配置「{name}」状态为「{status}」。"
            "本轮默认不执行该分支，也不会安装其依赖。"
            "若已另批且包已在环境中装好，须显式传入 --allow-optional。"
        )
    missing = []
    for pkg in meta.get("requires_packages") or []:
        if pkg_version(pkg) is None and _module_missing(pkg):
            missing.append(pkg)
    if missing:
        die(
            f"配置「{name}」已用 --allow-optional 打开，但本环境没有 {missing}。"
            "脚本不会 pip install，也不会安装 gptqmodel 7.12.2 或要求 torch>=2.14 的包。"
            "请走另批的单独脚本。"
        )
    # 再挡一层：即使包在，也拒绝已知会破坏冻结栈的 gptqmodel 版本。
    if name.startswith("gptq"):
        ver = pkg_version("gptqmodel")
        if ver:
            _reject_bad_gptqmodel(ver)
    log(f"警告：将运行待另批配置「{name}」（{status}）。请确认已获批准。")
    return meta


def _module_missing(pkg: str) -> bool:
    aliases = {
        "autoawq": ["awq", "autoawq"],
        "gptqmodel": ["gptqmodel", "auto_gptq"],
        "aqlm": ["aqlm"],
        "llama_cpp": ["llama_cpp", "llama_cpp_python"],
        "transformer_engine": ["transformer_engine"],
    }
    for mod in aliases.get(pkg, [pkg]):
        try:
            importlib.import_module(mod)
            return False
        except Exception:
            continue
    return True


def _reject_bad_gptqmodel(ver: str) -> None:
    # VERSION_LOCK.md 第 3.1 节：禁止 7.12.2 及声明 torch>=2.14 的版本（7.7.0–7.10.0 等）。
    def _tuple(v: str) -> tuple[int, ...]:
        parts = []
        for x in re.split(r"[^\d]+", v):
            if x:
                parts.append(int(x))
        return tuple(parts) or (0,)

    t = _tuple(ver)
    if t >= (7, 7, 0):
        die(
            f"检测到 gptqmodel {ver}。VERSION_LOCK 禁止 7.12.2 以及任何要求 "
            "torch>=2.14 的版本（7.7.0 起多属此类）。拒绝继续，不会卸载或改装。"
        )


# ---------------------------------------------------------------------------
# 环境硬检查（写死目标卡与冻结版本）
# ---------------------------------------------------------------------------


def apply_runtime_env(protocol: dict[str, Any]) -> None:
    hw = protocol["hardware"]
    os.environ.setdefault("HF_ENDPOINT", hw["hf_endpoint"])
    os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", hw["hf_endpoint"])
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    cuda_home = hw["cuda_home"]
    if Path(cuda_home).is_dir():
        os.environ.setdefault("CUDA_HOME", cuda_home)
        os.environ.setdefault("CUDA_PATH", cuda_home)
        bin_dir = str(Path(cuda_home) / "bin")
        if bin_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        lib64 = Path(cuda_home) / "lib64"
        if lib64.is_dir():
            old = os.environ.get("LD_LIBRARY_PATH", "")
            prefix = str(lib64)
            if prefix not in old.split(os.pathsep):
                os.environ["LD_LIBRARY_PATH"] = prefix + (os.pathsep + old if old else "")
    _refresh_cudnn_paths()


def _refresh_cudnn_paths() -> None:
    try:
        import nvidia.cudnn as c  # type: ignore

        p = Path(c.__file__).resolve().parent
        if p.is_dir():
            os.environ.setdefault("CUDNN_PATH", str(p))
            os.environ.setdefault("CUDNN_HOME", str(p))
            lib = p / "lib"
            if lib.is_dir():
                old = os.environ.get("LD_LIBRARY_PATH", "")
                prefix = str(lib)
                if prefix not in old.split(os.pathsep):
                    os.environ["LD_LIBRARY_PATH"] = prefix + (os.pathsep + old if old else "")
    except Exception:
        pass


def check_conda_env(protocol: dict[str, Any]) -> None:
    hw = protocol["hardware"]
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if env == hw["forbidden_conda_env"]:
        die(f"当前环境是被禁用的 {hw['forbidden_conda_env']}，拒绝继续。")
    if env != hw["conda_env"]:
        die(
            f"当前 conda 环境为「{env or '（未激活）'}」，锁定值是 {hw['conda_env']}。"
            "请先 source ~/anaconda3/etc/profile.d/conda.sh && conda activate "
            f"{hw['conda_env']}（推荐用 run_gate0.sh）。"
        )


def check_python_version(protocol: dict[str, Any]) -> None:
    exp = protocol["versions"]["python"]
    got = "%d.%d.%d" % sys.version_info[:3]
    if got != exp:
        die(f"Python 为 {got}，锁定值是 {exp}。")


def _nvidia_smi_query(field: str) -> list[str]:
    import subprocess

    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        die("找不到 nvidia-smi，拒绝在无 NVIDIA 驱动的机器上评测。")
    except Exception as exc:
        die(f"nvidia-smi 查询 {field} 失败：{exc}")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        die(f"nvidia-smi 查询 {field} 返回空。")
    return lines


def check_gpu(protocol: dict[str, Any]) -> dict[str, Any]:
    """只允许单张官方档 GeForce RTX 4090 24GB（sm_89）。改装 48G 或非 24GB 立即失败。"""
    import torch

    hw = protocol["hardware"]
    if not torch.cuda.is_available():
        die("torch.cuda.is_available() 为 False，拒绝继续。")
    n = torch.cuda.device_count()
    if n < 1:
        die("可见 GPU 数量为 0。")

    names = _nvidia_smi_query("name")
    mems = _nvidia_smi_query("memory.total")
    drivers = _nvidia_smi_query("driver_version")
    name0 = names[0]
    mem0_s = mems[0].replace(" MiB", "").replace("MiB", "").strip()
    try:
        mem0 = int(float(mem0_s))
    except ValueError:
        die(f"无法解析 cuda:0 显存：{mems[0]!r}")
    cap = torch.cuda.get_device_capability(0)
    cap_t = (int(cap[0]), int(cap[1]))
    exp_cap = tuple(hw["required_compute_capability"])
    reject_re = re.compile(hw["reject_name_regex"])
    min_mib = int(hw.get("min_memory_mib", 24000))
    max_mib = int(hw.get("max_memory_mib", 25000))
    expected_mib = int(hw.get("expected_memory_mib", 24564))
    reject_at = int(hw.get("reject_memory_mib_at_least", 40000))
    require_single = bool(hw.get("require_single_gpu", True))
    require_rtx = bool(hw.get("required_name_must_contain_rtx", True))

    problems: list[str] = []
    if require_single and (len(names) != 1 or n != 1):
        problems.append(
            f"官方验收要求整机只有一张 GPU，当前 nvidia-smi 报告 {len(names)} 张、torch 可见 {n} 张。"
            "软件显存上限或 CUDA_VISIBLE_DEVICES 不能作为官方 24GB 验收。"
        )
    if reject_re.search(name0):
        problems.append(f"GPU 名称「{name0}」命中禁卡规则（含 Tesla P4 / 4090 D 等），拒绝继续。")
    if hw["required_name_substr"].lower() not in name0.lower():
        problems.append(
            f"GPU 名称「{name0}」不含「{hw['required_name_substr']}」。"
            "禁止在 Tesla P4 或其它非目标卡上评测。"
        )
    if require_rtx and "rtx" not in name0.lower():
        problems.append(f"GPU 名称「{name0}」不含 RTX，不像官方 GeForce RTX 4090。")
    if cap_t != exp_cap:
        problems.append(f"计算能力为 {cap_t[0]}.{cap_t[1]}，锁定主机应为 sm_89（官方 GeForce RTX 4090）。")
    if mem0 >= reject_at:
        problems.append(
            f"显存 {mem0} MiB 属于 ≥{reject_at} MiB 档（改装 4090 48G 历史记录约 49140 MiB）。"
            "不能作为 V2.8 官方 24GB 验收。拒绝继续。"
        )
    elif mem0 < min_mib or mem0 > max_mib:
        problems.append(
            f"显存 {mem0} MiB 不在官方 GeForce RTX 4090 24GB 接受范围 {min_mib}–{max_mib} MiB"
            f"（典型约 {expected_mib} MiB，以新主机 nvidia-smi 为准）。拒绝继续。"
            "软件显存限制不能替代官方 24GB 验收。"
        )
    if problems:
        die("；".join(problems))

    driver = drivers[0].replace(" ", "")
    exp_driver = str(protocol.get("versions", {}).get("driver") or "")
    require_driver = bool(hw.get("require_driver_exact", False))
    if require_driver and exp_driver and exp_driver not in ("待实测", "None"):
        if driver != exp_driver:
            die(f"驱动为 {driver}，锁定值是 {exp_driver}。拒绝继续（不会升级驱动）。")
    else:
        log(f"驱动版本（记录，不把旧 48G 主机的 580.159.03 作为新机硬门槛）：{driver}")

    info = {
        "hostname": socket.gethostname(),
        "gpu_index": 0,
        "gpu_name": name0,
        "gpu_memory_mib": mem0,
        "compute_capability": f"{cap_t[0]}.{cap_t[1]}",
        "driver_version": driver,
        "gpu_count_visible": n,
        "all_visible_names": names,
        "memory_in_official_24g_range": min_mib <= mem0 <= max_mib,
        "expected_memory_mib_typical": expected_mib,
    }
    log(
        f"GPU 检查通过：{name0}，{mem0} MiB（官方 24GB 档 {min_mib}–{max_mib}，典型约 {expected_mib}），"
        f"sm_{cap_t[0]}{cap_t[1]}，驱动 {driver}"
    )
    return info


def check_frozen_python_packages(protocol: dict[str, Any], *, need_te: bool) -> dict[str, Any]:
    import torch

    vers = protocol["versions"]
    errors: list[str] = []
    got: dict[str, Any] = {}

    tv = torch.__version__
    got["torch"] = tv
    if tv != vers["torch"]:
        errors.append(f"PyTorch 为 {tv}，锁定值是 {vers['torch']}")
    tcuda = torch.version.cuda or ""
    got["torch.version.cuda"] = tcuda
    if tcuda != vers["torch_cuda"]:
        errors.append(f"torch.version.cuda 为 {tcuda}，锁定值是 {vers['torch_cuda']}")

    mapping = {
        "huggingface_hub": "huggingface_hub",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "datasets": "datasets",
        "lm_eval": "lm-eval",
        "tokenizers": "tokenizers",
        "safetensors": "safetensors",
        "nvidia_cudnn_cu12": "nvidia-cudnn-cu12",
    }
    for key, dist in mapping.items():
        v = pkg_version(dist)
        got[dist] = v
        if v != vers[key]:
            errors.append(f"{dist} 为 {v}，锁定值是 {vers[key]}")

    te_ver = pkg_version("transformer-engine") or pkg_version("transformer_engine")
    got["transformer-engine"] = te_ver
    if need_te:
        if te_ver != vers["transformer_engine"]:
            errors.append(
                f"transformer-engine 为 {te_ver}，锁定值是 {vers['transformer_engine']}。"
                "VERSION_LOCK 记载第一次安装时 TE 可能尚未装完；未装成就绪前不得把 TE 路径记成已测。"
            )
        try:
            import transformer_engine.pytorch as te  # noqa: F401
        except Exception as exc:
            errors.append(f"import transformer_engine.pytorch 失败：{exc}")

    cudnn_api = None
    try:
        cudnn_api = int(torch.backends.cudnn.version() or 0)
    except Exception:
        cudnn_api = None
    got["torch.backends.cudnn.version"] = cudnn_api
    if cudnn_api is not None and cudnn_api < 91200 and need_te:
        errors.append(
            f"torch.backends.cudnn.version()={cudnn_api}，低于 TE 2.21.0 要求的 9.12（91200）。"
        )

    if errors:
        die("冻结栈偏离锁定值，已停止（不会改 VERSION_LOCK，也不会自行升级）：\n  - " + "\n  - ".join(errors))
    return got


def check_cuda_toolkit(protocol: dict[str, Any]) -> dict[str, Any]:
    import subprocess

    cuda_home = protocol["hardware"]["cuda_home"]
    exp = protocol["versions"]["torch_cuda"]
    if not Path(cuda_home).is_dir():
        die(f"找不到 CUDA 工具包目录 {cuda_home}")
    nvcc = Path(cuda_home) / "bin" / "nvcc"
    if not nvcc.is_file():
        die(f"在 {cuda_home} 中找不到 nvcc，拒绝继续")
    proc = subprocess.run([str(nvcc), "--version"], check=True, capture_output=True, text=True)
    m = re.search(r"release\s+([0-9]+\.[0-9]+)", proc.stdout)
    ver = m.group(1) if m else ""
    if ver != exp:
        die(f"nvcc 报告 CUDA {ver}，锁定值是 {exp}，拒绝继续（不会升级 CUDA）。")
    return {"cuda_home": cuda_home, "nvcc_release": ver, "nvcc_output": proc.stdout.strip()}


def inspect_checkpoint(protocol: dict[str, Any]) -> dict[str, Any]:
    model_cfg = protocol["model"]
    path = Path(model_cfg["path"])
    if not path.is_dir():
        die(
            f"检查点目录不存在：{path}。"
            "脚本只读取该路径，不会执行 huggingface-cli download，也不会向你索取 token。"
        )
    config_path = path / "config.json"
    if not config_path.is_file():
        die(f"{path} 缺少 config.json，不像可用的 HF 检查点。")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = model_cfg["expected"]
    mismatches: list[str] = []
    # head_dim 可能写在 config 里，或由 hidden/heads 推出。
    exp_head = int(expected["head_dim"])
    got_head = config.get("head_dim")
    if got_head is None:
        hs, nh = config.get("hidden_size"), config.get("num_attention_heads")
        if hs and nh and int(hs) % int(nh) == 0:
            got_head = int(hs) // int(nh)
    for key, exp in expected.items():
        if key == "head_dim":
            if int(got_head or 0) != exp_head:
                mismatches.append(f"head_dim 为 {got_head}，期望 {exp_head}")
            continue
        got = config.get(key)
        if got != exp:
            mismatches.append(f"{key} 为 {got}，期望 {exp}")
    if mismatches:
        die(
            f"{path} 的结构参数与 Llama 3 8B 官方形状不符，拒绝加载以免测错检查点：\n  - "
            + "\n  - ".join(mismatches)
        )

    safetensors = sorted(
        p for p in path.glob("*.safetensors") if p.is_file() and p.parent == path
    )
    if not safetensors:
        die(
            f"{path} 根目录没有 *.safetensors。"
            f"只使用 HF safetensors，排除 {model_cfg['exclude_subdir']}/*。"
        )
    original = path / model_cfg["exclude_subdir"]
    file_list = []
    for p in safetensors:
        file_list.append(
            {
                "path": str(p),
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            }
        )
    index = path / "model.safetensors.index.json"
    info = {
        "path": str(path),
        "config_sha256": sha256_file(config_path),
        "config_excerpt": {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "head_dim": got_head,
            "vocab_size": config.get("vocab_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "torch_dtype": config.get("torch_dtype"),
            "transformers_version": config.get("transformers_version"),
        },
        "safetensors": file_list,
        "index_sha256": sha256_file(index) if index.is_file() else None,
        "original_dir_exists": original.is_dir(),
        "original_dir_excluded": True,
        "tokenizer_files": [
            n
            for n in (
                "tokenizer.json",
                "tokenizer.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
            )
            if (path / n).is_file()
        ],
    }
    if original.is_dir():
        log(f"发现 {original}，已按协议排除，不从其加载。")
    log(f"检查点通过结构校验：{path}，{len(safetensors)} 个根目录 safetensors。")
    return info


def collect_env_snapshot(protocol: dict[str, Any], gpu: dict[str, Any], pkgs: dict[str, Any], cuda: dict[str, Any], ckpt: dict[str, Any]) -> dict[str, Any]:
    return {
        "captured_at_timezone": "Asia/Shanghai",
        "captured_at_unix": time.time(),
        "hostname": gpu.get("hostname"),
        "gpu": gpu,
        "packages": pkgs,
        "cuda_toolkit": cuda,
        "checkpoint": ckpt,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": "%d.%d.%d" % sys.version_info[:3],
        "version_lock_path": str(VERSION_LOCK),
        "protocol_seed": protocol["measurement"]["seed"],
        "protocol_seed_frozen": protocol["measurement"].get("seed_frozen", False),
        "notes": [
            "本快照由脚本在测量前写入。字段值来自当前进程与 nvidia-smi，不是事先编造的评测分数。",
            "S2 阈值位不得在本快照或报告中填入 H100 的 1.2 ms。",
        ],
    }


def run_preflight(protocol: dict[str, Any], *, need_te: bool) -> dict[str, Any]:
    apply_runtime_env(protocol)
    check_conda_env(protocol)
    check_python_version(protocol)
    gpu = check_gpu(protocol)
    cuda = check_cuda_toolkit(protocol)
    pkgs = check_frozen_python_packages(protocol, need_te=need_te)
    ckpt = inspect_checkpoint(protocol)
    return collect_env_snapshot(protocol, gpu, pkgs, cuda, ckpt)


# ---------------------------------------------------------------------------
# 模型加载与 TE 转换
# ---------------------------------------------------------------------------


def set_determinism(seed: int) -> None:
    import random

    import numpy as np
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_tokenizer(protocol: dict[str, Any]):
    from transformers import AutoTokenizer

    path = protocol["model"]["path"]
    tok = AutoTokenizer.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_hf_model(protocol: dict[str, Any], precision: str):
    import torch
    from transformers import AutoModelForCausalLM

    path = protocol["model"]["path"]
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    elif precision == "te_fp8_delayed_scaling":
        dtype = torch.bfloat16
    else:
        die(f"未知 precision：{precision}")
    t0 = time.perf_counter()
    common = dict(
        low_cpu_mem_usage=True,
        local_files_only=True,
        trust_remote_code=False,
        device_map=None,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, **common)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **common)
    model = model.to(device="cuda:0", dtype=dtype)
    model.eval()
    load_s = time.perf_counter() - t0
    return model, load_s


def build_delayed_scaling_recipe(protocol: dict[str, Any]):
    from transformer_engine.common.recipe import DelayedScaling, Format

    params = protocol["te_fp8"]["recipe"]
    fmt_name = params["fp8_format"]
    if not hasattr(Format, fmt_name):
        die(f"TE Format 没有 {fmt_name}")
    recipe = DelayedScaling(
        margin=int(params["margin"]),
        amax_history_len=int(params["amax_history_len"]),
        amax_compute_algo=params["amax_compute_algo"],
        fp8_format=getattr(Format, fmt_name),
    )
    return recipe, {
        "class": type(recipe).__name__,
        "margin": params["margin"],
        "amax_history_len": params["amax_history_len"],
        "amax_compute_algo": params["amax_compute_algo"],
        "fp8_format": fmt_name,
        "scaling": protocol["te_fp8"]["scaling"],
        "use_block_scaling": protocol["te_fp8"]["use_block_scaling"],
        "source": "TE 2.21.0 DelayedScaling 官方默认值（Common API / Delayed Scaling 文档，2026-11-26 查阅）",
    }


def te_autocast(recipe):
    import transformer_engine.pytorch as te

    if hasattr(te, "autocast"):
        return te.autocast(enabled=True, recipe=recipe)
    # 2.21 文档将 fp8_autocast 列为弃用别名；仅在无 autocast 时回退。
    if hasattr(te, "fp8_autocast"):
        return te.fp8_autocast(enabled=True, fp8_recipe=recipe)
    die("transformer_engine.pytorch 既无 autocast 也无 fp8_autocast。")
    raise AssertionError


def install_te_autocast_on_forward(model, recipe):
    orig = model.forward

    def wrapped(*args, **kwargs):
        with te_autocast(recipe):
            return orig(*args, **kwargs)

    model.forward = wrapped
    return orig


def replace_linear_with_te(model) -> dict[str, Any]:
    """把 nn.Linear 换成 te.Linear，并复制权重。不替换已是 te.Linear 的模块。"""
    import torch
    import torch.nn as nn
    import transformer_engine.pytorch as te

    te_linear_cls = te.Linear
    replaced: list[str] = []
    skipped: list[dict[str, str]] = []

    def _convert(parent: nn.Module, name: str, module: nn.Linear):
        kwargs = dict(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            params_dtype=module.weight.dtype,
        )
        # device 参数在部分版本可用；失败则先 CPU 再建再搬。
        te_lin = None
        last_err = None
        for extra in ({"device": module.weight.device}, {}):
            try:
                te_lin = te_linear_cls(**kwargs, **extra)
                break
            except TypeError as exc:
                last_err = exc
                te_lin = None
        if te_lin is None:
            raise RuntimeError(f"构造 te.Linear 失败：{last_err}")
        te_lin = te_lin.to(device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            if not hasattr(te_lin, "weight"):
                raise RuntimeError("te.Linear 没有 .weight，无法从 HF 线性层复制参数")
            te_lin.weight.copy_(module.weight.detach())
            if module.bias is not None:
                if not hasattr(te_lin, "bias") or te_lin.bias is None:
                    raise RuntimeError("HF 层有 bias，但 te.Linear 没有可复制的 bias")
                te_lin.bias.copy_(module.bias.detach())
        setattr(parent, name, te_lin)
        replaced.append(name if parent is model else "child")

    # 按路径替换，记录完整名字。
    to_swap: list[tuple[nn.Module, str, str, nn.Linear]] = []
    for qual, module in model.named_modules():
        if isinstance(module, te_linear_cls):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not qual:
            continue
        if "." in qual:
            parent_name, child = qual.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            child = qual
            parent = model
        to_swap.append((parent, child, qual, module))

    for parent, child, qual, module in to_swap:
        try:
            _convert(parent, child, module)
            replaced[-1] = qual
        except Exception as exc:
            skipped.append({"name": qual, "error": str(exc)})

    leftover_linear = [
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and not isinstance(m, te_linear_cls)
    ]
    te_modules = [n for n, m in model.named_modules() if isinstance(m, te_linear_cls)]
    return {
        "replaced": replaced,
        "replaced_count": len(replaced),
        "skipped": skipped,
        "leftover_nn_linear": leftover_linear,
        "te_linear_modules": te_modules,
        "method": "replace nn.Linear with te.Linear; wrap forward in te.autocast(DelayedScaling)",
        "not_done": [
            "未把 LlamaDecoderLayer 整层替换为 TE TransformerLayer（官方 tutorial 路径）。",
            "注意力分数 / softmax 仍走 Hugging Face 实现。",
            "RMSNorm 未替换为 te.RMSNorm。",
        ],
    }


# ---------------------------------------------------------------------------
# FP8 kernel 检测（只记录证据，不编造结论）
# ---------------------------------------------------------------------------

_FP8_NAME_RE = re.compile(r"(fp8|e4m3|e5m2|float8)", re.IGNORECASE)


def te_support_probe() -> dict[str, Any]:
    out: dict[str, Any] = {"status": "待实测"}
    try:
        import transformer_engine.pytorch as te
    except Exception as exc:
        return {"status": "import_failed", "error": str(exc), "fp8_available": False}

    out["te_module"] = getattr(te, "__file__", None)
    for fn_name in (
        "is_fp8_available",
        "is_fp8_block_scaling_available",
        "is_mxfp8_available",
        "get_cudnn_version",
        "get_device_compute_capability",
        "get_default_recipe",
    ):
        fn = getattr(te, fn_name, None)
        if fn is None:
            out[fn_name] = None
            continue
        try:
            if fn_name == "is_fp8_available":
                try:
                    out[fn_name] = fn(return_reason=True)
                except TypeError:
                    out[fn_name] = fn()
            else:
                val = fn()
                out[fn_name] = val if not hasattr(val, "__class__") or val is None or isinstance(val, (bool, int, float, str, tuple, list)) else type(val).__name__
                if fn_name == "get_default_recipe":
                    out["default_recipe_type"] = type(val).__name__
                    out["default_recipe_repr"] = repr(val)
        except Exception as exc:
            out[fn_name] = f"error: {exc}"

    # 可选：FP8GlobalStateManager / check_fp8_support
    for mod_name in (
        "transformer_engine.pytorch.quantization",
        "transformer_engine.pytorch.fp8",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(mod, "check_fp8_support"):
            try:
                out["check_fp8_support"] = list(mod.check_fp8_support())
            except Exception as exc:
                out["check_fp8_support"] = f"error: {exc}"
        mgr = getattr(mod, "FP8GlobalStateManager", None)
        if mgr is not None:
            out["FP8GlobalStateManager"] = True
            for attr in ("is_fp8_available", "is_fp8_enabled"):
                if hasattr(mgr, attr):
                    try:
                        out[f"FP8GlobalStateManager.{attr}"] = getattr(mgr, attr)()
                    except Exception as exc:
                        out[f"FP8GlobalStateManager.{attr}"] = f"error: {exc}"
        break
    out["status"] = "probed"
    return out


def _amax_history_nonzero(model) -> dict[str, Any]:
    found = 0
    nonzero = 0
    examples: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        for attr in ("fp8_meta", "amax_history", "_amax_history"):
            if not hasattr(module, attr):
                continue
            obj = getattr(module, attr)
            tensors: list[Any] = []
            if obj is None:
                continue
            try:
                import torch

                if isinstance(obj, torch.Tensor):
                    tensors = [obj]
                elif isinstance(obj, dict):
                    tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
                else:
                    for maybe in ("amax_history", "scale"):
                        v = getattr(obj, maybe, None)
                        if isinstance(v, torch.Tensor):
                            tensors.append(v)
            except Exception:
                continue
            for t in tensors:
                found += 1
                try:
                    nz = bool(int((t.detach().float().abs().sum() > 0).item()))
                except Exception:
                    nz = False
                if nz:
                    nonzero += 1
                    if len(examples) < 8:
                        examples.append({"module": name, "attr": attr, "shape": list(t.shape)})
    return {"tensors_seen": found, "tensors_nonzero": nonzero, "examples": examples}


def profile_cuda_kernel_names(fn: Callable[[], None], wait_s: float = 0.2) -> dict[str, Any]:
    """对一次 CUDA 调用做短 profile，返回内核名。找不到证据时结论为「未证实」，不编造成功。"""
    import torch
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    kernel_names: list[str] = []
    try:
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            fn()
            torch.cuda.synchronize()
        time.sleep(wait_s)
        for ev in prof.key_averages():
            key = ev.key
            if key and key not in kernel_names:
                kernel_names.append(key)
    except Exception as exc:
        return {
            "status": "profiler_failed",
            "error": str(exc),
            "kernel_names": [],
            "fp8_name_matches": [],
            "verdict": "未证实",
            "verdict_reason": "torch.profiler 失败，不能据此宣称走到或未走到 FP8 kernel。",
        }

    matches = [n for n in kernel_names if _FP8_NAME_RE.search(n)]
    if matches:
        verdict = "见到名称含 fp8/e4m3/e5m2/float8 的 CUDA kernel"
        reason = "这是「真正走到 FP8 kernel」的支持性证据，仍须结合 te.Linear 替换与 autocast 上下文解读。"
    else:
        verdict = "未证实"
        reason = (
            "本次 profile 未出现名称匹配 fp8/e4m3/e5m2/float8 的 kernel。"
            "这不等于证明没有走 FP8（kernel 名可能不含这些字样），只能记为未证实，禁止改写成已走 FP8。"
        )
    return {
        "status": "ok",
        "kernel_names": kernel_names,
        "fp8_name_matches": matches,
        "verdict": verdict,
        "verdict_reason": reason,
    }


def isolated_te_linear_probe(recipe, dtype=None) -> dict[str, Any]:
    import torch
    import transformer_engine.pytorch as te

    if dtype is None:
        dtype = torch.bfloat16
    layer = te.Linear(1024, 1024, bias=False, params_dtype=dtype).to(device="cuda:0", dtype=dtype)
    x = torch.randn(8, 128, 1024, dtype=dtype, device="cuda:0")

    def _run():
        with torch.inference_mode():
            with te_autocast(recipe):
                y = layer(x)
            return y

    # warmup once so DelayedScaling 有历史后再 profile
    with torch.inference_mode():
        for _ in range(3):
            with te_autocast(recipe):
                _ = layer(x)
        torch.cuda.synchronize()
    prof = profile_cuda_kernel_names(lambda: _run())
    amax = _amax_history_nonzero(layer)
    del layer, x
    torch.cuda.empty_cache()
    return {
        "probe": "isolated te.Linear 1024x1024, DelayedScaling autocast, 8x128 tokens",
        "profiler": prof,
        "amax_history": amax,
    }


def detect_fp8_after_forward(model, recipe, run_probe: bool = True) -> dict[str, Any]:
    support = te_support_probe()
    fp8_ok = False
    reason = ""
    avail = support.get("is_fp8_available")
    if isinstance(avail, (list, tuple)) and avail:
        fp8_ok = bool(avail[0])
        reason = str(avail[1]) if len(avail) > 1 else ""
    elif isinstance(avail, bool):
        fp8_ok = avail
    isolated = isolated_te_linear_probe(recipe) if run_probe else {"status": "skipped"}
    amax = _amax_history_nonzero(model)
    te_count = 0
    try:
        import transformer_engine.pytorch as te

        te_count = sum(1 for _, m in model.named_modules() if isinstance(m, te.Linear))
    except Exception:
        te_count = -1

    # 综合判定：只升到「有证据」或保持「未证实」/「不支持」。永不在证据不足时写「已走 FP8」。
    if support.get("status") == "import_failed":
        overall = "无法检测（TE 导入失败）"
    elif avail is not None and fp8_ok is False:
        overall = "当前环境报告 FP8 不可用"
    elif te_count == 0:
        overall = "未替换为 te.Linear，fp8 autocast 不会让原生 nn.Linear 走 FP8 kernel"
    elif isinstance(isolated, dict) and isolated.get("profiler", {}).get("fp8_name_matches"):
        overall = "隔离 te.Linear 探针见到 FP8 风格 kernel 名；全模型路径仍须结合本文件的全模型 profile"
    else:
        overall = "待实测 / 未证实"

    return {
        "support": support,
        "fp8_available": fp8_ok if avail is not None else None,
        "fp8_available_reason": reason,
        "te_linear_count": te_count,
        "isolated_linear_probe": isolated,
        "model_amax_history": amax,
        "overall_verdict": overall,
        "policy": "没有足够证据时一律记「未证实」或「待实测」，禁止编造「已走 FP8 kernel」。",
    }


def fp8_forward_context(precision: str, recipe):
    if precision == "te_fp8_delayed_scaling":
        return te_autocast(recipe)
    return nullcontext()


@contextmanager
def cuda_sync_timer():
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    ev_s = torch.cuda.Event(enable_timing=True)
    ev_e = torch.cuda.Event(enable_timing=True)
    ev_s.record()
    yield_box: dict[str, float] = {}
    try:
        yield yield_box
    finally:
        ev_e.record()
        ev_e.synchronize()
        wall = time.perf_counter() - start
        yield_box["wall_s"] = wall
        yield_box["cuda_ms"] = float(ev_s.elapsed_time(ev_e))


def peak_vram_mib() -> dict[str, Any]:
    import torch

    alloc = torch.cuda.max_memory_allocated(0) / (1024 * 1024)
    reserved = torch.cuda.max_memory_reserved(0) / (1024 * 1024)
    smi = None
    try:
        used = _nvidia_smi_query("memory.used")[0]
        smi = used
    except Exception as exc:
        smi = f"error: {exc}"
    return {
        "torch_max_allocated_mib": alloc,
        "torch_max_reserved_mib": reserved,
        "nvidia_smi_memory_used": smi,
    }


def reset_peak_vram() -> None:
    import torch

    torch.cuda.reset_peak_memory_stats(0)
    torch.cuda.synchronize()
