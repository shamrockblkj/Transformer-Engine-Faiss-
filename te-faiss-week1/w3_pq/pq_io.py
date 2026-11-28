#!/usr/bin/env python3
"""配置、权重（只读本地 HF safetensors）、环境检查。不下载、不索要令牌、不登录远程主机。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

ROLE_TO_PARAM = {
    "gate": "gate_proj",
    "up": "up_proj",
    "down": "down_proj",
}

ROLE_INPUT_WIDTH_KEY = {
    "gate": "hidden_size",
    "up": "hidden_size",
    "down": "intermediate_size",
}

ROLE_OUTPUT_WIDTH_KEY = {
    "gate": "intermediate_size",
    "up": "intermediate_size",
    "down": "hidden_size",
}


def fail(msg: str, code: int = 2) -> None:
    print(f"FATAL: {msg}", file=sys.stderr)
    raise SystemExit(code)


def warn(msg: str) -> None:
    print(f"WARN: {msg}", file=sys.stderr)


def load_json(path: Path) -> dict:
    if not path.is_file():
        fail(f"找不到配置文件：{path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        fail(f"配置文件必须是 JSON 对象：{path}")
    return data


def resolve_model_dir(model_path: str) -> Path:
    path = Path(model_path)
    if not path.is_dir():
        fail(
            f"模型目录不存在：{path}。请确认目标主机上已放置官方 Meta-Llama-3-8B 的 "
            "HF safetensors（路径锁定为 /data/models/Meta-Llama-3-8B）。"
            "本脚本不会下载权重，也不会索要 Hugging Face token。"
        )
    return path


def load_hf_config(model_dir: Path) -> dict:
    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        fail(f"模型目录缺少 config.json：{cfg_path}（拒绝从网络补全）")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def weight_map(model_dir: Path) -> dict[str, Path]:
    """解析 HF 分片索引或单文件，排除 original/*。"""
    original = model_dir / "original"
    index = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"
    mapping: dict[str, Path] = {}
    if index.is_file():
        with index.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        wmap = payload.get("weight_map") or {}
        for key, rel in wmap.items():
            rel_str = str(rel)
            if rel_str.startswith("original/") or rel_str.startswith("original\\"):
                continue
            full = model_dir / rel_str
            if original in full.resolve().parents or full.resolve() == original.resolve():
                continue
            mapping[key] = full
        if not mapping:
            fail(f"索引 {index} 中没有可用的非 original safetensors 权重")
        return mapping
    if single.is_file():
        return {"__single__": single}
    # 允许目录内多个 *.safetensors（仍排除 original/）
    files = sorted(
        p
        for p in model_dir.glob("*.safetensors")
        if "original" not in p.parts
    )
    if not files:
        fail(
            f"{model_dir} 中未找到 HF safetensors（已排除 original/*）。"
            "本脚本不会去网上拉取。"
        )
    return {p.name: p for p in files}


def _open_and_load(tensor_name: str, file_path: Path):
    try:
        from safetensors import safe_open
    except ImportError:
        fail("未安装 safetensors。请使用 VERSION_LOCK 中与 transformers 一起锁定的环境，不要另装量化包。")
    if not file_path.is_file():
        fail(f"权重分片不存在：{file_path}")
    with safe_open(str(file_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if tensor_name not in keys:
            fail(f"{file_path} 中没有张量 {tensor_name}")
        return f.get_tensor(tensor_name)


def linear_weight_name(layer: int, role: str) -> str:
    if role not in ROLE_TO_PARAM:
        fail(f"未知 FFN 角色：{role}（允许 gate / up / down）")
    return f"model.layers.{int(layer)}.mlp.{ROLE_TO_PARAM[role]}.weight"


def load_linear_weight(model_dir: Path, layer: int, role: str, mapping: dict[str, Path]):
    """只读加载一层一个线性层的权重，返回 FP32 CPU numpy [N, D]（out, in）。"""
    import torch

    name = linear_weight_name(layer, role)
    if "__single__" in mapping:
        tensor = _open_and_load(name, mapping["__single__"])
    elif name in mapping:
        tensor = _open_and_load(name, mapping[name])
    else:
        # 无索引时遍历分片
        tensor = None
        for path in sorted(set(mapping.values())):
            try:
                from safetensors import safe_open
            except ImportError:
                fail("未安装 safetensors")
            with safe_open(str(path), framework="pt", device="cpu") as f:
                if name in f.keys():
                    tensor = f.get_tensor(name)
                    break
        if tensor is None:
            fail(f"未找到权重 {name}（已跳过 original/*）")
    if not isinstance(tensor, torch.Tensor):
        fail(f"{name} 不是 torch.Tensor")
    if tensor.ndim != 2:
        fail(f"{name} 维数异常：{tuple(tensor.shape)}")
    return tensor.detach().to(dtype=torch.float32, device="cpu").contiguous().numpy()


def expected_weight_shape(hf_config: dict, role: str) -> tuple[int, int]:
    n_out = int(hf_config[ROLE_OUTPUT_WIDTH_KEY[role]])
    n_in = int(hf_config[ROLE_INPUT_WIDTH_KEY[role]])
    return n_out, n_in


def pq_m_for_dim(dim: int, dsub: int = 4) -> int:
    if dim % dsub != 0:
        fail(f"输入维 D={dim} 不能被 dsub={dsub} 整除，无法做 PQ")
    return dim // dsub


def check_gpu_policy(
    refuse_names: list[str],
    expected_substr: str,
    device: str,
    required_capability: tuple[int, int] = (8, 9),
) -> str:
    """正式路径硬检：单张官方 GeForce RTX 4090 24GB（名称含 RTX 4090，sm_89，显存 24000–25000 MiB），否则直接失败。

    Tesla P4、其它卡、非 CUDA 或 --device cpu 一律拒绝。
    本函数只应在正式测量路径调用；--self-check / --dry-run 不得走这里。
    PQ 训练本身仍用 CPU Faiss，但激活采集与评测必须在该 4090 上进行。
    """
    if not device or not str(device).startswith("cuda"):
        fail(
            f"正式路径拒绝 device={device!r}。除 --self-check / --dry-run 外，"
            "不得用 CPU 做激活采集、评测或正式测量。请使用 cuda:0（官方 GeForce RTX 4090 24GB，sm_89）。"
        )
    import torch

    if not torch.cuda.is_available():
        fail(
            "torch.cuda.is_available() 为 False。正式路径必须在官方 GeForce RTX 4090 24GB（sm_89）上运行；"
            "PQ 训练虽走 CPU Faiss，激活抽取与评测不得改走 CPU。"
        )
    if torch.cuda.device_count() != 1:
        fail(
            f"官方验收要求整机只有一张 GPU，torch 可见 {torch.cuda.device_count()} 张。"
            "软件显存上限不能作为官方 24GB 验收。"
        )
    name = torch.cuda.get_device_name(0)
    problems: list[str] = []
    for needle in refuse_names:
        if needle and needle.lower() in name.lower():
            problems.append(f"GPU 名称「{name}」命中禁卡规则（{needle}，含 Tesla P4）")
            break
    expect = expected_substr or "4090"
    if expect.lower() not in name.lower():
        problems.append(
            f"GPU 名称「{name}」不含「{expect}」。禁止在 Tesla P4 或其它非目标卡上测量。"
        )
    if "rtx" not in name.lower():
        problems.append(f"GPU 名称「{name}」不含 RTX，不像官方 GeForce RTX 4090。")
    if "4090 d" in name.lower() or "4090d" in name.lower().replace(" ", ""):
        problems.append(f"GPU 名称「{name}」像 RTX 4090 D，不是标准 GeForce RTX 4090。")
    major, minor = torch.cuda.get_device_capability(0)
    sm = f"sm_{major}{minor}"
    exp_major, exp_minor = int(required_capability[0]), int(required_capability[1])
    if (major, minor) != (exp_major, exp_minor):
        problems.append(f"计算能力为 {sm}，正式测量必须是 sm_{exp_major}{exp_minor}（官方 GeForce RTX 4090）。")
    # nvidia-smi memory.total is integer MiB; do not convert via decimal GB.
    mem_mib = None
    try:
        import subprocess
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
        mem_mib = int(float(proc.stdout.strip().splitlines()[0].replace(" ", "")))
    except Exception:
        props = torch.cuda.get_device_properties(0)
        mem_mib = int(props.total_memory // (1024 * 1024))
    if mem_mib >= 40000:
        problems.append(
            f"显存 {mem_mib} MiB 属于 ≥40000 MiB 档（改装 48G 历史记录约 49140 MiB）。"
            "不能作为 V2.8 官方 24GB 验收。"
        )
    elif mem_mib < 24000 or mem_mib > 25000:
        problems.append(
            f"显存 {mem_mib} MiB 不在官方 24GB 档 24000–25000 MiB（典型约 24564 MiB）。"
            "软件显存限制不能替代官方 24GB 验收。"
        )
    if problems:
        fail("；".join(problems) + "。拒绝继续。")
    print(f"INFO: GPU 硬检通过：CUDA device 0 = {name}  capability={sm}  memory={mem_mib} MiB（官方 24GB 档）")
    return name


def check_frozen_env(cfg_env: dict) -> dict:
    """核对锁定的 conda 环境名与关键包版本；偶离即退。不安装、不升级任何包。

    硬检项（与 VERSION_LOCK 对齐）：conda 环境名、PyTorch、transformers、faiss-cpu。
    禁用 torch2.14_cuda13.0。本函数只应在正式测量路径调用。
    """
    import importlib
    from importlib.metadata import PackageNotFoundError, version

    report: dict = {}

    def pkg_ver(*names: str) -> Optional[str]:
        for name in names:
            try:
                return version(name)
            except PackageNotFoundError:
                continue
        return None

    expect_env = str(cfg_env.get("conda_env") or "torch2.8_cuda12.8")
    forbidden_env = str(cfg_env.get("forbidden_conda_env") or "torch2.14_cuda13.0")
    conda_env = os.environ.get("CONDA_DEFAULT_ENV") or ""
    report["conda_default_env"] = conda_env or None
    if conda_env == forbidden_env:
        fail(f"当前环境是被禁用的 {forbidden_env}，拒绝继续。请改用 {expect_env}。")
    if conda_env != expect_env:
        fail(
            f"当前 conda 环境为「{conda_env or '（未激活）'}」，锁定值是 {expect_env}。"
            "请先 source ~/anaconda3/etc/profile.d/conda.sh && conda activate "
            f"{expect_env}（推荐用 run_w3_pq.sh）。"
        )

    torch_mod = importlib.import_module("torch")
    torch_ver = getattr(torch_mod, "__version__", "unknown")
    report["torch"] = torch_ver
    expect_torch = cfg_env.get("torch") or "2.8.0+cu128"
    if str(torch_ver) != str(expect_torch):
        fail(
            f"PyTorch 为 {torch_ver}，锁定值是 {expect_torch}。"
            "禁止升级或换用 torch2.14_cuda13.0。"
        )

    expect_tf = str(cfg_env.get("transformers") or "5.24.1")
    got_tf = pkg_ver("transformers")
    report["transformers"] = got_tf
    if got_tf != expect_tf:
        fail(f"transformers 为 {got_tf}，锁定值是 {expect_tf}。拒绝继续。")

    expect_faiss = str(cfg_env.get("faiss_cpu") or "1.16.0")
    got_faiss = pkg_ver("faiss-cpu", "faiss_cpu")
    report["faiss-cpu"] = got_faiss
    if got_faiss != expect_faiss:
        fail(
            f"faiss-cpu 为 {got_faiss}，锁定值是 {expect_faiss}。"
            "本实验只允许 CPU 包，禁止 faiss-gpu。拒绝继续。"
        )

    for forbidden in cfg_env.get("forbid_packages") or []:
        ver = pkg_ver(forbidden)
        if ver is not None:
            fail(
                f"检测到不应在本轮安装的包 {forbidden}=={ver}。"
                "W3 默认不安装 AWQ/GPTQ/AQLM，也不允许要求 torch>=2.14 的量化包。"
            )
        if forbidden in sys.modules:
            fail(f"当前进程已 import 被禁模块 {forbidden}")

    report["huggingface_hub"] = pkg_ver("huggingface-hub", "huggingface_hub")
    report["safetensors"] = pkg_ver("safetensors")
    report["numpy"] = pkg_ver("numpy")
    print(
        f"INFO: 冻结环境硬检通过：conda={conda_env}  torch={torch_ver}  "
        f"transformers={got_tf}  faiss-cpu={got_faiss}"
    )
    return report


def enforce_offline_hub() -> None:
    """强制本地文件，避免 transformers / huggingface_hub 访问网络。不读写任何 token 文件。"""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def list_groups(cfg: dict, selected: Optional[list[str]]) -> list[dict]:
    groups = cfg.get("groups") or []
    if not groups:
        fail("配置缺少 groups")
    by_id = {g["id"]: g for g in groups}
    if not selected:
        return list(groups)
    out = []
    for gid in selected:
        if gid not in by_id:
            fail(f"未知对照分组 {gid}。可选：{', '.join(by_id)}")
        out.append(by_id[gid])
    return out


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
