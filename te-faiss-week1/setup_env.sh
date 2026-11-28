#!/usr/bin/env bash
# 第 1 周 W0 环境安装脚本。
# 只在目标主机 conda 环境 torch2.8_cuda12.8 中，按同目录 VERSION_LOCK.md 做可重复安装。
# 不登录其它机器、不索要凭据、不下载模型权重、不安装 GPU Faiss。
# 已决策：单独钉死 nvidia-cudnn-cu12 9.28.1.4 后再编译 TE 2.21.0，不回退到 2.19.0；
# 不安装 AWQ/GPTQ/AQLM；失败则停止并把日志写入同目录 logs/。
# 主机实测（2026-11-26）：pip 无法在同一求解中同时满足 torch 2.8.0 元数据的
# nvidia-cudnn-cu12==9.10.2.21 与锁定的 9.28.1.4。TE 及硬依赖 torch 的包改为
# 先装 CUDA 12 核心/运行时依赖，再 --no-build-isolation --no-deps 安装；
# 不安装 transformer-engine-cu13。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCK_FILE="${SCRIPT_DIR}/VERSION_LOCK.md"
CONSTRAINTS_FILE="${SCRIPT_DIR}/.pip-constraints.w0.txt"
CONSTRAINTS_SOLVER_FILE="${SCRIPT_DIR}/.pip-constraints.w0.solver.txt"
LOG_DIR="${SCRIPT_DIR}/logs"

EXPECTED_CONDA_ENV="torch2.8_cuda12.8"
FORBIDDEN_CONDA_ENV="torch2.14_cuda13.0"
EXPECTED_PYTHON="3.12.0"
EXPECTED_TORCH_PREFIX="2.8.0"
EXPECTED_TORCH_CUDA_TAG="cu128"
EXPECTED_TORCH="2.8.0+cu128"
EXPECTED_TORCH_CUDA="12.8"
# 旧 48G 主机曾锁定 580.159.03。新官方 24GB 主机的驱动待连接后写入快照，不作为硬门槛。
# 若设置 STRICT_DRIVER=1 且 EXPECTED_DRIVER 为具体版本，才做精确匹配。
EXPECTED_DRIVER="${EXPECTED_DRIVER:-}"
HISTORICAL_DRIVER_48G="580.159.03"
EXPECTED_CUDA_HOME="/usr/local/cuda-12.8"
EXPECTED_HUB="2.4.0"
EXPECTED_CUDNN="9.28.1.4"
TE_MIN_CUDNN_INT=91200
EXPECTED_TE="2.21.0"
EXPECTED_FAISS="1.16.0"
EXPECTED_TRANSFORMERS="5.24.1"
EXPECTED_ACCELERATE="1.16.0"
EXPECTED_DATASETS="5.2.0"
EXPECTED_LM_EVAL="0.4.13"
EXPECTED_TOKENIZERS="0.23.3"
EXPECTED_SAFETENSORS="0.8.0"
EXPECTED_SENTENCEPIECE="0.2.3"
EXPECTED_EINOPS="0.8.2"
EXPECTED_PEFT="0.22.0"
MODEL_PATH="/data/models/Meta-Llama-3-8B"
HF_MIRROR="https://hf-mirror.com"

CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"

mkdir -p "${LOG_DIR}"
RUN_STAMP="$(TZ=Asia/Shanghai date '+%Y%m%d-%H%M%S')"
LOG_FILE="${LOG_DIR}/setup_env-${RUN_STAMP}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() { printf '[setup_env] %s\n' "$*"; }
die() {
  printf '[setup_env] 错误：%s\n' "$*" >&2
  printf '[setup_env] 已停止，不会把 Transformer Engine 从 %s 改成 2.19.0，也不会改其它锁定主版本。\n' "${EXPECTED_TE}" >&2
  printf '[setup_env] 完整日志：%s\n' "${LOG_FILE}" >&2
  exit 1
}

fail_step() {
  local step_log="$1"
  local msg="$2"
  printf '[setup_env] 错误：%s\n' "${msg}" >&2
  printf '[setup_env] 本步日志：%s\n' "${step_log}" >&2
  printf '[setup_env] 完整日志：%s\n' "${LOG_FILE}" >&2
  printf '[setup_env] 已停止，不会把 Transformer Engine 从 %s 改成 2.19.0，也不会改其它锁定主版本。\n' "${EXPECTED_TE}" >&2
  exit 1
}

run_logged() {
  local step_log="$1"
  shift
  log "运行：$*"
  log "本步日志：${step_log}"
  set +e
  "$@" >"${step_log}" 2>&1
  local rc=$?
  set -e
  cat "${step_log}"
  if [[ "${rc}" -ne 0 ]]; then
    fail_step "${step_log}" "命令失败（退出码 ${rc}）：$*"
  fi
}

[[ -f "${LOCK_FILE}" ]] || die "找不到版本锁定文件：${LOCK_FILE}"
log "本次日志：${LOG_FILE}"

# ---------------------------------------------------------------------------
# 激活已锁定的 conda 环境（非交互 shell 必须先 source conda.sh）
# ---------------------------------------------------------------------------
[[ -f "${CONDA_SH}" ]] || die "找不到 ${CONDA_SH}。请在目标主机用该路径的 Anaconda 安装。"
# shellcheck disable=SC1090
source "${CONDA_SH}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${FORBIDDEN_CONDA_ENV}" ]]; then
  die "当前环境是被禁用的 ${FORBIDDEN_CONDA_ENV}，拒绝继续。"
fi

conda activate "${EXPECTED_CONDA_ENV}" || die "无法激活 conda 环境 ${EXPECTED_CONDA_ENV}"

[[ "${CONDA_DEFAULT_ENV:-}" == "${EXPECTED_CONDA_ENV}" ]] \
  || die "激活后环境为「${CONDA_DEFAULT_ENV:-}」，不是锁定的 ${EXPECTED_CONDA_ENV}"
if [[ "${CONDA_DEFAULT_ENV}" == "${FORBIDDEN_CONDA_ENV}" ]]; then
  die "拒绝在 ${FORBIDDEN_CONDA_ENV} 中安装。"
fi

command -v python >/dev/null || die "激活后找不到 python"
command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi，无法校验驱动与 GPU"

# ---------------------------------------------------------------------------
# 校验已冻结版本，任何偏离都退出且不安装
# ---------------------------------------------------------------------------
assert_frozen_stack() {
  local when="$1"
  python - "${when}" "${EXPECTED_PYTHON}" "${EXPECTED_TORCH}" "${EXPECTED_TORCH_CUDA}" "${EXPECTED_HUB}" <<'PY' \
    || die "${when}：已冻结的 PyTorch / Python / huggingface_hub 偏离锁定值，停止（不会继续编译 TE，也不会改锁定主版本）。"
import sys
import importlib.metadata as m
import torch

when, exp_py, exp_torch, exp_cuda, exp_hub = sys.argv[1:]

def ver(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return "MISSING"

errors = []
py = "%d.%d.%d" % sys.version_info[:3]
if py != exp_py:
    errors.append(f"Python 为 {py}，锁定值是 {exp_py}")
if torch.__version__ != exp_torch:
    errors.append(f"PyTorch 为 {torch.__version__}，锁定值是 {exp_torch}")
if (torch.version.cuda or "") != exp_cuda:
    errors.append(f"torch.version.cuda 为 {torch.version.cuda}，锁定值是 {exp_cuda}")
if not torch.cuda.is_available():
    errors.append("torch.cuda.is_available() 为 False")
if ver("huggingface_hub") != exp_hub:
    errors.append(f"huggingface_hub 为 {ver('huggingface_hub')}，锁定值是 {exp_hub}")
if errors:
    print(when)
    for e in errors:
        print(e)
    raise SystemExit(1)
print("OK")
PY
}

assert_frozen_stack "安装前检查"

driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
driver_ver="${driver_ver%%$'\n'*}"
driver_ver="${driver_ver// /}"
log "驱动版本（记录，不把旧 48G 主机的 ${HISTORICAL_DRIVER_48G} 作为新机硬门槛）：${driver_ver}"
if [[ "${STRICT_DRIVER:-0}" == "1" && -n "${EXPECTED_DRIVER}" && "${EXPECTED_DRIVER}" != "待实测" ]]; then
  [[ "${driver_ver}" == "${EXPECTED_DRIVER}" ]] \
    || die "STRICT_DRIVER=1 且驱动为 ${driver_ver}，期望 ${EXPECTED_DRIVER}，拒绝继续（不会升级驱动）"
fi

[[ -d "${EXPECTED_CUDA_HOME}" ]] || die "找不到 CUDA 工具包目录 ${EXPECTED_CUDA_HOME}"
export CUDA_HOME="${EXPECTED_CUDA_HOME}"
export CUDA_PATH="${EXPECTED_CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
if [[ -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if command -v nvcc >/dev/null; then
  nvcc_ver="$(nvcc --version | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | awk 'NR==1{print; exit}')"
  [[ "${nvcc_ver}" == "${EXPECTED_TORCH_CUDA}" ]] \
    || die "nvcc 报告 CUDA ${nvcc_ver}，锁定值是 ${EXPECTED_TORCH_CUDA}，拒绝继续（不会升级 CUDA）"
else
  die "在 ${CUDA_HOME} 中找不到 nvcc，拒绝继续"
fi

# 官方 GeForce RTX 4090 24GB 硬断言：非 24GB 或改装 48G 立即失败，不再警告后继续。
ASSERT_SH="${SCRIPT_DIR}/official-4090-24g/assert_official_4090_24g.sh"
if [[ -x "${ASSERT_SH}" ]]; then
  log "运行官方 24GB 硬件断言：${ASSERT_SH}"
  "${ASSERT_SH}" || die "官方 24GB 硬件断言失败，拒绝安装。改装 48G 或非 24GB 卡不能作为 V2.8 官方验收。"
else
  log "未找到 ${ASSERT_SH}，改用内联官方 24GB 硬检。"
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader)"
  gpu_name="${gpu_name%%$'\n'*}"
  gpu_mem="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)"
  gpu_mem="${gpu_mem%%$'\n'*}"
  gpu_mem="${gpu_mem// /}"
  gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
  [[ "${gpu_count}" == "1" ]] || die "官方验收要求整机只有一张 GPU，当前 ${gpu_count} 张。软件显存上限不能作为官方 24GB 验收。"
  if echo "${gpu_name}" | grep -Eiq '4090[[:space:]]*D'; then
    die "GPU 名称「${gpu_name}」像 RTX 4090 D，不是标准 GeForce RTX 4090。"
  fi
  echo "${gpu_name}" | grep -Eiq 'RTX' || die "GPU 名称「${gpu_name}」不含 RTX。"
  echo "${gpu_name}" | grep -Eiq '4090' || die "GPU 名称「${gpu_name}」不含 4090。"
  mem_int="${gpu_mem%%.*}"
  if (( mem_int >= 40000 )); then
    die "显存 ${mem_int} MiB 属于 ≥40000 MiB 档（改装 48G 历史记录约 49140 MiB）。不能作为 V2.8 官方 24GB 验收。"
  fi
  if (( mem_int < 24000 || mem_int > 25000 )); then
    die "显存 ${mem_int} MiB 不在官方 24GB 档 24000–25000 MiB（典型约 24564 MiB）。拒绝继续。"
  fi
  log "内联硬检通过：${gpu_name}，${mem_int} MiB（官方 24GB 档）。"
fi

gpu_cc="$(python -c 'import torch; m=torch.cuda.get_device_capability(0); print("%d.%d"%m)')"
log "torch 计算能力：${gpu_cc}"
[[ "${gpu_cc}" == "8.9" ]] || die "计算能力为 ${gpu_cc}，官方 GeForce RTX 4090 应为 sm_89。拒绝在其它 GPU 上安装"

# Optional local wheelhouse to speed up / allow more offline installs.
if [[ -z "${WHEELHOUSE:-}" && -d "${SCRIPT_DIR}/official-4090-24g/wheelhouse" ]]; then
  WHEELHOUSE="${SCRIPT_DIR}/official-4090-24g/wheelhouse"
fi
if [[ -n "${WHEELHOUSE:-}" && -d "${WHEELHOUSE}" ]]; then
  export PIP_FIND_LINKS="${WHEELHOUSE}${PIP_FIND_LINKS:+:${PIP_FIND_LINKS}}"
  log "使用本地 wheelhouse（PIP_FIND_LINKS）：${WHEELHOUSE}"
fi

# ---------------------------------------------------------------------------
# 镜像、约束文件、cuDNN 路径
# ---------------------------------------------------------------------------
export HF_ENDPOINT="${HF_MIRROR}"
export HUGGINGFACE_HUB_ENDPOINT="${HF_MIRROR}"
export NVTE_FRAMEWORK="pytorch"
export NVTE_CUDA_ARCHS="${NVTE_CUDA_ARCHS:-89}"
export MAX_JOBS="${MAX_JOBS:-8}"

refresh_cudnn_paths() {
  cudnn_pkg_path="$(python - <<'PY'
import pathlib
try:
    import nvidia.cudnn as c
    p = pathlib.Path(c.__file__).resolve().parent
    print(p if p.is_dir() else "")
except Exception:
    print("")
PY
)"
  if [[ -n "${cudnn_pkg_path}" ]]; then
    export CUDNN_PATH="${cudnn_pkg_path}"
    export CUDNN_HOME="${cudnn_pkg_path}"
    if [[ -d "${cudnn_pkg_path}/lib" ]]; then
      export LD_LIBRARY_PATH="${cudnn_pkg_path}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
    log "使用包内 cuDNN 路径：${cudnn_pkg_path}"
  fi
}

write_constraints() {
  {
    cat <<EOF
torch==${EXPECTED_TORCH_PREFIX}
huggingface_hub==${EXPECTED_HUB}
huggingface-hub==${EXPECTED_HUB}
nvidia-cudnn-cu12==${EXPECTED_CUDNN}
transformers==${EXPECTED_TRANSFORMERS}
accelerate==${EXPECTED_ACCELERATE}
datasets==${EXPECTED_DATASETS}
lm-eval==${EXPECTED_LM_EVAL}
tokenizers==${EXPECTED_TOKENIZERS}
safetensors==${EXPECTED_SAFETENSORS}
sentencepiece==${EXPECTED_SENTENCEPIECE}
einops==${EXPECTED_EINOPS}
faiss-cpu==${EXPECTED_FAISS}
transformer-engine==${EXPECTED_TE}
transformer-engine-cu12==${EXPECTED_TE}
transformer-engine-torch==${EXPECTED_TE}
EOF
    python - <<'PY'
import importlib.metadata as m
skip = {"nvidia-cudnn-cu12"}
seen = set()
for dist in m.distributions():
    name = dist.metadata.get("Name") or getattr(dist, "name", "")
    if not name:
        continue
    key = name.lower().replace("_", "-")
    if key.startswith("nvidia-") and key not in skip and key not in seen:
        seen.add(key)
        print(f"{name}=={dist.version}")
PY
  } > "${CONSTRAINTS_FILE}"
  grep -v -E '^nvidia-cudnn-cu12==' "${CONSTRAINTS_FILE}" > "${CONSTRAINTS_SOLVER_FILE}"
  log "已写入 pip 约束：${CONSTRAINTS_FILE}（钉死 torch / huggingface_hub / nvidia-cudnn-cu12==${EXPECTED_CUDNN}，并冻结其它已装 nvidia-* 包）"
  log "已写入求解用约束：${CONSTRAINTS_SOLVER_FILE}（不含 nvidia-cudnn-cu12，避免与 torch 元数据的 9.10.2.21 引脚冲突；装完后必须复检 cuDNN 未被降级）"
}

write_constraints

# ---------------------------------------------------------------------------
# 方案 (a)：按约束单独安装并钉死 nvidia-cudnn-cu12 9.28.1.4
# ---------------------------------------------------------------------------
read_cudnn() {
  python - <<'PY'
import importlib.metadata as m
import torch

def ver(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return "MISSING"

print(f"pkg={ver('nvidia-cudnn-cu12')}")
v = torch.backends.cudnn.version()
print(f"api={v if v is not None else 0}")
PY
}

assert_cudnn_pin() {
  local when="$1"
  eval "$(read_cudnn)"
  if [[ "${pkg}" != "${EXPECTED_CUDNN}" ]]; then
    die "${when}：nvidia-cudnn-cu12 为 ${pkg}，锁定值是 ${EXPECTED_CUDNN}。已停止，不改锁定主版本。"
  fi
  if (( api < TE_MIN_CUDNN_INT )); then
    die "${when}：torch.backends.cudnn.version()=${api}，低于 ${TE_MIN_CUDNN_INT}。已停止，不改锁定主版本。"
  fi
  if python -m pip show transformer-engine-cu13 >/dev/null 2>&1; then
    die "${when}：检测到 transformer-engine-cu13。本环境锁定 CUDA 12.8，只允许 transformer-engine-cu12。已停止。"
  fi
  log "${when}：nvidia-cudnn-cu12=${pkg}，torch.backends.cudnn.version()=${api}"
}


eval "$(read_cudnn)"
log "安装前 cuDNN：nvidia-cudnn-cu12=${pkg}，torch.backends.cudnn.version()=${api}"

if [[ "${pkg}" != "${EXPECTED_CUDNN}" ]]; then
  step_log="${LOG_DIR}/cudnn-install-${RUN_STAMP}.log"
  log "按方案 (a) 安装 nvidia-cudnn-cu12==${EXPECTED_CUDNN}（不升级 PyTorch / CUDA 工具包 / 驱动）"
  run_logged "${step_log}" python -m pip install --upgrade-strategy only-if-needed \
    --constraint "${CONSTRAINTS_FILE}" \
    "nvidia-cudnn-cu12==${EXPECTED_CUDNN}"
else
  log "nvidia-cudnn-cu12 已是锁定的 ${EXPECTED_CUDNN}，跳过重装。"
fi

refresh_cudnn_paths
assert_frozen_stack "cuDNN 安装之后"

eval "$(read_cudnn)"
log "安装后 cuDNN：nvidia-cudnn-cu12=${pkg}，torch.backends.cudnn.version()=${api}"
if [[ "${pkg}" != "${EXPECTED_CUDNN}" ]]; then
  die "nvidia-cudnn-cu12 为 ${pkg}，锁定值是 ${EXPECTED_CUDNN}。已停止，不继续编译 TE。"
fi
if (( api < TE_MIN_CUDNN_INT )); then
  die "torch.backends.cudnn.version()=${api}，仍低于 TE ${EXPECTED_TE} 要求的 ${TE_MIN_CUDNN_INT}（9.12）。可能仍在加载旧库。已停止，不继续编译 TE，也不回退到 2.19.0。"
fi

# ---------------------------------------------------------------------------
# 模型路径只检查，不下载
# ---------------------------------------------------------------------------
if [[ -d "${MODEL_PATH}" ]]; then
  st_count="$(find "${MODEL_PATH}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.safetensors.index.json' \) | wc -l | tr -d ' ')"
  log "模型目录存在：${MODEL_PATH}（顶层 safetensors 相关文件 ${st_count} 个）。不执行 huggingface-cli download。"
  if [[ "${st_count}" == "0" ]]; then
    log "警告：顶层未见 HF safetensors。按锁定要求应使用 HF 权重并排除 original/*。权重不在本脚本范围，继续安装环境。"
  fi
else
  log "警告：模型目录尚不存在：${MODEL_PATH}。不下载权重，继续安装环境。"
fi

python -m pip install --upgrade pip >/dev/null || log "警告：未能升级 pip 本身（可忽略，只要后续安装成功）"

# 构建工具：仅在缺失或 CMake 过旧时安装，不钉死 cmake 主版本
need_build_tools=()
if ! command -v ninja >/dev/null && ! python -c 'import ninja' 2>/dev/null; then
  need_build_tools+=(ninja)
fi
if ! command -v cmake >/dev/null; then
  need_build_tools+=(cmake)
else
  cmake_ok="$(cmake --version | awk 'NR==1{print $3; exit}')"
  python - "${cmake_ok}" <<'PY' || need_build_tools+=(cmake)
import re, sys
raw = sys.argv[1]
parts = []
for token in raw.split("."):
    digits = re.match(r"(\d+)", token)
    parts.append(int(digits.group(1)) if digits else 0)
while len(parts) < 3:
    parts.append(0)
sys.exit(0 if tuple(parts[:3]) >= (3, 18, 0) else 1)
PY
fi
if ! python -c 'import pybind11' 2>/dev/null; then
  need_build_tools+=(pybind11)
fi
if ! python -c 'import packaging' 2>/dev/null; then
  need_build_tools+=(packaging)
fi
if ((${#need_build_tools[@]})); then
  log "安装构建工具：${need_build_tools[*]}"
  run_logged "${LOG_DIR}/build-tools-${RUN_STAMP}.log" \
    python -m pip install --constraint "${CONSTRAINTS_FILE}" "${need_build_tools[@]}"
fi
if ! command -v g++ >/dev/null && ! command -v clang++ >/dev/null; then
  die "找不到 g++ 或 clang++。TE 2.21.0 官方要求 GCC 9+ 或 Clang 10+（C++17），本脚本不擅自用系统包管理器装编译器。"
fi

# ---------------------------------------------------------------------------
# 安装锁定包（不安装 aqlm / autoawq / gptqmodel / auto-gptq）
# ---------------------------------------------------------------------------
# 主机实测（2026-11-26）：nvidia-cudnn-cu12==9.28.1.4 与 torch==2.8.0 写在同一份
# 约束里时，pip 会与 torch 2.8.0 / 2.8.0+cu128 元数据中的 nvidia-cudnn-cu12==9.10.2.21
# 冲突并报 ResolutionImpossible。PyPI 上 transformer-engine-torch 2.21.0 的静态
# Requires-Dist 写的是 transformer_engine_cu13==2.21.0（sdist 在 CUDA 13 环境打包，
# 该字段为 Dynamic），但 setup.py 会按 torch.version.cuda 主版本选择 cu12/cu13；
# 本环境是 2.8.0+cu128，应安装 transformer-engine-cu12，禁止安装 cu13 包。
log "安装 TE ${EXPECTED_TE} 的 CUDA 12 核心包与运行时依赖（不把 torch 的 cuDNN 引脚纳入同一求解）"
# nvdlfw-inspect 依赖 torch>=2.1.0；若和它一起求解，pip 会按 torch 元数据拉取 nvidia-cudnn-cu12==9.10.2.21。
# 因此先装其余依赖，再 --no-deps 安装 nvdlfw-inspect。
run_logged "${LOG_DIR}/te-deps-${RUN_STAMP}.log" \
  python -m pip install --upgrade-strategy only-if-needed \
    --constraint "${CONSTRAINTS_SOLVER_FILE}" \
    "transformer-engine-cu12==${EXPECTED_TE}" \
    einops onnxscript onnx packaging pydantic \
    "nvidia-cudnn-frontend>=1.29.0" "importlib-metadata>=1.0"
assert_frozen_stack "TE 依赖安装之后（不含 nvdlfw-inspect）"
assert_cudnn_pin "TE 依赖安装之后（不含 nvdlfw-inspect）"
run_logged "${LOG_DIR}/te-deps-${RUN_STAMP}.log" \
  python -m pip install --no-deps nvdlfw-inspect
assert_frozen_stack "TE 依赖安装之后"
assert_cudnn_pin "TE 依赖安装之后"

log "安装 Transformer Engine ${EXPECTED_TE}（--no-build-isolation --no-deps 源码编译 torch 插件；不回退到 2.19.0）"
run_logged "${LOG_DIR}/te-install-${RUN_STAMP}.log" \
  python -m pip install --no-build-isolation --no-deps \
    "transformer-engine==${EXPECTED_TE}" \
    "transformer-engine-cu12==${EXPECTED_TE}" \
    "transformer-engine-torch==${EXPECTED_TE}"

assert_frozen_stack "TE 编译之后"
assert_cudnn_pin "TE 编译之后"

log "运行 TE 逐张量 DelayedScaling FP8 冒烟（官方 Getting Started 口径：te.autocast + DelayedScaling）"
step_log="${LOG_DIR}/fp8-smoke-${RUN_STAMP}.log"
set +e
python - >"${step_log}" 2>&1 <<'PY'
import torch
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

if not torch.cuda.is_available():
    raise SystemExit("FP8 冒烟失败：CUDA 不可用")

device = "cuda"
dtype = torch.bfloat16
# 尺寸取 16 的倍数，避免 FP8 GEMM 对齐问题；只做正确性冒烟，不记录为 4090 基线。
layer = te.Linear(256, 256, params_dtype=dtype).to(device)
x = torch.randn(4, 32, 256, dtype=dtype, device=device)
recipe = DelayedScaling(
    fp8_format=Format.HYBRID,
    amax_history_len=16,
    amax_compute_algo="max",
)
with torch.no_grad():
    with te.autocast(enabled=True, recipe=recipe):
        y = layer(x)
torch.cuda.synchronize()
if not torch.isfinite(y).all():
    raise SystemExit("FP8 冒烟失败：输出含非有限值")
print("fp8_smoke_ok")
print("shape", tuple(y.shape), "dtype", str(y.dtype))
print("recipe", "DelayedScaling", "Format.HYBRID")
PY
rc=$?
set -e
cat "${step_log}"
if [[ "${rc}" -ne 0 ]]; then
  fail_step "${step_log}" "TE FP8 冒烟失败。已停止，不改锁定主版本（TE 仍为 ${EXPECTED_TE}）。"
fi
log "FP8 冒烟通过。本步日志：${step_log}"

log "安装 CPU Faiss ${EXPECTED_FAISS}（不装 GPU Faiss）"
run_logged "${LOG_DIR}/faiss-install-${RUN_STAMP}.log" \
  python -m pip install --upgrade-strategy only-if-needed \
    --constraint "${CONSTRAINTS_SOLVER_FILE}" "faiss-cpu==${EXPECTED_FAISS}"
assert_frozen_stack "Faiss 安装之后"
assert_cudnn_pin "Faiss 安装之后"

if python -m pip show faiss-gpu >/dev/null 2>&1 || python -m pip show faiss-gpu-cuvs >/dev/null 2>&1; then
  die "检测到 faiss-gpu 或 faiss-gpu-cuvs。本锁定只允许 CPU Faiss，请卸载 GPU 包后重跑。"
fi

log "安装 transformers / datasets 及 Gate 0 基础工具（这些包不硬依赖 torch，使用不含 cuDNN 引脚的求解约束）"
run_logged "${LOG_DIR}/gate0-tools-${RUN_STAMP}.log" \
  python -m pip install --upgrade-strategy only-if-needed \
    --constraint "${CONSTRAINTS_SOLVER_FILE}" \
    "transformers==${EXPECTED_TRANSFORMERS}" \
    "datasets==${EXPECTED_DATASETS}" \
    "tokenizers==${EXPECTED_TOKENIZERS}" \
    "safetensors==${EXPECTED_SAFETENSORS}" \
    "sentencepiece==${EXPECTED_SENTENCEPIECE}" \
    "einops==${EXPECTED_EINOPS}"
assert_frozen_stack "transformers 等工具安装之后"
assert_cudnn_pin "transformers 等工具安装之后"

log "安装 accelerate ${EXPECTED_ACCELERATE}（硬依赖 torch，改用 --no-deps，避免再次触发 cuDNN 引脚冲突）"
run_logged "${LOG_DIR}/accelerate-install-${RUN_STAMP}.log" \
  python -m pip install --upgrade-strategy only-if-needed \
    --constraint "${CONSTRAINTS_SOLVER_FILE}" \
    "numpy>=1.17" "packaging>=20.0" psutil pyyaml
run_logged "${LOG_DIR}/accelerate-install-${RUN_STAMP}.log" \
  python -m pip install --no-deps "accelerate==${EXPECTED_ACCELERATE}"

log "安装 lm-eval ${EXPECTED_LM_EVAL}（先装不带 [hf] 额外依赖的核心；再 --no-deps 安装 peft ${EXPECTED_PEFT}。查阅于 2026-11-26：peft 0.22.0 要求 torch>=1.13.0，未被 yank）"
run_logged "${LOG_DIR}/lm-eval-install-${RUN_STAMP}.log" \
  python -m pip install --upgrade-strategy only-if-needed \
    --constraint "${CONSTRAINTS_SOLVER_FILE}" \
    "lm-eval==${EXPECTED_LM_EVAL}"
run_logged "${LOG_DIR}/lm-eval-install-${RUN_STAMP}.log" \
  python -m pip install --upgrade-strategy only-if-needed \
    --constraint "${CONSTRAINTS_SOLVER_FILE}" \
    "numpy>=1.17" "packaging>=20.0" psutil pyyaml tqdm
run_logged "${LOG_DIR}/lm-eval-install-${RUN_STAMP}.log" \
  python -m pip install --no-deps "peft==${EXPECTED_PEFT}"
assert_frozen_stack "accelerate / lm-eval 安装之后"
assert_cudnn_pin "accelerate / lm-eval 安装之后"

# ---------------------------------------------------------------------------
# 安装后复检：冻结栈未被改动，锁定包版本正确
# ---------------------------------------------------------------------------
assert_frozen_stack "全部安装之后"

post_check="$(python - <<'PY'
import importlib.metadata as m

def ver(name):
    try:
        return m.version(name)
    except m.PackageNotFoundError:
        return "MISSING"

errors = []
expected = {
    "nvidia-cudnn-cu12": "9.28.1.4",
    "transformer-engine": "2.21.0",
    "transformer-engine-cu12": "2.21.0",
    "transformer-engine-torch": "2.21.0",
    "faiss-cpu": "1.16.0",
    "transformers": "5.24.1",
    "accelerate": "1.16.0",
    "datasets": "5.2.0",
    "lm_eval": "0.4.13",
    "tokenizers": "0.23.3",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.3",
    "einops": "0.8.2",
    "peft": "0.22.0",
}
for name, want in expected.items():
    have = ver(name)
    if have == "MISSING" and name == "lm_eval":
        have = ver("lm-eval")
    if have != want:
        errors.append(f"{name} 为 {have}，期望 {want}")

for forbidden in ("faiss-gpu", "faiss-gpu-cuvs", "auto-gptq", "transformer-engine-cu13"):
    if ver(forbidden) != "MISSING":
        errors.append(f"发现禁止安装的包 {forbidden}=={ver(forbidden)}")

# 已决策本轮不装；若环境里已有要求 torch>=2.14 的 gptqmodel，只告警，不在这里改主版本
gptq = ver("gptqmodel")
if gptq != "MISSING":
    print(f"WARN gptqmodel=={gptq} 已存在于环境中；本脚本未安装它。禁止使用 7.12.2 及任何要求 torch>=2.14 的版本。")

if errors:
    print("FAIL")
    for e in errors:
        print(e)
    raise SystemExit(1)
print("OK")
PY
)"
log "版本复检：${post_check}"

python - <<'PY'
import faiss
import transformer_engine
import transformer_engine.pytorch
import transformers
import accelerate
import datasets
import lm_eval

if hasattr(faiss, "StandardGpuResources") or hasattr(faiss, "GpuIndexFlatL2"):
    raise SystemExit("导入的 faiss 暴露了 GPU 符号，可能误装了 GPU Faiss")

pq = faiss.ProductQuantizer(16, 4, 8)
if not hasattr(pq, "train") or not hasattr(pq, "compute_codes"):
    raise SystemExit("faiss.ProductQuantizer 缺少 train / compute_codes")

print("import_ok", "te", transformer_engine.__version__ if hasattr(transformer_engine, "__version__") else "present")
print("faiss_ProductQuantizer_train_compute_codes=OK")
PY

log "安装与导入校验通过。"

# ---------------------------------------------------------------------------
# 将 conda list / nvidia-smi 写入 VERSION_LOCK.md 标记块（可重入，不无限追加）
# ---------------------------------------------------------------------------
tmp_lock="$(mktemp)"
tmp_block="$(mktemp)"
trap 'rm -f "${tmp_lock}" "${tmp_block}"' EXIT

{
  echo "<!-- BEGIN HOST_SNAPSHOT -->"
  echo "## 主机实测快照（由 setup_env.sh 写入，可重入替换）"
  echo
  echo "- 写入时间（Asia/Shanghai）：$(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "- conda 环境：${CONDA_DEFAULT_ENV}"
  echo "- HF_ENDPOINT=${HF_ENDPOINT}"
  echo "- CUDA_HOME=${CUDA_HOME}"
  echo "- NVTE_CUDA_ARCHS=${NVTE_CUDA_ARCHS}"
  echo "- nvidia-cudnn-cu12：$(python -c 'import importlib.metadata as m; print(m.version("nvidia-cudnn-cu12"))')"
  echo "- torch.backends.cudnn.version()：$(python -c 'import torch; print(torch.backends.cudnn.version())')"
  echo "- 本次日志：${LOG_FILE}"
  echo
  echo "### nvidia-smi"
  echo
  echo '```'
  nvidia-smi
  echo '```'
  echo
  echo "### conda list"
  echo
  echo '```'
  conda list
  echo '```'
  echo
  echo "<!-- END HOST_SNAPSHOT -->"
} > "${tmp_block}"

python - "${LOCK_FILE}" "${tmp_block}" "${tmp_lock}" <<'PY'
import sys
from pathlib import Path

lock_path, block_path, out_path = map(Path, sys.argv[1:])
text = lock_path.read_text(encoding="utf-8")
block = block_path.read_text(encoding="utf-8")
if not block.endswith("\n"):
    block += "\n"
begin = "<!-- BEGIN HOST_SNAPSHOT -->"
end = "<!-- END HOST_SNAPSHOT -->"
start = text.find(begin)
stop = text.find(end)
if start == -1 or stop == -1 or stop < start:
    if not text.endswith("\n"):
        text += "\n"
    text = text + "\n" + block
else:
    stop += len(end)
    if stop < len(text) and text[stop] == "\n":
        stop += 1
    text = text[:start] + block + text[stop:]
out_path.write_text(text, encoding="utf-8")
PY

mv "${tmp_lock}" "${LOCK_FILE}"
log "已将 nvidia-smi 与 conda list 写入 ${LOCK_FILE}（标记块已替换，可重复运行）。"
log "W0 安装完成。完整日志：${LOG_FILE}"
