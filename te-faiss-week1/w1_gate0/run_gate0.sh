#!/usr/bin/env bash
# 在目标主机上激活锁定环境并启动 Gate 0。失败即退出。
# 不登录其它机器，不下载模型权重，不 pip install，不索要或写入密码 / SSH / HF token。
# 不修改 ../VERSION_LOCK.md 与 ../setup_env.sh。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WEEK1_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

EXPECTED_CONDA_ENV="torch2.8_cuda12.8"
FORBIDDEN_CONDA_ENV="torch2.14_cuda13.0"
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
EXPECTED_CUDA_HOME="/usr/local/cuda-12.8"
HF_MIRROR="https://hf-mirror.com"

die() { echo "[gate0][FATAL] $*" >&2; exit 1; }
log() { echo "[gate0] $*" >&2; }

ASSERT_SH="${WEEK1_DIR}/official-4090-24g/assert_official_4090_24g.sh"
if [[ -x "${ASSERT_SH}" ]]; then
  "${ASSERT_SH}" || die "官方 24GB 硬件断言失败，拒绝启动 Gate 0。"
fi

# --- conda ---
[[ -f "${CONDA_SH}" ]] || die "找不到 ${CONDA_SH}。请在目标主机用该路径的 Anaconda，并先 source 再激活 ${EXPECTED_CONDA_ENV}。"
# shellcheck disable=SC1090
source "${CONDA_SH}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${FORBIDDEN_CONDA_ENV}" ]]; then
  die "当前环境是被禁用的 ${FORBIDDEN_CONDA_ENV}，拒绝继续。"
fi
conda activate "${EXPECTED_CONDA_ENV}" || die "无法激活 conda 环境 ${EXPECTED_CONDA_ENV}"
[[ "${CONDA_DEFAULT_ENV:-}" == "${EXPECTED_CONDA_ENV}" ]] \
  || die "激活后环境为「${CONDA_DEFAULT_ENV:-}」，不是锁定的 ${EXPECTED_CONDA_ENV}"

command -v python >/dev/null || die "激活后找不到 python"
command -v nvidia-smi >/dev/null || die "找不到 nvidia-smi"

# --- CUDA / 镜像 / cuDNN 路径（不安装任何包）---
[[ -d "${EXPECTED_CUDA_HOME}" ]] || die "找不到 CUDA 工具包目录 ${EXPECTED_CUDA_HOME}"
export CUDA_HOME="${EXPECTED_CUDA_HOME}"
export CUDA_PATH="${EXPECTED_CUDA_HOME}"
export PATH="${CUDA_HOME}/bin:${PATH}"
if [[ -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
export HF_ENDPOINT="${HF_ENDPOINT:-${HF_MIRROR}}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-${HF_MIRROR}}"
export HF_HUB_DISABLE_IMPLICIT_TOKEN="${HF_HUB_DISABLE_IMPLICIT_TOKEN:-1}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"

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

# --- 本轮输出目录（可重入：每次新时间戳，不覆盖旧日志）---
STAMP="$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="${W1_OUTPUT_DIR:-${SCRIPT_DIR}/results}"
RUN_DIR="${RUN_ROOT}/gate0-${STAMP}"
mkdir -p "${RUN_DIR}" "${SCRIPT_DIR}/logs"
# 把本轮控制台完整抄一份到 logs/，同时写入 run dir
CONSOLE_LOG="${SCRIPT_DIR}/logs/gate0-${STAMP}.log"
ln -sfn "${RUN_DIR}" "${SCRIPT_DIR}/results/latest" || true

log "脚本目录：${SCRIPT_DIR}"
log "版本锁定（只读）：${WEEK1_DIR}/VERSION_LOCK.md"
log "评测协议（只读）：${WEEK1_DIR}/eval_protocol.md"
log "本轮输出：${RUN_DIR}"
log "控制台日志：${CONSOLE_LOG}"
log "不会执行 huggingface-cli download，也不会写入 token。"

cd "${SCRIPT_DIR}"
# 把余下参数交给 Python 编排器；默认跑 bf16,te_fp8。
set +e
python "${SCRIPT_DIR}/run_gate0.py" --run-dir "${RUN_DIR}" "$@" 2>&1 | tee "${CONSOLE_LOG}" | tee -a "${RUN_DIR}/console.log"
rc=${PIPESTATUS[0]}
set -e
exit "${rc}"
