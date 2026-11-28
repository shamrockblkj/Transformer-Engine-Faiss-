#!/usr/bin/env bash
# 在新租的官方 4090 24GB 主机上创建或激活冻结 conda 环境 torch2.8_cuda12.8，
# 并安装 PyTorch 2.8.0+cu128（若尚未安装）。不安装 TE / Faiss / 量化包。
# 随后应运行仓库根目录的 setup_env.sh。
# 不假定主机布局：可通过 CONDA_SH / WHEELHOUSE 覆盖；找不到 conda 则失败并说明，不擅自安装 Anaconda。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEEK1_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXPECTED_CONDA_ENV="torch2.8_cuda12.8"
FORBIDDEN_CONDA_ENV="torch2.14_cuda13.0"
EXPECTED_PYTHON="3.12.0"
EXPECTED_TORCH="2.8.0+cu128"
WHEELHOUSE="${WHEELHOUSE:-${SCRIPT_DIR}/wheelhouse}"

die() { printf '[create-env][FATAL] %s\n' "$*" >&2; exit 1; }
log() { printf '[create-env] %s\n' "$*"; }

find_conda_sh() {
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    printf '%s\n' "${CONDA_SH}"
    return 0
  fi
  local cand
  for cand in \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "/opt/conda/etc/profile.d/conda.sh" \
    "/opt/anaconda3/etc/profile.d/conda.sh" \
    "/usr/local/anaconda3/etc/profile.d/conda.sh"
  do
    if [[ -f "${cand}" ]]; then
      printf '%s\n' "${cand}"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
      printf '%s\n' "${base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  return 1
}

CONDA_SH="$(find_conda_sh)" || die "找不到 conda.sh。非交互 SSH 通常需要 source ~/anaconda3/etc/profile.d/conda.sh，但新主机布局待实测，请设置 CONDA_SH=/path/to/conda.sh 后重跑。本脚本不擅自安装 Anaconda。"
# shellcheck disable=SC1090
source "${CONDA_SH}"
log "已 source：${CONDA_SH}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "${FORBIDDEN_CONDA_ENV}" ]]; then
  die "当前环境是被禁用的 ${FORBIDDEN_CONDA_ENV}。"
fi

if conda env list | awk '{print $1}' | grep -qx "${FORBIDDEN_CONDA_ENV}"; then
  log "警告：主机上存在 ${FORBIDDEN_CONDA_ENV}，本脚本不会激活或向其中安装。"
fi

if conda env list | awk '{print $1}' | grep -qx "${EXPECTED_CONDA_ENV}"; then
  log "环境 ${EXPECTED_CONDA_ENV} 已存在，直接激活。"
else
  log "创建环境 ${EXPECTED_CONDA_ENV}，Python ${EXPECTED_PYTHON}（锁定值，不是最新 3.12.x）。"
  conda create -y -n "${EXPECTED_CONDA_ENV}" "python=${EXPECTED_PYTHON}"
fi

conda activate "${EXPECTED_CONDA_ENV}" || die "无法激活 ${EXPECTED_CONDA_ENV}"
[[ "${CONDA_DEFAULT_ENV:-}" == "${EXPECTED_CONDA_ENV}" ]] || die "激活后环境为 ${CONDA_DEFAULT_ENV:-}，不是 ${EXPECTED_CONDA_ENV}"

pyver="$(python -c 'import sys; print("%d.%d.%d"%sys.version_info[:3])')"
[[ "${pyver}" == "${EXPECTED_PYTHON}" ]] || die "Python 为 ${pyver}，锁定值是 ${EXPECTED_PYTHON}。请重建该环境，不要用 3.12.x 最新版凑合。"

need_torch=1
if python -c 'import torch' >/dev/null 2>&1; then
  have="$(python -c 'import torch; print(torch.__version__)')"
  if [[ "${have}" == "${EXPECTED_TORCH}" ]]; then
    log "PyTorch 已是锁定的 ${EXPECTED_TORCH}，跳过安装。"
    need_torch=0
  else
    die "环境中 PyTorch 为 ${have}，锁定值是 ${EXPECTED_TORCH}。拒绝自动升级或降级，请换干净环境后重跑。"
  fi
fi

if [[ "${need_torch}" -eq 1 ]]; then
  pip_args=(python -m pip install --upgrade-strategy only-if-needed "torch==${EXPECTED_TORCH}")
  if [[ -d "${WHEELHOUSE}" ]] && compgen -G "${WHEELHOUSE}/torch-2.8.0+cu128*" >/dev/null; then
    log "使用本地 wheelhouse：${WHEELHOUSE}"
    export PIP_FIND_LINKS="${WHEELHOUSE}"
    pip_args+=( --find-links "${WHEELHOUSE}" )
    if [[ "${WHEELHOUSE_OFFLINE:-0}" == "1" ]]; then
      pip_args+=( --no-index )
    fi
  else
    log "wheelhouse 中没有 torch 轮子，改用官方 PyTorch cu128 索引。"
    pip_args+=( --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple )
  fi
  "${pip_args[@]}"
fi

python -c '
import torch, sys
exp = "2.8.0+cu128"
if torch.__version__ != exp:
    print("PyTorch=%s, expected %s" % (torch.__version__, exp))
    sys.exit(1)
if (torch.version.cuda or "") != "12.8":
    print("torch.version.cuda=%s, expected 12.8" % (torch.version.cuda,))
    sys.exit(1)
print("OK", torch.__version__, "cuda", torch.version.cuda)
' || die "安装后 PyTorch 校验失败。"

log "环境就绪。下一步："
log "  export WHEELHOUSE=${WHEELHOUSE}"
log "  export PIP_FIND_LINKS=\${WHEELHOUSE}"
log "  bash ${WEEK1_DIR}/setup_env.sh"
log "不要在本环境安装 AWQ/GPTQ/AQLM/GGUF，也不要使用 ${FORBIDDEN_CONDA_ENV}。"
