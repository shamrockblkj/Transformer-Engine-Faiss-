#!/usr/bin/env bash
# 第 1 周 W3：在目标主机（晨涧云 RTX 4090，conda 环境 torch2.8_cuda12.8）上跑离线 PQ。
# 不下载权重或语料，不索要令牌，不安装 AWQ/GPTQ/AQLM，不推 GitHub。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ASSERT_SH="$(cd "${ROOT}/.." && pwd)/official-4090-24g/assert_official_4090_24g.sh"
if [[ -x "${ASSERT_SH}" ]]; then
  "${ASSERT_SH}" || { echo "FATAL: 官方 24GB 硬件断言失败，拒绝启动 W3。" >&2; exit 2; }
fi
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "FATAL: 找不到 ${CONDA_SH}。非交互 shell 须先 source 该文件再激活 torch2.8_cuda12.8。" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate torch2.8_cuda12.8

# 强制离线，避免 transformers / huggingface_hub 访问网络。不读写 token。
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${ROOT}/logs" "${ROOT}/results"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="${ROOT}/logs/w3_pq-${STAMP}.log"

echo "INFO: log=${LOG}"
echo "INFO: python=$(command -v python) python_version=$(python -c 'import sys; print(sys.version.split()[0])')"
echo "INFO: 正式路径硬检 conda=torch2.8_cuda12.8（禁用 torch2.14_cuda13.0）、GPU 名称含 4090 且 sm_89；Tesla P4 / 其它卡 / --device cpu 一律失败。PQ 训练走 CPU faiss。"

# 所有参数原样转交给 Python。常用：
#   --dry-run
#   --self-check
#   --groups all
#   --groups plain,row_rms,sigma,row_rms_sigma
#   --skip-kl
#   --weights-only   # 仅当校准语料未就绪、只想检查旋转域重建时
python -u "${ROOT}/run_pq_single_layer.py" --config "${ROOT}/configs/default.json" "$@" 2>&1 | tee "${LOG}"
