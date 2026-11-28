#!/usr/bin/env bash
# 预下载 V2.8 冻结栈的官方 PyPI / PyTorch cu128 轮子到本目录 wheelhouse/。
# 可重入：已存在的目标文件会跳过。不下载模型权重，不安装 AWQ/GPTQ/AQLM/GGUF。
# 禁止把求解过程中的 torch>=2.14 / nvidia-cublas-cu12 12.9 写入仓库。
# nvidia-cudnn-cu12 9.10.2.21 只作为 torch 2.8.0+cu128 元数据依赖缓存；
# 最终安装引脚必须是 9.28.1.4，求解约束里不得钉死 9.10.2.21。
# 目标解释器：Linux x86_64 / CPython 3.12，不是本盒系统 Python。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEELHOUSE="${WHEELHOUSE:-${SCRIPT_DIR}/wheelhouse}"
MANIFEST="${WHEELHOUSE}/MANIFEST.txt"
CONSTRAINTS="${SCRIPT_DIR}/wheelhouse-constraints.txt"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${WHEELHOUSE}" "${LOG_DIR}"
STAMP="$(TZ=Asia/Shanghai date '+%Y%m%d-%H%M%S')"
LOG="${LOG_DIR}/download_wheelhouse-${STAMP}.log"

PY="${PYTHON:-python3}"
PIP=( "${PY}" -m pip )

PIP_PY312_BIN=(
  --python-version 3.12
  --implementation cp
  --abi cp312
  --platform manylinux_2_28_x86_64
  --platform manylinux_2_27_x86_64
  --platform manylinux2014_x86_64
  --platform linux_x86_64
  --only-binary=:all:
)

log() { printf '[wheelhouse] %s\n' "$*" | tee -a "${LOG}"; }
die() { printf '[wheelhouse] 错误：%s\n' "$*" | tee -a "${LOG}" >&2; exit 1; }

have_spec() {
  # $1 = pep 427 normalized prefix, e.g. nvidia_cudnn_cu12-9.28.1.4
  local prefix="$1"
  compgen -G "${WHEELHOUSE}/${prefix}*" >/dev/null 2>&1
}

reject_unwanted() {
  local bad
  bad="$(ls -1 "${WHEELHOUSE}" 2>/dev/null | grep -E '^(torch-2\.(1[4-9]|[2-9][0-9])|nvidia_cublas_cu12-12\.9|transformer_engine_cu13)' || true)"
  if [[ -n "${bad}" ]]; then
    die "wheelhouse 出现禁止缓存的包，已停止。请删除后重跑：${bad}"
  fi
}

download_one_bin() {
  local spec="$1"
  log "下载（二进制 --no-deps，cp312）：${spec}"
  "${PIP[@]}" download -d "${WHEELHOUSE}" --no-deps "${PIP_PY312_BIN[@]}" "${spec}" >>"${LOG}" 2>&1 \
    || die "二进制下载失败：${spec}（详见 ${LOG}）"
}

download_one_any() {
  local spec="$1"
  log "下载（允许 sdist，--no-deps，Python 3.12）：${spec}"
  "${PIP[@]}" download -d "${WHEELHOUSE}" --no-deps --python-version 3.12 "${spec}" >>"${LOG}" 2>&1 \
    || die "下载失败：${spec}（详见 ${LOG}）"
}

log "开始：${STAMP}（Asia/Shanghai）"
log "目标目录：${WHEELHOUSE}"
log "日志：${LOG}"
"${PIP[@]}" --version | tee -a "${LOG}"

# ---------------------------------------------------------------------------
# 1) 冻结的 PyTorch 2.8.0+cu128（官方 cu128 索引）
# ---------------------------------------------------------------------------
if have_spec "torch-2.8.0+cu128"; then
  log "已存在 torch-2.8.0+cu128，跳过重下。"
else
  log "下载 torch==2.8.0+cu128 及其官方 cu128 依赖（可能含 nvidia-cudnn-cu12==9.10.2.21）"
  "${PIP[@]}" download -d "${WHEELHOUSE}" "${PIP_PY312_BIN[@]}" \
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.8.0+cu128" >>"${LOG}" 2>&1 \
    || die "torch==2.8.0+cu128 下载失败（详见 ${LOG}）"
fi
reject_unwanted
have_spec "torch-2.8.0+cu128" || die "未找到 torch-2.8.0+cu128 轮子"
have_spec "nvidia_cublas_cu12-12.8.4.1" || log "警告：未看到 nvidia-cublas-cu12 12.8.4.1，请核对本步日志。"

# ---------------------------------------------------------------------------
# 2) 锁定包：一律 --no-deps，避免 accelerate/lm-eval 从 PyPI 解出 torch 2.15
# ---------------------------------------------------------------------------
BIN_PINS=(
  "huggingface_hub==2.4.0"
  "nvidia-cudnn-cu12==9.28.1.4"
  "faiss-cpu==1.16.0"
  "transformers==5.24.1"
  "accelerate==1.16.0"
  "datasets==5.2.0"
  "lm-eval==0.4.13"
  "tokenizers==0.23.3"
  "safetensors==0.8.0"
  "sentencepiece==0.2.3"
  "peft==0.22.0"
  "einops==0.8.2"
  "ninja"
  "cmake"
  "pybind11"
  "packaging"
  "pydantic"
  "importlib-metadata"
  "psutil"
  "pyyaml"
  "tqdm"
  "numpy"
)

for spec in "${BIN_PINS[@]}"; do
  download_one_bin "${spec}"
done

ANY_PINS=(
  "transformer-engine==2.21.0"
  "transformer-engine-cu12==2.21.0"
  "transformer-engine-torch==2.21.0"
  "nvidia-cudnn-frontend>=1.29.0"
  "nvdlfw-inspect"
  "onnxscript"
  "onnx"
  "sqlitedict"
  "evaluate>=0.4.0"
  "rouge-score>=0.0.4"
  "sacrebleu>=1.5.0"
  "scikit-learn>=0.24.1"
  "pytablewriter"
  "dill"
  "pandas"
  "pyarrow"
  "xxhash"
  "multiprocess"
  "regex"
  "typer"
  "httpx2>=2.0.0"
  "hf_xet"
  "filelock"
  "fsspec"
  "jinja2"
  "markupsafe"
  "typing_extensions"
)

for spec in "${ANY_PINS[@]}"; do
  download_one_any "${spec}"
done

reject_unwanted
have_spec "nvidia_cudnn_cu12-9.28.1.4" || die "未找到 nvidia-cudnn-cu12 9.28.1.4 轮子"

# ---------------------------------------------------------------------------
# 3) 清单
# ---------------------------------------------------------------------------
{
  echo "# official-4090-24g wheelhouse manifest"
  echo "# generated: $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S %Z')"
  echo "# purpose: cache only; install still must assert frozen pins after setup_env.sh"
  echo "# final nvidia-cudnn-cu12 must be 9.28.1.4"
  echo "# 9.10.2.21 may be present as torch 2.8.0+cu128 metadata wheel only"
  echo "# no model weights, no AWQ/GPTQ/AQLM/GGUF, no transformer-engine-cu13"
  echo
  ls -1 "${WHEELHOUSE}" | grep -v -E '^(MANIFEST.txt|SHA256SUMS)$' | sort | while read -r f; do
    sz="$(wc -c < "${WHEELHOUSE}/${f}" | tr -d ' ')"
    printf '%s  %s bytes\n' "${f}" "${sz}"
  done
} > "${MANIFEST}"

if command -v sha256sum >/dev/null; then
  (
    cd "${WHEELHOUSE}"
    ls -1 | grep -v -E '^(MANIFEST.txt|SHA256SUMS)$' | sort | xargs -r sha256sum
  ) > "${WHEELHOUSE}/SHA256SUMS"
fi

log "完成。文件数：$(ls -1 "${WHEELHOUSE}" | grep -c -v -E '^(MANIFEST.txt|SHA256SUMS)$' || true)"
log "清单：${MANIFEST}"
if have_spec "nvidia_cudnn_cu12-9.10.2.21"; then
  log "注意：wheelhouse 中存在 torch 元数据依赖的 nvidia-cudnn-cu12 9.10.2.21；最终安装必须是 9.28.1.4。"
fi
log "OK"
