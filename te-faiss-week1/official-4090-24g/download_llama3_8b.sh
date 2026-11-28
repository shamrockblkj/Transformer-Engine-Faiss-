#!/usr/bin/env bash
# 仅在新租的官方 4090 24GB 主机上、硬件断言通过后，下载官方 Meta-Llama-3-8B
#（Hugging Face safetensors，排除 original/*）。预期占用约 16 GB。
# 默认 HF_ENDPOINT=https://hf-mirror.com，允许覆盖。
# 若仓库 gated，只从环境变量读取令牌（HF_TOKEN 或 HUGGING_FACE_HUB_TOKEN），绝不写入文件或日志。
# 不下载 Qwen2.5-32B / Gemma 2 27B / 其它权重，也不启动第二路并发下载。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-meta-llama/Meta-Llama-3-8B}"
DEST="${MODEL_PATH:-/data/models/Meta-Llama-3-8B}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HUGGINGFACE_HUB_ENDPOINT="${HUGGINGFACE_HUB_ENDPOINT:-${HF_ENDPOINT}}"

die() { printf '[download-8b][FATAL] %s\n' "$*" >&2; exit 1; }
log() { printf '[download-8b] %s\n' "$*"; }

if [[ -n "${HF_TOKEN:-}" || -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  log "检测到环境变量中的 HF 令牌（不会打印或写入文件）。"
else
  log "未设置 HF_TOKEN / HUGGING_FACE_HUB_TOKEN。若 Meta-Llama-3-8B 需要门禁授权，请只在环境变量中提供，不要写入磁盘。"
fi

if [[ -x "${SCRIPT_DIR}/assert_official_4090_24g.sh" ]]; then
  "${SCRIPT_DIR}/assert_official_4090_24g.sh" || die "硬件断言失败，拒绝下载。"
fi

LOCK_DIR="${DEST}.download.lock"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  die "检测到 ${LOCK_DIR}，可能已有另一路下载。本脚本不启动第二路并发下载。"
fi
cleanup() { rmdir "${LOCK_DIR}" 2>/dev/null || true; }
trap cleanup EXIT

mkdir -p "$(dirname "${DEST}")"
log "仓库：${REPO}"
log "目标：${DEST}"
log "HF_ENDPOINT=${HF_ENDPOINT}"
log "预期磁盘约 16 GB（官方 BF16/FP16 safetensors 量级，不是本机已测占用）。排除 original/*。"

if command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${REPO}" \
    --local-dir "${DEST}" \
    --local-dir-use-symlinks False \
    --exclude "original/*" "original/**" \
    --resume-download
elif python -c 'import huggingface_hub' >/dev/null 2>&1; then
  export REPO DEST
  python -c "
import os, sys
from huggingface_hub import snapshot_download
token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN') or None
snapshot_download(
    repo_id=os.environ['REPO'],
    local_dir=os.environ['DEST'],
    local_dir_use_symlinks=False,
    resume_download=True,
    token=token,
    ignore_patterns=['original/*', 'original/**'],
    endpoint=os.environ.get('HF_ENDPOINT') or None,
)
print('OK', os.environ['DEST'])
" 
else
  die "找不到 huggingface-cli 或 huggingface_hub。请先创建冻结环境并运行 setup_env.sh，再执行本脚本。"
fi

st_count="$(find "${DEST}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.safetensors.index.json' \) | wc -l | tr -d ' ')"
[[ "${st_count}" != "0" ]] || die "下载后 ${DEST} 顶层未见 HF safetensors。应按锁定使用 HF 权重并排除 original/*。"
log "完成：${DEST} 顶层 safetensors 相关文件 ${st_count} 个。"
if [[ -d "${DEST}/original" ]]; then
  log "提示：目录中仍有 original/，评测脚本会排除它，不要用其中的权重。"
fi
