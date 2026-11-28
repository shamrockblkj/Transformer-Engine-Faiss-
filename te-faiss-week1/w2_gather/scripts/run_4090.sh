#!/usr/bin/env bash
# Build (if needed) and run the W2 gather microbench on the target 4090 host.
# Do not run this on Tesla P4 or any non-target machine.
# Default run enables the L2 vs SMEM partial-sum check before timing.
# Do not add --skip-partial-check here unless debugging.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

ASSERT_SH="$(cd "${ROOT}/.." && pwd)/official-4090-24g/assert_official_4090_24g.sh"
if [[ -x "${ASSERT_SH}" ]]; then
  "${ASSERT_SH}" || { echo "error: official 24GB hardware assert failed; refusing to start W2 benches." >&2; exit 2; }
fi
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"

if [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # Non-interactive shells on the target host need this (VERSION_LOCK).
  # The microbench itself does not require the conda env.
  # shellcheck disable=SC1091
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "error: nvidia-smi not found. Run this script on the Chenjian Cloud 4090 host." >&2
  exit 2
fi

echo "=== nvidia-smi (identity only; this is not a timing result) ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || true

make -C "${ROOT}"

mkdir -p "${ROOT}/out"
JSON="${ROOT}/out/gather_microbench.json"
"${ROOT}/out/gather_microbench" \
  --path both \
  --warmup 5 \
  --iters 11 \
  --layers 32 \
  --seed 42 \
  --json "${JSON}" \
  "$@"

echo "JSON written to ${JSON}"
echo "Fill gather_microbench_report.md from that JSON. Leave S2 threshold blank."
