#!/usr/bin/env bash
# Optional Nsight Compute pass for L2 traffic and shared-memory bank conflicts.
# Not required for the default timing run. If ncu is missing, exit with a
# clear error and leave those columns as 待实测 — do not invent numbers.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="${CUDA_HOME}/bin:${PATH}"

NCU="${NCU:-ncu}"
if ! command -v "${NCU}" >/dev/null 2>&1 && [[ -x "${CUDA_HOME}/bin/ncu" ]]; then
  NCU="${CUDA_HOME}/bin/ncu"
fi
if ! command -v "${NCU}" >/dev/null 2>&1; then
  echo "error: ncu not found (looked in PATH and ${CUDA_HOME}/bin)." >&2
  echo "  L2 traffic and Nsight bank-conflict columns stay 待实测." >&2
  echo "  Do not substitute the plan B2.1 estimate (1.41e9 * 16 B ≈ 22 GB) as a measurement." >&2
  exit 2
fi

BIN="${ROOT}/out/gather_microbench"
if [[ ! -x "${BIN}" ]]; then
  make -C "${ROOT}"
fi

OUT_DIR="${ROOT}/out/ncu"
mkdir -p "${OUT_DIR}"

# Metric names follow Nsight Compute 2024+/CUDA 12.x style. If a name is
# rejected on the host's ncu, re-run with `ncu --query-metrics` and record
# the actual names in the report; do not invent a byte count.
METRICS="dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_bytes_srcunit_tex_op_read.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_requests_mem_shared_op_ld.sum,smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct"

# Shortened protocol so ncu stays tractable. Timing numbers from ncu are
# not the official median; official times come from run_4090.sh / CUDA events.
# Skip the pre-timing L2-vs-SMEM check so path-specific ncu metrics are not
# mixed with the other kernel. Correctness is covered by the default timing run.
COMMON=(--warmup 1 --iters 3 --layers 1 --seed 42 --skip-partial-check)

echo "=== ncu L2-direct (layer=1 shortened run; not the official median) ==="
"${NCU}" --target-processes all \
  --metrics "${METRICS}" \
  --csv \
  -o "${OUT_DIR}/l2_layer1" \
  "${BIN}" --path l2 "${COMMON[@]}" \
  || { echo "error: ncu L2 pass failed. Leave L2 traffic as 待实测." >&2; exit 2; }

echo "=== ncu SMEM-tile (layer=1 shortened run; not the official median) ==="
"${NCU}" --target-processes all \
  --metrics "${METRICS}" \
  --csv \
  -o "${OUT_DIR}/smem_layer1" \
  "${BIN}" --path smem "${COMMON[@]}" \
  || { echo "error: ncu SMEM pass failed. Leave bank-conflict Nsight cells as 待实测." >&2; exit 2; }

echo "Nsight reports under ${OUT_DIR} (*.ncu-rep / csv depending on ncu version)."
echo "Scale layer-1 traffic to 32 layers only if the report states the scaling"
echo "assumption; default is to also collect a --layers 32 pass when time allows."
