#!/usr/bin/env bash
# 官方 GeForce RTX 4090 24GB 硬件断言。
# 在任何 V2.8 官方验收安装 / Gate 0 / W2 / W3 之前运行。
# 必须确认「单张、官方档 24GB」GPU；改装 48G（约 49140 MiB）或非约 24GB 立即失败。
# 不建议、不使用 MIG。软件显存上限 / CUDA_VISIBLE_DEVICES 伪装不是官方 24GB 验收，本脚本不提供该路径。
# 不登录其它主机，不写密码或令牌。
set -euo pipefail

# 官方 24GB 档：nvidia-smi memory.total 典型约 24564 MiB（24 GiB 量级）。
# 接受范围 24000–25000 MiB，避免把 24GB 与 48GB / 16GB / 专业卡 48GB 混为一谈。
# 24564 是公开档常见读数，不是本机已测值；连接后以 nvidia-smi 为准。
MIN_MIB="${OFFICIAL_24G_MIN_MIB:-24000}"
MAX_MIB="${OFFICIAL_24G_MAX_MIB:-25000}"
TYPICAL_MIB=24564
# 上一台改装 4090 48G 的记录为 49140 MiB。
MOD48_MIN_MIB=48000
MOD48_MAX_MIB=51000

die() {
  printf '[hw-assert][FATAL] %s\n' "$*" >&2
  printf '[hw-assert][FATAL] 这不是 V2.8 官方 GeForce RTX 4090 24GB 验收机。拒绝继续。\n' >&2
  printf '[hw-assert][FATAL] 改装 48G 或非 24GB 卡上的结果不能作为官方 24GB 验收。\n' >&2
  printf '[hw-assert][FATAL] 软件显存限制不能替代官方 24GB 硬件，本脚本不提供该路径，也不使用 MIG。\n' >&2
  exit 1
}
log() { printf '[hw-assert] %s\n' "$*"; }

command -v nvidia-smi >/dev/null 2>&1 || die "找不到 nvidia-smi，无法确认 GPU。"

# 不用 CUDA_VISIBLE_DEVICES 过滤 nvidia-smi：官方验收看整机真实可见 GPU。
mapfile -t NAMES < <(nvidia-smi --query-gpu=name --format=csv,noheader)
mapfile -t MEMS < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
mapfile -t DRIVERS < <(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
# compute_cap 在较新驱动可用；缺失时不编造，改为记录「待用 torch 复检 sm_89」。
HAS_CC=0
CCS=()
if nvidia-smi --query-gpu=compute_cap --format=csv,noheader >/tmp/official-4090-cc.$$ 2>/dev/null; then
  if grep -qE '^[0-9]+\.[0-9]+' /tmp/official-4090-cc.$$; then
    mapfile -t CCS < /tmp/official-4090-cc.$$
    HAS_CC=1
  fi
fi
rm -f /tmp/official-4090-cc.$$

N="${#NAMES[@]}"
[[ "${N}" -ge 1 ]] || die "nvidia-smi 未报告任何 GPU。"
if [[ "${N}" -ne 1 ]]; then
  die "官方验收要求整机只有一张 GPU，当前 nvidia-smi 报告 ${N} 张：${NAMES[*]}。不要用 CUDA_VISIBLE_DEVICES 或软件显存上限把多卡/大显存卡伪装成单张 24GB。"
fi

NAME="$(echo "${NAMES[0]}" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
MEM_RAW="$(echo "${MEMS[0]}" | tr -d ' ')"
DRIVER="$(echo "${DRIVERS[0]}" | tr -d ' ')"

if [[ ! "${MEM_RAW}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
  die "无法解析显存读数：${MEMS[0]@Q}"
fi
MEM="${MEM_RAW%%.*}"

log "nvidia-smi 名称：${NAME}"
log "nvidia-smi 显存：${MEM} MiB（官方 24GB 档接受范围 ${MIN_MIB}–${MAX_MIB} MiB，典型约 ${TYPICAL_MIB} MiB）"
log "驱动版本（只记录，新主机待写入快照，不沿用已停用 48G 主机的 580.159.03 作为硬门槛）：${DRIVER}"

# 禁卡：Tesla P4、本盒测量、专业 48G 等。
if echo "${NAME}" | grep -Eiq 'tesla[[:space:]]*p4|[[:space:]]P4$|^P4$'; then
  die "GPU 名称「${NAME}」像 Tesla P4。本盒 P4 不得作为测量机。"
fi
if echo "${NAME}" | grep -Eiq '6000|5000|A100|H100|H200|B200|V100|A6000|L40|L20'; then
  die "GPU 名称「${NAME}」不是 GeForce RTX 4090。RTX 6000 Ada 等专业 48GB 卡不能当作「官方 4090 24G」。"
fi
# 4090 D 是中国市场另一 SKU（24GB，但 CUDA 核心数与标准 GeForce RTX 4090 不同），不能混进官方 4090 验收。
if echo "${NAME}" | grep -Eiq '4090[[:space:]]*D'; then
  die "GPU 名称「${NAME}」像 GeForce RTX 4090 D，不是标准 GeForce RTX 4090。结果不能作为 V2.8 官方 4090 24GB 验收。"
fi
if ! echo "${NAME}" | grep -Eiq '4090'; then
  die "GPU 名称「${NAME}」不含 4090。"
fi
if ! echo "${NAME}" | grep -Eiq 'RTX'; then
  die "GPU 名称「${NAME}」不含 RTX，不像 GeForce RTX 4090。"
fi

# 先拦改装 48G，避免落入「不是 24GB」的笼统报错。
if (( MEM >= MOD48_MIN_MIB && MEM <= MOD48_MAX_MIB )); then
  die "显存 ${MEM} MiB 落在改装 4090 48G 档（历史记录约 49140 MiB）。用户已要求不再连接上一台改装机；即使是另一张 48G 改装卡，其结果也不能作为 V2.8 官方 24GB 验收。"
fi
if (( MEM > MAX_MIB )); then
  die "显存 ${MEM} MiB 高于官方 24GB 档上限 ${MAX_MIB} MiB，不是官方 GeForce RTX 4090 24GB。"
fi
if (( MEM < MIN_MIB )); then
  die "显存 ${MEM} MiB 低于官方 24GB 档下限 ${MIN_MIB} MiB，不是官方 GeForce RTX 4090 24GB。"
fi

if [[ "${HAS_CC}" -eq 1 ]]; then
  CC="$(echo "${CCS[0]}" | tr -d ' ')"
  log "compute_cap：${CC}"
  if [[ "${CC}" != "8.9" ]]; then
    die "计算能力为 ${CC}，官方 GeForce RTX 4090 应为 sm_89（8.9）。"
  fi
else
  log "警告：当前 nvidia-smi 不能查询 compute_cap。名称与 24GB 显存已通过；请在激活 torch2.8_cuda12.8 后用 torch.cuda.get_device_capability(0) 确认是 (8, 9)。setup_env.sh / Gate 0 会再检。"
fi

# 打印简表，便于写入快照。
nvidia-smi --query-gpu=index,name,compute_cap,memory.total,driver_version --format=csv || \
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv

log "通过：单张官方档 24GB GPU（名称含 RTX 4090，显存 ${MEM} MiB 在 ${MIN_MIB}–${MAX_MIB}）。"
log "可以继续安装与 Gate 0。不要把非 24GB 或改装卡结果写入官方验收。"
exit 0
