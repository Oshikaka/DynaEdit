#!/bin/bash
#SBATCH -A csc662
#SBATCH -J dyna-sweep
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 02:00:00
#
# 单节点 8 个 GCD 并行跑一批编辑。
#
#   sbatch slurm/run_sweep.sbatch joblist.txt
#
# joblist.txt 每行一个任务： <exp_yaml> <item_name> <exp_name>
# 例：
#   configs/phase2.yaml swan_take_off p2_base
#   configs/phase2.yaml swan_take_off p2_sga_t0.001
#
# 任务静态分配给 8 个 worker（worker g 领 g, g+8, g+16 ...）。
# 单条约 35 分钟（N=40），所以 2 小时墙钟内每个 worker 能跑 3 条，即一批最多 24 条。

set -uo pipefail
JOBLIST="${1:?用法: sbatch slurm/run_sweep.sbatch <joblist.txt>}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load miniforge3/23.11.0-0
module load PrgEnv-gnu rocm/6.4.0 craype-accel-amd-gfx90a
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dynaedit
export PYTHONNOUSERSITE=1

export HF_HOME=/lustre/orion/csc662/proj-shared/xinru/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=7          # 56 核 / 8 进程

NGPU=${NGPU:-8}
mapfile -t TASKS < <(grep -vE '^\s*(#|$)' "$JOBLIST")
echo "job=${SLURM_JOB_ID} node=$(hostname) 任务数=${#TASKS[@]} 并行度=${NGPU}"
printf '  %s\n' "${TASKS[@]}"

worker() {
  local gid=$1 k exp item slot rc
  for ((k=gid; k<${#TASKS[@]}; k+=NGPU)); do
    read -r exp item expname <<< "${TASKS[$k]}"
    slot="logs/sweep-${SLURM_JOB_ID}-gpu${gid}-${expname:-all}-${item}.log"
    # MIOpen 的 kernel cache 每进程独立，否则 8 个进程会锁在一起
    export MIOPEN_USER_DB_PATH="/tmp/miopen-${USER}-${SLURM_JOB_ID}-${gid}"
    export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
    mkdir -p "${MIOPEN_USER_DB_PATH}"
    echo "[gpu${gid}] 开始 ${expname:-all} / ${item} -> ${slot}"
    ROCR_VISIBLE_DEVICES=${gid} python run_script.py \
        --exp_yaml "${exp}" --items "${item}" --device_number 0 \
        ${expname:+--exps "${expname}"} > "${slot}" 2>&1
    rc=$?
    echo "[gpu${gid}] 结束 ${expname:-all} / ${item} rc=${rc}"
    rm -rf "${MIOPEN_USER_DB_PATH}"
  done
}

for ((g=0; g<NGPU; g++)); do worker "$g" & done
wait
echo "sweep 全部结束"
