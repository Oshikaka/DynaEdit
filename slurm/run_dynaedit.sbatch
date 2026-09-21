#!/bin/bash
#SBATCH -A csc662
#SBATCH -J dynaedit
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -t 02:00:00
#
# 用法:
#   sbatch slurm/run_dynaedit.sbatch                          # 默认 configs/wan_i2v_exp.yaml
#   sbatch slurm/run_dynaedit.sbatch configs/smoke.yaml
#   sbatch slurm/run_dynaedit.sbatch configs/wan_i2v_exp.yaml swan_take_off   # 只跑一条
#
# Frontier 的 batch 分区按节点数分档限制墙钟，1-8 节点通常是 2 小时。
# 单条 832x480 / 49 帧 / 50 步的编辑估计 1-2 小时，所以一个作业跑一条比较稳妥；
# 要一次跑多条就拆成多个作业，或用 extended 分区。

set -euo pipefail
EXP="${1:-configs/wan_i2v_exp.yaml}"
ITEMS="${2:-}"
cd "${SLURM_SUBMIT_DIR:-$PWD}"

module load miniforge3/23.11.0-0
module load PrgEnv-gnu rocm/6.4.0 craype-accel-amd-gfx90a
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dynaedit
export PYTHONNOUSERSITE=1

# 计算节点没有外网：权重必须已在登录节点下载好，这里纯离线加载
export HF_HOME=/lustre/orion/csc662/proj-shared/xinru/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# MIOpen 的 kernel cache 默认写 $HOME(NFS)，多进程会锁死且很慢 -> 放节点本地
export MIOPEN_USER_DB_PATH="/tmp/miopen-${USER}-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${MIOPEN_USER_DB_PATH}"

# 单 GCD 只有 64 GB，分配器碎片化会直接 OOM。
# 注意 expandable_segments 在 ROCm 上不支持（torch 会 warn 后忽略），
# 用 garbage_collection_threshold：占用高时主动回收缓存块。
export PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.9
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=7

echo "job=${SLURM_JOB_ID} node=$(hostname) exp=${EXP}"
srun -N1 -n1 -c56 --gpus-per-node=8 --gpu-bind=none \
    python run_script.py --exp_yaml "${EXP}" --device_number 0 \
    ${ITEMS:+--items "${ITEMS}"}

rm -rf "${MIOPEN_USER_DB_PATH}"
echo "done"
