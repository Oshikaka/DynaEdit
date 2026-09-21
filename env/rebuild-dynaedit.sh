#!/bin/bash
# 重建 dynaedit 环境 —— AMD / ROCm 平台（OLCF Frontier, MI250X / gfx90a）
#
# 用法: bash env/rebuild-dynaedit.sh [环境名]      默认 dynaedit
#
# 为什么是重建脚本而不是 conda env export 的 yml：
#   yml 不记录 --index-url，`torch==2.8.0+rocm6.4` 这类轮子 PyPI 上根本没有，还原不回来。
#   详见 ~/STORAGE.md。
#
# NVIDIA 用户不要跑这个，跑 env/setup-cuda.sh。
set -euo pipefail

ENV_NAME="${1:-dynaedit}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

module load miniforge3/23.11.0-0
module load PrgEnv-gnu rocm/6.4.0 craype-accel-amd-gfx90a
source "$(conda info --base)/etc/profile.d/conda.sh"
export PYTHONNOUSERSITE=1      # 防止 ~/.local 里的包串进来

# 环境建在 Lustre（由 ~/.condarc 的 envs_dirs 指定），不占 $HOME 的 50 GB 配额
conda create -n "$ENV_NAME" python=3.11 -y
conda activate "$ENV_NAME"

# torch 必须从 ROCm 专用 index 装；PyPI 上没有 +rocm6.4 的轮子。
# 2.8.0+rocm6.4 是本机 orbit 环境已验证可用的组合。
pip install torch==2.8.0+rocm6.4 torchvision==0.23.0+rocm6.4 \
    --index-url https://download.pytorch.org/whl/rocm6.4

# 其余依赖与平台无关
pip install -r "${REPO_ROOT}/requirements.txt"

echo
echo "完成。验证（需在有 GPU 的节点上跑，登录节点没有 GPU）："
echo "  conda activate ${ENV_NAME}"
echo "  python -c \"import torch, diffusers, transformers; \\"
echo "             print(torch.__version__, torch.version.hip, torch.cuda.is_available(), \\"
echo "                   diffusers.__version__, transformers.__version__)\""
