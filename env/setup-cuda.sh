#!/bin/bash
# 安装 dynaedit 环境 —— NVIDIA / CUDA 平台
#
# 用法: bash env/setup-cuda.sh [环境名] [cuda 标签]
#   例: bash env/setup-cuda.sh dynaedit cu126     # 默认
#       bash env/setup-cuda.sh dynaedit cu128     # 新卡 / 新驱动
#
# 与 ROCm 版的唯一区别就是 torch 这一行；requirements.txt 完全共用。
# 注意：torch 2.8.0 只有 cu126 / cu128 的轮子，没有 cu121 / cu124。
set -euo pipefail

ENV_NAME="${1:-dynaedit}"
CUDA_TAG="${2:-cu126}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 如果你不用 conda，跳过下面两行，自己激活 venv 即可（Python 3.11）
conda create -n "$ENV_NAME" python=3.11 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate "$ENV_NAME"

pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

pip install -r "${REPO_ROOT}/requirements.txt"

echo
echo "完成。验证："
echo "  python -c \"import torch, diffusers, transformers; \\"
echo "             print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), \\"
echo "                   diffusers.__version__, transformers.__version__)\""
