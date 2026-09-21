# DynaEdit（复现）

Training-free 视频编辑：在预训练 flow 模型上做 inversion-free 编辑，额外引入两个模块

- **SGA**（Similarity Guided Aggregation，论文 5.1）—— 修低频时空错位
- **ANC**（Annealed Noise Correlation，论文 5.2）—— 修时间维高频抖动

论文：*Versatile Editing of Video Content, Actions, and Dynamics without Training*，arXiv:2603.17989。
基座模型：`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`。
算法地基：[FlowEdit](https://github.com/fallenshock/FlowEdit)（图像版，本仓 `FlowEdit/` 下留作对照）
与 [Awesome-Training-Free-WAN2.1-Editing](https://github.com/KyujinHan/Awesome-Training-Free-WAN2.1-Editing)（WAN 视频版 FlowEdit，作参考不作依赖）。

> **设计原则**：SGA / ANC 全部活在采样循环里，不碰 attention、不改 transformer forward。
> 本仓只依赖官方 diffusers 的公开接口，不需要任何 patch 过的 diffusers。
> 调研依据见 [`doc/phase0-report.md`](doc/phase0-report.md) 第 3 节。

## 当前状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 地基调研与方案 | ✅ 见 `doc/phase0-report.md` |
| 1 | 视频版 FlowEdit 基线跑通 | ✅ 已复现论文 Fig.2 现象。见 `doc/phase1-notes.md`（实现与 review）、`doc/phase1-results.md`（结果） |
| 2 | SGA（`use_sga`） | ✅ 低频对齐提升 2–2.4 倍。见 `doc/phase2-results.md` |
| 3 | ANC（`use_anc`） | ✅ 实现闭式验证通过、Fig.11 复现；端到端增量待 VLM 判定。见 `doc/phase3-results.md` |
| 4 | 超参扫描 + VLM 评测 | 🚧 18 次运行进行中；评测脚本就绪，待 API key |

## 安装

分两步：**先按平台装 torch，再装平台无关的依赖**。第二步两个平台完全一样。

### AMD / ROCm（OLCF Frontier, MI250X）

```bash
bash env/rebuild-dynaedit.sh            # 建名为 dynaedit 的环境
bash env/rebuild-dynaedit.sh myenv      # 或换个名字
```

脚本会 `module load` ROCm 工具链、从 ROCm 专用 index 装 `torch==2.8.0+rocm6.4`，
再 `pip install -r requirements.txt`。

> Frontier 上环境默认建在 Lustre（由 `~/.condarc` 的 `envs_dirs` 决定），不占 `$HOME` 配额。

### NVIDIA / CUDA

```bash
bash env/setup-cuda.sh                  # 默认 cu126
bash env/setup-cuda.sh dynaedit cu128   # 新卡 / 新驱动
```

等价于手动两步：

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

> `torch 2.8.0` 只有 `cu126` / `cu128` 的轮子，没有 `cu121` / `cu124`。
> 不用 conda 的话跳过脚本里的 `conda create`，自己激活一个 Python 3.11 的 venv 即可。

### 为什么这么拆

`requirements.txt` 里刻意**不放 torch**。ROCm / CUDA 的轮子只存在于各自的 PyTorch index，
PyPI 上没有，写进 requirements 会让另一个平台直接装错或装不上。
拆开之后，换平台只需要换 torch 那一行，其余依赖和代码原样不动。

## 模型权重

约 95 GB。**计算节点通常没有外网，必须在登录节点预下载**：

```bash
export HF_HOME=/lustre/orion/csc662/scratch/xinru/hf     # 放 Lustre，别放 $HOME
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P-Diffusers
```

作业脚本里设 `HF_HUB_OFFLINE=1` 离线加载。

## 目录结构

```
DynaEdit/
├── doc/phase0-report.md       # 阶段 0 调研报告（算法定位、地基 diff 证据、超参清单）
├── FlowEdit/                  # 官方图像版 FlowEdit，只作算法对照，不参与运行
├── env/
│   ├── rebuild-dynaedit.sh    # AMD / ROCm 安装入口
│   └── setup-cuda.sh          # NVIDIA / CUDA 安装入口
├── requirements.txt           # 平台无关依赖（不含 torch）
├── DynaEdit_utils.py          # 核心算法（阶段 1 起）
├── run_script.py              # 入口
├── configs/                   # 超参 yaml + 编辑内容 yaml
├── slurm/                     # Frontier 作业脚本
└── outputs/
```

## 运行

```bash
# 单条编辑
sbatch slurm/run_dynaedit.sbatch configs/wan_i2v_exp.yaml swan_take_off

# 一批（单节点 8 个 GCD 并行，joblist 每行 "<yaml> <item> <exp_name>"）
sbatch slurm/run_sweep.sbatch configs/p4a.joblist
```

非 Slurm 环境直接跑：

```bash
python run_script.py --exp_yaml configs/phase4.yaml --exps p4_g2 --items swan_take_off
```

## 配置

配置分两层：**超参**（`configs/<实验>.yaml`）与**编辑内容**（`configs/edits.yaml`），
沿用图像版 FlowEdit 的风格。超参 yaml 支持两种写法：

```yaml
# 写法一：纯列表（图像版风格）
- exp_name: my_exp
  T_steps: 40
  ...

# 写法二：defaults + experiments（扫参时避免大段重复）
defaults: {T_steps: 40, use_sga: true, ...}
experiments:
  - {exp_name: g1, tau: 1.0}
  - {exp_name: g2, tau: 0.001}
```

`edits.yaml` 里可用 `overrides` 给单条编辑指定不同超参（论文 App.D.2 就是按例子分配超参组的）；
扫参时用 `ignore_overrides: true` 关掉它，否则被覆盖的那条不会参与扫描。

### 开关

新增能力全部用开关控制，**默认全关时行为等同原始 FlowEdit**（有端到端逐比特测试保证，
见 `tests/test_loop.py`）。

| 开关 | 默认 | 含义 |
|---|---|---|
| `use_sga` | `false` | 相似度加权聚合替代多噪声取平均（论文 Eq.4–6） |
| `use_anc` | `false` | 跨步递增相关噪声替代 i.i.d. 噪声（论文 Eq.7） |

两个都开 = 完整 DynaEdit（论文 Algorithm 2）。

### 主要超参

| 键 | 默认 | 含义 |
|---|---|---|
| `T_steps` | 40 | 采样步数。论文未写，取 WAN 官方 I2V 推荐值 |
| `n_max` | 空（=N） | 从第几步开始编辑。论文用 n_max = N |
| `tau` | 0.001 | SGA 的 softmax 温度。**越小越贴近源** |
| `sga_n_hi` / `sga_k` | 5 / 3 | 前 k 步用 n_hi 个噪声，之后用 1（论文 6.1） |
| `anc_t_full` | 0.25 | a_t 线性升到 1 的时刻（论文 6.1） |
| `anc_carry` | `argmax` | n_SGA 收缩时保留哪条噪声链。⚠️ 论文未说明 |
| `negative_prompt_preset` | `wan_zh` | 见 `configs/negative_prompts.yaml`。**实测中文原版明显优于英译版** |
| `micro_batch` | 2 | 4 路 CFG 拆成几次前向。数值上惰性，纯显存/时间权衡 |
| `global_rng` | `false` | `false` = 按 (step, branch) 定种子，使开关 on/off 的噪声流对齐 |

**关于 `tau`**：实测 cosine 相似度的跨度约 0.1–0.8，因此 `tau <= 0.01` 等价于硬选择（argmax）、
`tau = 1.0` 等价于均匀平均（有效分支数 4.74–5.00）。软硬过渡带在 **tau ≈ 0.03–0.3**。
论文的两档 {1.0, 0.001} 实际是这两个极端。详见 `doc/phase2-results.md`。

## 评测

```bash
pip install -r requirements-eval.txt
export GEMINI_API_KEY=...
python eval/vlm_eval.py --list-models     # 先确认 key 能用哪个模型
python eval/vlm_eval.py                   # 三维度各 1-5 分，prompt 照搬论文附录 F
python eval/vlm_eval.py --report          # 只读缓存出表，不调 API
```

结果落盘缓存，中断可续跑；429/5xx 指数退避；按生效配置去重，避免重复运行被当成独立样本。
