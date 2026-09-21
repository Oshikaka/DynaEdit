# 阶段 1 笔记：环境、基线实现、代码 review

## 1. 环境与权重

- 安装拆成「平台无关依赖 + 平台相关 torch」两步：`requirements.txt`（不含 torch）
  \+ `env/rebuild-dynaedit.sh`（ROCm）/ `env/setup-cuda.sh`（CUDA）。README 有说明。
- 实测可用：`torch 2.8.0+rocm6.4 / diffusers 0.34.0 / transformers 4.51.3 / python 3.11`。
  ⚠️ `torch 2.8.0` 只有 `cu126` / `cu128` 轮子，没有 `cu121` / `cu124`。
  ⚠️ `huggingface_hub` 必须 `<1.0`，transformers 4.51 不兼容 hub 1.x。
- 权重：`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`，实际 **82 GB**（阶段 0 估的 95 GB 偏高）。
  放在 `.bashrc` 已配好的 `HF_HOME=/lustre/orion/csc662/proj-shared/xinru/hf-cache`。
  计算节点无外网，作业里 `HF_HUB_OFFLINE=1`。

## 2. 实现中发现的两个技术点

### 2.1 SGA 的 Eq.4-6 代数上恒等于速度加权平均

因为 softmax 权重和为 1：

```
V̄ = (z_edit − Σ_j s_j (z_edit − t_i V_j)) / t_i
  = (z_edit − z_edit·Σs_j + t_i Σ s_j V_j) / t_i
  = Σ_j s_j V_j
```

投影 `z^{edit-proj}` 只用来【算权重】，聚合本身就是速度的 softmax-cosine 加权平均。

默认走等价形式。字面形式要除以 `t_i`，而末段 `t_i ≈ 0.0576`，浮点误差被放大：
实测相对误差 `t_i=0.5 → 4.0e-7`，`0.0576 → 3.3e-6`，`0.01 → 3.3e-5`，`0.001 → 2.4e-4`。
`sga_literal: true` 可切回字面式，测试里两者在实际会用到的 `t_i` 上一致。

### 2.2 插值系数改用 `scheduler.sigmas` 而非 `t/1000`

`timesteps` 是整数（`[999, 992, ...]`），真 sigma 是 `[0.999666, 0.992907, ...]`，
最大差 9.9e-4，末段相对误差约 1.7%。图像版 SD3 路径和 WAN repo 都用 `t/1000`，
但图像版 FLUX 路径用的是精确 `scheduler.sigmas`（`FlowEdit_utils.py:328`）。这里取后者。

**后果**：结果不会与 WAN 视频版 FlowEdit 逐比特相同。属于有意偏离。

## 3. 代码 review 结论与修复

由独立 agent 对照论文、两个参考实现、以及实际安装的 diffusers 0.34.0 做的审查。

### 已修

| # | 严重度 | 问题 | 修法 |
|---|---|---|---|
| 1 | **BLOCKER** | ANC 噪声链每步被摧毁重建：`_anc_carry` 每步无条件把 `{w̃_j}` 截成 1 条，下一步再补零重建。补零链经 Eq.7 得到 std=√(1−a_t) 的噪声；`a_t=1` 时（T=50 的第 45 步起）直接退化成**全零张量**，使 `zt_src=(1−t_i)x_src` 成为模型没见过的无噪 latent | 只在下一执行步确实需要更少链时才收缩；新链用 `randn` 起步而非 0（`a_t=0` 时与补 0 完全等价，`a_t>0` 时保方差） |
| 2 | MAJOR | `_anc_carry("weighted")` 不保方差，实测 std=0.675 | 除以 `√Σw²` |
| 3 | MAJOR | 无帧数校验。非 4k+1 会在 fp32 VAE 编码跑完之后才在 diffusers 内部炸 reshape；`num_frames` 大于视频长度会静默截断，到 transformer 才炸 | 在任何编码之前 raise |
| 4 | MINOR | `model_id` / `flow_shift` 只从 `exp_configs[0]` 读，多组实验扫参时静默用第一组的值（`flow_shift` 会改变每个 sigma 和 a_t） | `flow_shift` 按 exp 重建 scheduler；`model_id` 不一致直接 raise |
| 5 | MINOR | `n_max < T_steps` 时 ANC 的 `w̃=0` 初始化失效（首个执行步 `a_t` 已 >0） | 同 #1 |
| 6 | MINOR | `test_offswitch_equivalence` 是空测试，断言的是 torch 的性质，主循环再错也会过 | 改为 `tests/test_loop.py::test_regression`：stub transformer + 真 UniPC scheduler，与从零独立写的 Algorithm 1 参考逐比特比对 |
| 7 | NIT | 字面式 vs 等价式只在友好的 `t_i=0.4` 上测 | 扫到实际最小 `t_i=0.0576` |
| 8 | NIT | 形状校验用 `assert`，`python -O` 下会被剥掉 | 改 `raise` |
| — | — | 无 `f_tar` 时第二次条件编码是逐字节重复劳动，而那正是显存峰值 | 复用 `cond_src`；另加 `vae_tiling` 开关 |
| — | — | SGA on/off 会改变每步的采样**次数**，全局 RNG 一旦错位，此后每步噪声都不同，A/B 被"噪声也变了"污染 | 加 `draw_noise(seed, step, branch)`，分支 0 的噪声与 `n_i` 取 1 还是 5 无关；`global_rng: true` 可退回原行为 |

### 已验证正确（review 交叉确认）

- **开关全关 == FlowEdit，逐比特**：`max|diff| = 0.000e+00`，覆盖
  `(T, n_avg, n_max, micro_batch)` 六组组合。`n_avg=3` 差 4.8e-7，是求和顺序差异。
- CFG 拼 batch 顺序 `[src_neg, src_pos, tar_neg, tar_pos]`、`chunk(4)` 拆包、两条 guidance
  公式，与 `calc_v_sd3`（`FlowEdit_utils.py:73-76`）完全一致。
- latent 归一化往返与 WAN repo 一致；`.mode()` 等价于 `retrieve_latents(sample_mode="argmax")`。
- `micro_batch` 数值上惰性（RoPE / 条件嵌入 / attention 都与 batch 无关），mb=1/2/4 逐比特相同。
- 用到的每个 diffusers 0.34.0 API 签名都对；`max_sequence_length=512` 是**必须**显式传的
  （`encode_prompt` 默认 226，pipeline `__call__` 默认 512）。
- ANC 的时间下标取 Alg.2 的读法（而非 Eq.7 正文的读法）是唯一与 `w̃=0` + `a_{t_N}=0` 自洽的。
  实测 `a_t` 在 T=50 的第 45 步（σ=0.213）恰好到 1，与「t=0.25 后保持 1」吻合。
- 分阶段 offload 的设备放置正确，`pipe._execution_device` 实际从未被触发，无跨条目显存泄漏。

## 4. 待定项（需要拍板）

1. **negative prompt 留空还是用 WAN 官方那条长负面 prompt。** 目前留空（跟随图像版 FlowEdit），
   但 `γ_tar=4.5` 下 uncond 分支有实际权重，WAN 自己的推荐配方是带长负面 prompt 的。
2. **τ = 0.001 还是 0.01。** 论文正文 Sec 6.1 写 `{0.01, 1}`，附录 D.1 写 `{1.0, 0.001}`，
   自相矛盾。代码取附录值。要对 Table 4 的话两个都得跑。
3. **总步数 N 论文未写**，取 50（diffusers 对 WAN 480p I2V 的默认；WAN 官方 I2V 推荐 40）。

## 5. 状态

- 冒烟作业（4 步 / 17 帧）已投队列，验证管线而非画质。
- 阶段 1 的判据工具：`tools/analyze.py`，输出论文 Fig.3 式时空切片 + 抽帧拼图 +
  两个启发式指标（高频抖动比值、低频运动相关系数）。自检通过（源对自己：1.00x / +1.0000）。
  ⚠️ 这两个指标不是论文评测，只用来回答「现象有没有复现」。
