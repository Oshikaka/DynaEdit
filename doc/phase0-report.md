# DynaEdit 阶段 0 调研报告

- 论文：Versatile Editing of Video Content, Actions, and Dynamics without Training（DynaEdit），arXiv:2603.17989v1，43 页
- 日期：2026-09-19
- 状态：阶段 0 完成，地基与方案已定，等待进入阶段 1

---

## 0. 决策摘要（已拍板）

| # | 事项 | 决定 | 依据 |
|---|---|---|---|
| 1 | 代码形态 | **方案 A**：新建 `DynaEdit_utils.py` + 官方 diffusers，把 WAN repo 的编辑循环逻辑搬过来 | 见第 3 节 diff 证据：取速度场走的是干净公开接口 |
| 2 | 基座模型 | **`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers`**，论文同款，直接上 | T2V + n_max=N 复现不出 Fig.2 的基线现象 |
| 3 | 硬件 | **OLCF Frontier**，AMD MI250X / ROCm，账户 `csc662` | 已实地确认 |
| 4 | 评测 | **有 Gemini API**（学生认证后 Pro 免费层），阶段 4 做真 VLM 打分，按免费层配额设计脚本 | — |
| 5 | 仓库落位 | WAN repo 暂存 scratchpad，方案确认后再进项目目录 | — |
| 6 | 源视频 | 先用 WAN repo 自带的 7 条（832×480 / 49 帧 / 16fps） | 规格与论文一致 |

---

## 1. 图像版 FlowEdit 解剖：Algorithm 1 第 6–10 行的代码定位

核心循环在 `FlowEdit/FlowEdit_utils.py:171-227`（SD3 版），FLUX 版 `:322-399` 结构完全一致。

| Alg.1 | 代码位置 | 内容 |
|---|---|---|
| L4 `for i = n_max to 1` | `:171` + `:173-174` | `for i,t in enumerate(timesteps)`，用 `if T_steps - i > n_max: continue` 跳过前面的高 t 步 |
| L5 采噪声 | `:188` | `fwd_noise = torch.randn_like(x_src)`，**每步每次都新采，跨步 i.i.d.** ← ANC 要改这里 |
| L6 构造 z_src | `:190` | `zt_src = (1-t_i)*x_src + t_i*fwd_noise` |
| L7 构造 z_tar | `:192` | `zt_tar = zt_edit + zt_src - x_src` |
| L8 算 V_Δ | `:194-198` | 见下 |
| L9 平均 | `:186` + `:198` | `for k in range(n_avg)` 累加 `(1/n_avg)*(Vt_tar - Vt_src)` ← SGA 要替换这里 |
| L10 传播 ODE | `:201-205` | `zt_edit += (t_im1 - t_i) * V_delta_avg`，中途升 fp32 再降回 |

**速度场调用 / CFG（SD3）**：4 份 latent 拼成一个 batch `[z_src, z_src, z_tar, z_tar]`（`:194`），
prompt embeds 拼成 `[src_neg, src_pos, tar_neg, tar_pos]`（`:165`），**一次 transformer 前向**（`:65`）后
`chunk(4)`，分别用 γ_src、γ_tar 做 CFG（`:76-78`）。
FLUX 版没有真 CFG（蒸馏 guidance embedding），改成两次独立前向（`:349-365`）。

**n_max / n_min**：n_max 只决定"从第几步开始编辑"。n_min 是尾部切到 SDEdit 式常规采样（`:207-225`）——
**DynaEdit 的 Alg.2 里没有 n_min 分支，不移植**。

**latent 处理**：VAE 编码 + `(x - shift_factor) * scaling_factor`（`run_script.py:85-86`），
形状 `(1,16,H/8,W/8)`。FLUX 额外 `_pack_latents` 打包成 `(B,seq,64)`，结束时 unpack。

---

## 2. WAN2.1 repo（`KyujinHan/Awesome-Training-Free-WAN2.1-Editing`）判定

**它确实实现了视频版 FlowEdit，但不是 DynaEdit 需要的那一版。**

### 2.1 可复用的部分

- `diffusers/src/diffusers/pipelines/wan/pipeline_wan.py` 的 `WanPipeline.flowedit()`（L817–1195），循环 L1017–1160。
  与 Alg.1 的对应：L5→`:1057`，L6→`:1059`，L7→`:1060`，L8→`:1095`（4 路 CFG 拼 batch 一次前向，
  和图像版同构），L10→`:1121`。
- 视频 VAE 处理正确：`flowedit_prepare_latents`（L750–813）做 WAN 特有的 per-channel
  `latents_mean / latents_std` 归一化，latent 形状 `(1,16,F',60,104)`，49 帧 → `F'=13`。
- n_max 用 `strength` 表达：`total_steps = int(N*strength)`，`get_timesteps`（L695–702）切掉前面的步。
  **`strength=1.0` 等价于论文的 n_max = N**。
- 分辨率 832×480、16fps、49 帧，与论文输入规格一致，自带 7 条测试视频。

### 2.2 三条硬缺口

1. **是 T2V 不是 I2V。** `model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"`（`awesome_wan_editing.py:51`）。
   DynaEdit 用 WAN2.1 14B 480p **I2V**，且论文 App.C.6 明确说首帧条件对本方法 "crucial"——
   n_max=N 时 z_src 退化成纯噪声，首帧是**唯一**还锚住源视频的信息通道。
   repo 里的 `pipeline_wan_i2v.py` 是原版 diffusers 文件，**没有 flowedit 方法**。
2. **n_avg 写死为 1 且没有循环**（`:1015` `n_avg = 1`，`:1118` 直接 `V_delta_avg = (Vt_tar - Vt_src)`）。
   SGA 需要 j = 1..n_i^SGA 的多样本循环。
3. 没有 edited first frame（f_tar）这条输入通路。

---

## 3. 关键问题：取速度场 V 依赖 vendored diffusers 魔改吗？

把 vendored diffusers 与官方 `v0.34.0` 做了逐文件 diff（fork 自称 `0.34.0.dev0`，
全树的大量 diff 都是 dev 快照与 release 之间的上游漂移，需要逐文件甄别）。

### 3.1 与 WAN 相关的 4 个文件的实际改动

| 文件 | 相对 v0.34.0 的差异 | 性质 |
|---|---|---|
| `schedulers/scheduling_unipc_multistep.py` | 只有 copyright 年份 + arxiv URL 文案 | **纯上游漂移，零改动** |
| `models/autoencoders/autoencoder_kl_wan.py` | 缺一个上游后来加的 conv-count 缓存优化 | **纯上游漂移，零改动** |
| `pipelines/wan/pipeline_wan_i2v.py` | 只有 docstring 文案差异 | **纯上游漂移，零改动** |
| `models/transformers/transformer_wan.py` | **有真改动**，见下 | 魔改 |
| `pipelines/wan/pipeline_wan.py` | **有真改动**：新增 `flowedit()` / `flowalign()` 两个方法 | 新增方法 |

全树只有这 2 个文件带 `kyujin` 标记（`grep -rln "kyujin" diffusers/src`）。

### 3.2 `transformer_wan.py` 的魔改内容

```
-        attn_output = self.attn1(hidden_states=..., rotary_emb=rotary_emb)
+        attn_output, _ = self.attn1(hidden_states=..., rotary_emb=rotary_emb)
-        attn_output = self.attn2(hidden_states=..., encoder_hidden_states=...)
+        attn_output, attn_map_list = self.attn2(hidden_states=..., encoder_hidden_states=...)
-        return hidden_states
+        return hidden_states, attn_map_list
```

加上：`attn_processors` / `set_attn_processor`（从 `UNet2DConditionModel` 抄的）、
`self.attn_map_list` 状态、一句 `print("Transoformers block:", num_layers)`。

**后果**：`WanTransformerBlock.forward` 的返回值从张量变成二元组，attention processor 也必须返回二元组。
官方 `WanAttnProcessor2_0` 只返回单个张量，所以**这棵树必须配它自己的 processor 才能跑**——
这就是为什么 `awesome_wan_editing.py` 在 flowedit 分支里也要调
`register_attention_processor(pipe.transformer, "WanAttnProcessor2_0")`（`:207-209`），
装的是 `utils/wan_attention.py` 里那个返回二元组的同名替身。

### 3.3 结论

分两层看，答案不一样：

- **算法层：干净。** flowedit 循环取速度场的唯一方式是
  `self.transformer(hidden_states=..., timestep=..., encoder_hidden_states=..., attention_kwargs=..., return_dict=False)[0]`，
  全是 diffusers 公开签名。循环里**没有任何一行**依赖 attn map、attn processor 或 transformer 内部状态。
- **工程层：不干净。** 那棵树为了 FlowAlign 的 DIFS attention masking 改了 block 的返回签名，
  导致 vendored tree 与官方 diffusers 双向不兼容。

也就是说：**flowedit 循环本身不依赖魔改，但它所在的那棵树依赖魔改**，而那部分魔改
（attn map 提取）是给 FlowAlign 的 masking 用的，DynaEdit 一行都不需要。

这正好印证 DynaEdit 的 inversion-free / model-agnostic 定位：SGA 和 ANC 全在采样循环里，
不碰 attention、不改 transformer forward。

→ **走方案 A**：新建 `DynaEdit_utils.py` + 官方 diffusers，把 flowedit 循环的算法逻辑搬过来，
attn_map 那套一行不带。`WanImageToVideoPipeline` 的组件（transformer / VAE / T5 / CLIP / scheduler）
全部是公开 API，够用。

---

## 4. 硬件现实（Frontier）

实地确认：`login08`，SLES 15-SP7，Slurm，账户 `csc662`，`LMOD_SYSTEM_NAME=frontier`。

- GPU：**AMD MI250X / gfx90a，Slurm 每节点暴露 8 张卡，每张 64 GB**（不是 A100 80G）。
  ROCm 栈（`module load rocm/6.x`）。现有 `orbit` 环境是 `torch 2.8.0+rocm6.4`，
  没装 diffusers/transformers，需要按 `rebuild-orbit.sh` 的套路新建 `dynaedit` 环境。
- 计算节点大概率无外网 → 权重必须在登录节点预下载，作业里 `HF_HUB_OFFLINE=1` 离线加载。
- `Wan2.1-I2V-14B-480P-Diffusers` 下载量 **≈95 GB**
  （transformer 14 分片 fp32 ≈68 GB + T5 5 分片 ≈24 GB + CLIP 1.3 GB + VAE 0.5 GB），
  **必须放 Lustre**（`$MEMBERWORK/csc662`），$HOME 只有 50 GB。
- 模型配置（已从 HF 拉取确认）：`in_channels=36`（16 latent + 4 mask + 16 cond）、`out_channels=16`、
  `num_layers=40`、`attention_head_dim=128`、`num_attention_heads=40`、`image_dim=1280`、
  scheduler `UniPCMultistepScheduler(prediction_type="flow_prediction", use_flow_sigmas=True, flow_shift=3.0)`。
- ⚠️ **粗估，未实测**：14B I2V、81 帧、50 步、4 路 CFG batch，单 GCD bf16
  ≈ **1.5–2 小时/条视频**；49 帧约 1 小时。
  显存 ≈ 32 GB 权重 + 15–20 GB 激活，64 GB 放得下但要 T5/VAE 用完即 offload，
  batch 4 可能要拆成 2×2。SGA 前 3 步各多 4 次采样，总开销只 +5~10%。
- 一节点 8 卡 → 阶段 4 的 4 组超参 × 7 条视频可以直接并行铺开。

---

## 5. 改造方案

```
DynaEdit/
├── doc/phase0-report.md       # 本文件
├── FlowEdit/                  # 已有，图像版，只作算法对照，不动
├── DynaEdit_utils.py          # 【新】核心算法，镜像 FlowEdit_utils.py
├── run_script.py              # 【新】入口，镜像图像版
├── configs/
│   ├── wan_i2v_exp.yaml       # 超参：T_steps/n_max/γ_src/γ_tar/use_sga/use_anc/τ/n_sga/...
│   └── edits.yaml             # 编辑内容：video_path/src_prompt/tar_prompt/first_frame_tar
├── slurm/run_dynaedit.sbatch  # 【新】Frontier 作业脚本
├── env/rebuild-dynaedit.sh    # 【新】环境重建脚本（仿 rebuild-orbit.sh）
├── eval/                      # 【阶段 4】Gemini VLM 打分
├── videos/                    # 源视频（先借 WAN repo 那 7 条）
└── outputs/
```

### 5.1 `DynaEdit_utils.py` 计划的函数（全部新增，互不侵入）

| 函数 | 职责 | 来源 |
|---|---|---|
| `prepare_video_latents(pipe, video, H, W)` | 视频 → `x_src` latent，含 WAN `latents_mean/std` 归一化 | 抄 `flowedit_prepare_latents` L750-813 |
| `prepare_i2v_condition(pipe, first_frame, num_frames, H, W)` | 首帧 → `(mask, cond_latent)` 4+16 通道 + CLIP `image_embeds`；**src / tar 各调一次** | 抄 `pipeline_wan_i2v.prepare_latents` L371-448 + `encode_image` L225 |
| `calc_v_wan_i2v(pipe, z_src, z_tar, cond_src, cond_tar, embeds, img_embeds, γ_src, γ_tar, t)` | 拼 36 通道输入、4 路 CFG 一次前向、`chunk(4)`、返回 `(V_src, V_tar)` | 对应图像版 `calc_v_sd3` |
| `sga_aggregate(z_edit, V_list, x_src, t_i, tau)` | **阶段 2**：Eq.4 投影 → Eq.5 cosine+softmax → Eq.6 回速度 | 新逻辑 |
| `anc_update(w_tilde, a_t, w_new)` | **阶段 3**：Eq.7 相关噪声更新 | 新逻辑 |
| `anc_schedule(t, t_full=0.25)` | a_t 线性 0→1，t=1 时 0，t≤0.25 时 1 | 新逻辑 |
| `sga_schedule(i, N, n_hi=5, n_lo=1, k=3)` | i > N−3 返回 5，否则 1 | 新逻辑 |
| `DynaEditWanI2V(pipe, ..., use_sga=False, use_anc=False, ...)` | 主循环，Alg.2 | 骨架 = `flowedit` L1017-1160 |

### 5.2 主循环与开关插入点

默认关时逐行等价于原 FlowEdit：

```python
for i, t in enumerate(timesteps):            # n_max = N → strength=1.0，不跳步
    n_i = sga_schedule(i, N) if use_sga else n_avg     # 关时 = n_avg（默认 1）
    V_list = []
    for j in range(n_i):
        if use_anc:
            w[j] = anc_update(w[j], a_t=anc_schedule(t_i), w_new=randn_like(x_src))  # Eq.7
        else:
            w[j] = torch.randn_like(x_src)                   # 原行为：跨步 i.i.d.
        z_src = (1 - t_i) * x_src + t_i * w[j]               # Alg.2 L7
        z_tar = z_edit + z_src - x_src                       # Alg.2 L8
        V_list.append(calc_v_wan_i2v(...))                   # Alg.2 L9
    V_bar = sga_aggregate(...) if use_sga else sum(V_list) / n_i   # Alg.2 L10 / Alg.1 L9
    z_edit = z_edit + (t_im1 - t_i) * V_bar                  # Alg.2 L11
```

`use_sga=False, use_anc=False, n_avg=1, strength=1.0` ⇒ 与 WAN repo 的 `flowedit` 数值完全一致。

### 5.3 各阶段最小验证

- **阶段 1**：latent shape 断言 `(1,16,13,60,104)`；开关全关 vs repo 原 `flowedit`，
  固定种子逐元素 `allclose`。
- **阶段 2**：`sga_aggregate` 单元测试——n=1 时必须恒等于 `V_list[0]`（softmax 单元素 = 1）；
  τ→0 时权重退化成 one-hot；权重和为 1。再跑 on/off 对比视频。
- **阶段 3**：`anc_update` 检查 `w̃` 边缘分布仍是 N(0,I)（均值≈0、方差≈1）；
  `a_t=0` 时 = 纯新噪声，`a_t=1` 时 = 完全复用上一步。
  再画论文 Fig.11 那种"相邻步 V_Δ 余弦相似度 vs t"的曲线，i.i.d. 与 ANC 两条线对比。

---

## 6. 论文明确写的 vs 猜测

### 6.1 论文明确（Sec.5.1 / 5.2 / 6.1、App.C / D.1 / F）

- Eq.4 `z_j^{edit-proj} = z^edit − t_i · V_j^Δ`
- Eq.5 `s_j = softmax(cosim(x_src, z_j^{edit-proj}) / τ)`，`z̄ = Σ_j s_j z_j^{edit-proj}`
- Eq.6 `V̄^Δ = (z^edit − z̄) / t_i`
- SGA schedule：`n_i^SGA = 5` 仅前 3 步（i > N−3），其余为 1
- Eq.7 `w̃_j ← √a_{t_i} · w̃_j + √(1−a_{t_i}) · w_j`；a_t 线性递增，
  t=1 时 0，**t=0.25 时到 1，之后保持 1**；Markovian 递增
  （App.C.3 消融了非 Markovian 与递减两种，都更差：前者 ghosting，后者 jitter）
- 初始化 `w̃_j = 0`（Alg.2 L3），配合 `a_{t_N}=0` ⇒ 第一步为纯新噪声
- 相似度用 **cosine**（App.C.4 对比过 −MSE，cosine 更好）
- 基座 WAN2.1 14B 480p I2V，832×480，16fps，49–81 帧，**n_max = N**
- 4 组超参：(γ_src, γ_tar) ∈ {(2.5, 4.5), (4.5, 8.5)} × τ ∈ {1.0, 0.001}
- App.F 有完整 VLM 打分 prompt（源保真 / 文本符合 / 视觉质量，各 1–5 分，Gemini 3 Pro），可照搬
- 需要 **edited first frame f_tar**（论文用 Gemini 2.5 Flash Image / Nano-Banana Pro 生成；
  不需要改首帧的编辑就直接用源视频首帧）
- 评测集 71 条，4 类：交互式插入 / 物体替换 / 动作改变 / 全局效果（未公开）

### 6.2 ⚠️ 推断或猜测（实现时全部做成可配参数）

1. ⚠️ **τ 的值**：正文写 0.01，附录 D.1 写 0.001，差 10 倍。倾向以附录为准（0.001），两个都进 yaml。
2. ⚠️ **总步数 N 没写**。WAN2.1 480p I2V 的 diffusers 默认 50，WAN 官方 I2V 推荐 40。
   倾向 50（与 SD3 版 FlowEdit 一致）。
3. ⚠️ **cosine 在哪个维度算**：整个 latent 展平成一个向量，还是 per-frame / per-channel 后平均？
   论文只说 "cosine similarity between each prediction and the source video"。倾向整体展平，做成可配。
4. ⚠️ **cosine 在 latent 空间还是像素空间**：`x_src` 在 Alg.2 L7 被用来构造 z_src，所以应是 latent 空间。
   把握较大，但仍属推断。
5. ⚠️ **n_i^SGA 从 5 掉到 1 时，5 个 `w̃_j` 保留哪一个**给后续 ANC 递推？论文没说。
   候选：保留 j=0 / 保留 SGA 权重最大者 / 按权重加权合并。倾向"保留权重最大者"，做成可配。
6. ⚠️ **a_t 的"线性"是对 t 线性还是对步数 index 线性**：两个锚点（t=1→0、t=0.25→1）明确，
   按对 t 线性插值 `a_t = clip((1 − t) / 0.75, 0, 1)`。
7. ⚠️ **n_min 是否为 0**：Alg.2 无 n_min 分支，按 0 处理（不做 SDEdit 尾段）。
8. ⚠️ **Eq.7 与 Alg.2 L6 的时间下标轻微不一致**（Eq.7 说"在 t_i 用 a_{t_i} 更新出 t_{i−1} 的噪声"，
   Alg.2 说"在步 i 开头用 a_{t_i} 更新"）。按 Alg.2 实现。
9. ⚠️ 第 4 节所有性能 / 显存数字均为估算，未实测。

---

## 7. 阶段 1 待办（下一步）

1. 建 `dynaedit` conda 环境（ROCm 6.x + torch 2.8.0+rocm6.4 + 官方 diffusers + transformers）
2. 登录节点预下载 `Wan2.1-I2V-14B-480P-Diffusers` 到 Lustre（≈95 GB）
3. WAN repo 落位到项目目录（只作参考，不作依赖）
4. 写 `DynaEdit_utils.py` 的 I2V 版 FlowEdit（开关全关）+ `run_script.py` + yaml
5. 跑通一条：源视频 + 源 prompt + 目标 prompt → 输出
6. 用 `n_max = N`（`strength=1.0`）复现论文 Fig.2 现象：能编辑，但有低频错位 + 高频抖动


