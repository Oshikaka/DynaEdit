# 阶段 1 结果：视频版 FlowEdit 基线（n_max = N）

配置：`configs/wan_i2v_exp.yaml`，`Wan2.1-I2V-14B-480P`，50 步，`n_max = N`，
CFG (γ_src, γ_tar) = (2.5, 4.5)，`use_sga=false, use_anc=false`，seed 42，49 帧 / 832×480 / 16fps。
f_tar = 源视频首帧（本阶段不涉及改首帧的编辑）。

## 性能实测

| 项 | 数值 |
|---|---|
| 单步耗时 | **45.0 s/步**（micro_batch=2，49 帧 → 13 latent 帧 → 20280 token） |
| 单条总耗时 | **41–42 分钟**（含约 2 分钟加载 14B + T5 + CLIP + VAE） |
| 峰值显存 | **36.2 GiB / 64 GiB**（编辑循环内） |

显存余量很大，后续可以提 `micro_batch` 到 4，或上到 81 帧（论文规格上限）。
⚠️ 这修正了阶段 0 的估计（当时估 1–2 小时 / 条，偏保守）。

## 三条结果

| 编辑 | 空间高频保留 | 低频运动相关 | 现象 |
|---|---|---|---|
| `boat_eagle_dive`（交互式插入） | **0.16x** | **+0.07** | 鹰成功插入并俯冲，但**整个源场景塌掉**——船消失、海岸线化为均匀雾霭 |
| `swan_take_off`（动作改变） | **0.66x** | **+0.45** | 天鹅确实展翅起飞、水花飞溅，但**糊成一团**，轨迹与源对不上 |
| `bear_rear_up`（动作改变） | 0.74x | +0.94 | **编辑没发生**，熊照常行走；对齐很好但 prompt 没被执行 |

图：`outputs/analysis/<item>/*_grid.png`（抽帧拼图）与 `*_slice.png`（论文 Fig.3 式时空切片）。

## 判定：论文 Fig.2 的现象复现了

`swan_take_off` 是最教科书的一条，三个判据齐备：

- **有编辑力**：target prompt 的动作被执行（展翅、起飞、水花）
- **高频退化**：空间高频能量只剩 0.66x，明显涂抹模糊
- **低频错位**：相关系数 0.45；时空切片里源的颈部轨迹是清晰波形，编辑后变成对不上的弥散暗斑

对应论文第 4 节："a video that adheres to the edit prompt, but exhibits extraneous low
frequency changes … and suffers from severe high frequency jitter artifacts
(evident by the blurry obstacle)"。

更值得注意的是**三条之间的巨大差异本身**：同一组 `n_max = N` 与同一组 CFG，
`boat` 编辑过头到场景崩溃，`bear` 则完全没编辑动。这正是论文的核心论点——
*"it is practically impossible to select a value for n_max that strikes a good balance
between output quality, prompt adherence, and loyalty to the source video"*。
SGA / ANC 要解决的就是这个。

**结论：基线正确，可以进入阶段 2。**

## 一处方法更正

最初的抖动指标（时间二阶差分 / 一阶差分）**测错了对象**，对 swan 报 0.99x，
差点让我误判"抖动没复现"。原因：这类退化表现为**模糊**而非逐帧乱跳，
而模糊会同时压低一阶和二阶差分（swan 源 d1=0.063 → 编辑 0.044），比值几乎不动。

已补上**空间高频能量**（梯度幅值均值）指标，它才对这种退化敏感：0.66x / 0.16x 一目了然。
时间粗糙度指标保留但降级为辅助，并在输出里注明它对模糊不敏感。
⚠️ 这一点对阶段 3 判断 ANC 是否生效很关键——不要只看时间维指标。

## 对后续阶段的影响

1. `bear_rear_up` 属于论文 App.D.1 说的"大动作改变"，应改用超参组 (3)：γ=(4.5, 8.5)。
   目前用的 (2.5, 4.5) 是给"小而精确的改动"的。阶段 2 起把它换到高 CFG 组。
2. `boat_eagle_dive` 的崩溃程度（corr 0.07）给了 SGA 一个很宽的改进空间，
   是观察 SGA 低频对齐效果的好样本。
3. 三条一起构成阶段 2/3 的对照集：一条编辑过头、一条居中、一条编辑不足。
