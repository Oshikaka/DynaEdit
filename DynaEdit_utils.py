"""DynaEdit —— WAN2.1 I2V 上的 inversion-free 视频编辑。

结构镜像官方图像版 FlowEdit 的 ``FlowEdit/FlowEdit_utils.py``：
纯函数 + 一个主循环，不继承 / 不 patch diffusers 的任何类。
SGA 与 ANC 全部活在采样循环里，不碰 attention、不改 transformer forward。

论文: Versatile Editing of Video Content, Actions, and Dynamics without Training
      (DynaEdit), arXiv:2603.17989
"""

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm



# --------------------------------------------------------------------------- #
# WAN latent 归一化                                                            #
# --------------------------------------------------------------------------- #
def _wan_latent_stats(vae, device, dtype):
    """WAN VAE 用 per-channel 的 mean / std 归一化 latent（不是 SD 那种标量 scaling_factor）。"""
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    return mean, inv_std


@torch.no_grad()
def encode_video_to_latent(
    pipe, video: List[Image.Image], height: int, width: int, device=None
) -> torch.Tensor:
    """源视频 -> x_src，形状 (1, 16, F', H/8, W/8)，F' = (F-1)//4 + 1。"""
    device = device or pipe._execution_device
    pixels = pipe.video_processor.preprocess_video(video, height=height, width=width)
    pixels = pixels.to(device=device, dtype=pipe.vae.dtype)

    latent = pipe.vae.encode(pixels).latent_dist.mode()
    mean, inv_std = _wan_latent_stats(pipe.vae, device, latent.dtype)
    return ((latent - mean) * inv_std).to(torch.float32)


@torch.no_grad()
def decode_latent_to_video(pipe, latent: torch.Tensor, output_type: str = "np", device=None):
    """x_edit latent -> 视频帧。encode_video_to_latent 的逆。"""
    latent = latent.to(device=device or latent.device, dtype=pipe.vae.dtype)
    mean, inv_std = _wan_latent_stats(pipe.vae, latent.device, latent.dtype)
    pixels = pipe.vae.decode(latent / inv_std + mean, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(pixels, output_type=output_type)


# --------------------------------------------------------------------------- #
# I2V 条件与 prompt                                                            #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def prepare_i2v_condition(
    pipe, first_frame: Image.Image, num_frames: int, height: int, width: int,
    device=None, dtype=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """首帧 -> (condition, image_embeds)。

    condition 是 (1, 20, F', h, w)：4 通道时间 mask + 16 通道首帧 latent，
    每步与 16 通道的 z_t 拼成 transformer 要的 36 通道输入。
    image_embeds 是 CLIP 视觉特征，走 ``encoder_hidden_states_image``。

    源侧传 f_src、目标侧传 f_tar，各调一次。
    """
    device = device or pipe._execution_device
    # dtype 默认跟 transformer 走，但显式可传：这样只装 VAE+CLIP 的场合
    # （比如 tests/test_vae_cpu.py）不必为了取个 dtype 而加载 14B 的 transformer
    dtype = dtype or (pipe.transformer.dtype if pipe.transformer is not None else torch.float32)
    pixels = pipe.video_processor.preprocess(first_frame, height=height, width=width)
    pixels = pixels.to(device=device, dtype=torch.float32)

    # prepare_latents 顺带生成一个随机 latents，我们不需要（编辑路径从 x_src 出发）
    _, condition = pipe.prepare_latents(
        image=pixels,
        batch_size=1,
        num_channels_latents=pipe.vae.config.z_dim,
        height=height,
        width=width,
        num_frames=num_frames,
        dtype=torch.float32,
        device=device,
        generator=None,
        latents=None,
    )
    image_embeds = pipe.encode_image(first_frame, device).to(dtype)
    return condition.to(torch.float32), image_embeds


@torch.no_grad()
def encode_src_tar_prompts(
    pipe,
    src_prompt: str,
    tar_prompt: str,
    negative_prompt: str = "",
    max_sequence_length: int = 512,
    device=None,
) -> torch.Tensor:
    """返回 (4, L, 4096)，顺序固定为 [src_neg, src_pos, tar_neg, tar_pos]。

    与图像版 FlowEdit 的 CFG 拼 batch 方式一致（FlowEdit_utils.py:165）。
    """
    device = device or pipe._execution_device
    src_pos, src_neg = pipe.encode_prompt(
        prompt=src_prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    tar_pos, tar_neg = pipe.encode_prompt(
        prompt=tar_prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=True,
        max_sequence_length=max_sequence_length,
        device=device,
    )
    embeds = torch.cat([src_neg, src_pos, tar_neg, tar_pos], dim=0)
    return embeds.to(pipe.transformer.dtype)


# --------------------------------------------------------------------------- #
# 速度场                                                                        #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def calc_v_wan_i2v(
    pipe,
    zt_src: torch.Tensor,
    zt_tar: torch.Tensor,
    cond_src: torch.Tensor,
    cond_tar: torch.Tensor,
    prompt_embeds: torch.Tensor,
    image_embeds_src: torch.Tensor,
    image_embeds_tar: torch.Tensor,
    src_guidance_scale: float,
    tar_guidance_scale: float,
    t: torch.Tensor,
    micro_batch: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """一次拿到 V_src 与 V_tar，各自做 CFG。

    对应图像版的 ``calc_v_sd3``：4 行一批 [src_neg, src_pos, tar_neg, tar_pos]。
    ``micro_batch`` 用来在显存紧张时把这 4 行拆成若干次前向（纯显存/时间权衡，数值不变）。
    """
    dtype = pipe.transformer.dtype
    src_in = torch.cat([zt_src, cond_src], dim=1)
    tar_in = torch.cat([zt_tar, cond_tar], dim=1)
    hidden = torch.cat([src_in, src_in, tar_in, tar_in], dim=0).to(dtype)
    img_emb = torch.cat([image_embeds_src, image_embeds_src, image_embeds_tar, image_embeds_tar], dim=0)

    outs = []
    for s in range(0, 4, micro_batch):
        e = min(s + micro_batch, 4)
        outs.append(
            pipe.transformer(
                hidden_states=hidden[s:e],
                timestep=t.expand(e - s),
                encoder_hidden_states=prompt_embeds[s:e],
                encoder_hidden_states_image=img_emb[s:e],
                attention_kwargs=None,
                return_dict=False,
            )[0]
        )
    v_src_neg, v_src_pos, v_tar_neg, v_tar_pos = torch.cat(outs, dim=0).float().chunk(4)

    v_src = v_src_neg + src_guidance_scale * (v_src_pos - v_src_neg)
    v_tar = v_tar_neg + tar_guidance_scale * (v_tar_pos - v_tar_neg)
    return v_src, v_tar


# --------------------------------------------------------------------------- #
# SGA —— Similarity Guided Aggregation（论文 5.1，Eq.4-6）                      #
# --------------------------------------------------------------------------- #
def sga_schedule(edit_step: int, n_hi: int = 5, n_lo: int = 1, k: int = 3) -> int:
    """论文 6.1: 前 k 步用 n_hi 个噪声，其余用 n_lo。

    论文写作 ``i > N - 3``（i 从 N 递减）。这里 ``edit_step`` 从 0 递增计数
    【实际执行的】编辑步，所以前 k 步就是 edit_step < k。
    """
    return n_hi if edit_step < k else n_lo


def sga_weights(
    z_edit: torch.Tensor,
    v_stack: torch.Tensor,
    x_src: torch.Tensor,
    t_i: float,
    tau: float,
) -> torch.Tensor:
    """Eq.4 + Eq.5 的权重部分：投影 -> 与源视频算 cosine -> softmax(温度 tau)。"""
    # v_stack 必须是 (n, *z_edit.shape)。形状不对时广播会静默生成多余维度，
    # 后面的 reshape(n, -1) 就会错切，所以这里显式拦住。
    # 用 raise 不用 assert：python -O 会剥掉 assert，而那恰恰是出错时最需要它的场景。
    if v_stack.shape[1:] != z_edit.shape:
        raise ValueError(f"v_stack {tuple(v_stack.shape)} 与 z_edit {tuple(z_edit.shape)} 不匹配")
    n = v_stack.shape[0]
    z_proj = z_edit.unsqueeze(0) - t_i * v_stack                    # Eq.4
    sims = F.cosine_similarity(z_proj.reshape(n, -1), x_src.reshape(1, -1), dim=1)
    return torch.softmax(sims / tau, dim=0)                          # Eq.5


def sga_aggregate(
    z_edit: torch.Tensor,
    v_stack: torch.Tensor,
    x_src: torch.Tensor,
    t_i: float,
    tau: float,
    literal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """返回 (V_bar, weights)。

    代数恒等式（实现上很重要）：因为 softmax 权重和为 1，

        V_bar = (z_edit - sum_j s_j (z_edit - t_i V_j)) / t_i
              = (z_edit - z_edit + t_i sum_j s_j V_j) / t_i
              = sum_j s_j V_j

    也就是说 Eq.4-6 精确等价于"按 softmax-cosine 权重对速度加权平均"，
    投影只是用来【算权重】的。默认走这个等价形式：t_i 在最后几步很小，
    字面形式的 ``/ t_i`` 会放大浮点误差。``literal=True`` 走论文字面公式，
    仅用于验证两者一致。
    """
    if v_stack.shape[1:] != z_edit.shape:
        raise ValueError(f"v_stack {tuple(v_stack.shape)} 与 z_edit {tuple(z_edit.shape)} 不匹配")
    weights = sga_weights(z_edit, v_stack, x_src, t_i, tau)
    if literal:
        z_proj = z_edit.unsqueeze(0) - t_i * v_stack
        z_bar = (weights.view(-1, *([1] * (z_proj.dim() - 1))) * z_proj).sum(0)  # Eq.5
        v_bar = (z_edit - z_bar) / t_i                                            # Eq.6
    else:
        v_bar = (weights.view(-1, *([1] * (v_stack.dim() - 1))) * v_stack).sum(0)
    return v_bar, weights


# --------------------------------------------------------------------------- #
# ANC —— Annealed Noise Correlation（论文 5.2，Eq.7）                           #
# --------------------------------------------------------------------------- #
def anc_schedule(t: float, t_full: float = 0.25) -> float:
    """a_t 线性递增：t=1 时 0，t=t_full 时 1，之后保持 1。"""
    return float(min(max((1.0 - t) / (1.0 - t_full), 0.0), 1.0))


def anc_update(w_tilde: torch.Tensor, w_new: torch.Tensor, a_t: float) -> torch.Tensor:
    """Eq.7: w~ <- sqrt(a_t) * w~ + sqrt(1 - a_t) * w。

    保方差：系数平方和为 1，所以 w~ 的边缘分布始终是 N(0, I)。
    初始 w~ = 0 且 a_{t_N} = 0，第一步退化成纯新噪声。
    """
    return (a_t ** 0.5) * w_tilde + ((1.0 - a_t) ** 0.5) * w_new


def _anc_carry(noises: List[torch.Tensor], weights: Optional[torch.Tensor], mode: str) -> List[torch.Tensor]:
    """n_i^SGA 从 5 掉到 1 时，保留哪一条噪声继续递推。

    ⚠️ 论文未说明。默认 "argmax"：保留 SGA 权重最大的那条，与软选择语义一致。
    """
    if len(noises) <= 1:
        return noises
    if weights is None or mode == "first":
        return [noises[0]]
    if mode == "argmax":
        return [noises[int(torch.argmax(weights).item())]]
    if mode == "weighted":
        # 独立噪声的加权和 std = sqrt(sum w_j^2) < 1，必须重新归一，
        # 否则 w~ 的 N(0,I) 不变量被破坏，且误差会随步数累积。
        w = weights.view(-1, *([1] * noises[0].dim()))
        mixed = (w * torch.stack(noises, 0)).sum(0)
        return [mixed / weights.pow(2).sum().sqrt()]
    raise ValueError(f"unknown anc_carry mode: {mode}")


# --------------------------------------------------------------------------- #
# 噪声采样                                                                      #
# --------------------------------------------------------------------------- #
def draw_noise(x_src: torch.Tensor, seed: Optional[int], step: int, branch: int, salt: int = 0):
    """按 (step, branch) 确定性地采噪声。

    为什么不用全局 torch.randn_like：开关状态会改变每步的采样【次数】
    （SGA 前 3 步要 5 个噪声、关掉只要 1 个），全局流一旦错位，
    此后每一步的噪声都不同 —— SGA on/off 的对比就被"噪声也不一样"污染了。
    按 (step, branch) 定种子后，分支 0 在第 i 步拿到的噪声与 n_i 取 1 还是 5 无关，
    use_sga=False 恰好就是 use_sga=True 的 j=0 分支，A/B 才是干净的。

    seed=None 则退回全局 RNG（与原始 FlowEdit 行为一致）。
    """
    if seed is None:
        return torch.randn_like(x_src)
    g = torch.Generator(device="cpu").manual_seed(
        ((int(seed) * 1000003 + step) * 1009 + branch) * 7 + salt
    )
    return torch.randn(x_src.shape, generator=g, dtype=torch.float32).to(x_src.device)


# --------------------------------------------------------------------------- #
# 主循环 —— 论文 Algorithm 2（两个开关都关时退化为 Algorithm 1 / FlowEdit）      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def DynaEditWanI2V(
    pipe,
    x_src: torch.Tensor,
    cond_src: torch.Tensor,
    cond_tar: torch.Tensor,
    image_embeds_src: torch.Tensor,
    image_embeds_tar: torch.Tensor,
    prompt_embeds: torch.Tensor,
    T_steps: int = 50,
    n_avg: int = 1,
    src_guidance_scale: float = 2.5,
    tar_guidance_scale: float = 4.5,
    n_max: Optional[int] = None,
    # --- SGA ---
    use_sga: bool = False,
    tau: float = 0.001,
    sga_n_hi: int = 5,
    sga_k: int = 3,
    sga_literal: bool = False,
    # --- ANC ---
    use_anc: bool = False,
    anc_t_full: float = 0.25,
    anc_carry: str = "argmax",
    # --- 工程 ---
    noise_seed: Optional[int] = None,
    micro_batch: int = 4,
    verbose: bool = False,
) -> torch.Tensor:
    """返回编辑后的 latent z_edit，形状同 x_src。

    ``use_sga=False, use_anc=False, n_avg=1`` 时逐行等价于视频版 FlowEdit。
    ``n_max=None`` 等价于 n_max = T_steps，即论文用的 n_max = N。

    已知的与论文的偏离（都只在非默认配置下才显现）：
    - ``use_sga=True`` 时 ``n_avg`` 被忽略，每步的噪声个数由 ``sga_schedule`` 决定。
    - ``sga_schedule`` 数的是【实际执行】的编辑步。论文写 ``i > N-3``；n_max=N
      （论文与本项目的默认）时两者相同，n_max<N 时按字面读会导致全程 n_SGA=1，
      这里取"前三个执行步"的读法。⚠️ 属于推断。
    """
    device = x_src.device
    # 与上游 WanImageToVideoPipeline 完全一致的取 timesteps 方式（pipeline_wan_i2v.py:643）
    pipe.scheduler.set_timesteps(T_steps, device=device)
    timesteps = pipe.scheduler.timesteps          # 整数 t，喂给 transformer 的时间嵌入
    sigmas = pipe.scheduler.sigmas                # 精确 sigma，长度 T_steps+1，末位为 0
    T_steps = len(timesteps)
    n_max = T_steps if n_max is None else n_max

    z_edit = x_src.clone().to(torch.float32)
    noises: List[torch.Tensor] = []     # ANC 的持久噪声状态 {w~_j}
    edit_step = 0
    prev_v_bar = None                   # 用于复现论文 Fig.11 的相邻步速度相关
    prev_noise = None

    for i, t in enumerate(tqdm(timesteps, disable=not verbose)):
        if T_steps - i > n_max:         # n_max：跳过最前面的高 t 步（同 FlowEdit_utils.py:173）
            continue

        # 插值系数用 scheduler.sigmas 而不是 t/1000：后者是整数截断近似，
        # 与真 sigma 最大差 ~1e-3（末段相对误差约 1.7%）。FLUX 版 FlowEdit
        # 也是直接取 scheduler.sigmas（FlowEdit_utils.py:328）。
        t_i = sigmas[i].item()
        t_im1 = sigmas[i + 1].item()

        n_i = sga_schedule(edit_step, n_hi=sga_n_hi, k=sga_k) if use_sga else n_avg

        if use_anc:
            a_t = anc_schedule(t_i, t_full=anc_t_full)
            # 新开的链用 fresh N(0,I) 起步，不用 0。
            # Alg.2 L3 的 w~=0 只在【第一步】成立，因为那里 a_{t_N}=0，
            # anc_update(0, w, 0) == w。一旦 a_t>0（中途要补链，或 n_max<N 从中途起步），
            # 补 0 会得到 std=sqrt(1-a_t) 的噪声；a_t 到 1 时更是直接退化成全零张量，
            # 使 zt_src=(1-t_i)x_src 成为模型从未见过的无噪 latent。
            # 用 randn 起步在 a_t=0 时与补 0 完全等价，在 a_t>0 时保方差。
            while len(noises) < n_i:
                noises.append(draw_noise(x_src, noise_seed, i, len(noises), salt=1))
            noises = [
                anc_update(w, draw_noise(x_src, noise_seed, i, j), a_t)      # Eq.7
                for j, w in enumerate(noises[:n_i])
            ]
        else:
            # 原 FlowEdit：跨步 i.i.d.
            noises = [draw_noise(x_src, noise_seed, i, j) for j in range(n_i)]

        v_list = []
        for w in noises:
            zt_src = (1.0 - t_i) * x_src + t_i * w                   # Alg.2 L7
            zt_tar = z_edit + zt_src - x_src                         # Alg.2 L8
            v_src, v_tar = calc_v_wan_i2v(                           # Alg.2 L9
                pipe, zt_src, zt_tar, cond_src, cond_tar,
                prompt_embeds, image_embeds_src, image_embeds_tar,
                src_guidance_scale, tar_guidance_scale, t, micro_batch,
            )
            v_list.append(v_tar - v_src)

        v_stack = torch.stack(v_list, dim=0)
        if use_sga:
            v_bar, weights = sga_aggregate(z_edit, v_stack, x_src, t_i, tau, literal=sga_literal)  # Alg.2 L10
            if verbose and len(v_list) > 1:
                # 诊断：cosine 的绝对值域很窄，softmax 有多尖完全取决于 tau 与这个跨度的比值。
                # 没有这行就无法判断某个 tau 落在「硬选择」还是「软平均」区。
                sims = F.cosine_similarity(
                    (z_edit.unsqueeze(0) - t_i * v_stack).reshape(len(v_list), -1),
                    x_src.reshape(1, -1), dim=1)
                print(f"[sga] step={i} t={t_i:.4f} n={len(v_list)} "
                      f"sim=[{sims.min():.5f}, {sims.max():.5f}] 跨度={sims.max()-sims.min():.2e} "
                      f"w_max={weights.max():.4f} 有效分支数={1.0/weights.pow(2).sum():.2f}", flush=True)
        else:
            v_bar, weights = v_stack.mean(dim=0), None                                             # Alg.1 L9

        if use_anc:
            # 只有当下一个【实际执行】的步需要更少的链时才收缩。
            # 每步都无条件收缩会把 {w~_j} 截成 1 条，下一步再从头补链，
            # 等于 ANC 的跨步相关性每步都被摧毁重建。
            n_next = sga_schedule(edit_step + 1, n_hi=sga_n_hi, k=sga_k) if use_sga else n_avg
            if n_next < len(noises):
                noises = _anc_carry(noises, weights, anc_carry)

        if verbose:
            # 论文 Fig.11：ANC 应使【相邻步的 V_Δ】更相关。这是对 ANC 唯一直接的度量，
            # 不经过 VAE 解码、不依赖任何视频级启发式指标。
            cv = cw = float("nan")
            if prev_v_bar is not None:
                cv = F.cosine_similarity(v_bar.flatten(), prev_v_bar.flatten(), dim=0).item()
            if prev_noise is not None:
                cw = F.cosine_similarity(noises[0].flatten(), prev_noise.flatten(), dim=0).item()
            print(f"[corr] step={i} t={t_i:.4f} a_t={anc_schedule(t_i, anc_t_full) if use_anc else 0.0:.3f} "
                  f"cos_V={cv:+.4f} cos_w={cw:+.4f}", flush=True)
            prev_v_bar = v_bar.clone()
            prev_noise = noises[0].clone()

        z_edit = z_edit + (t_im1 - t_i) * v_bar                      # Alg.2 L11
        edit_step += 1

    return z_edit
