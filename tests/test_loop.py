"""主循环的端到端验证：stub transformer + 真 UniPC scheduler，纯 CPU。

回答三个问题：
  1. 开关全关时，DynaEditWanI2V 是否【逐比特】等于从零独立写的 FlowEdit 参考实现？
  2. ANC 的噪声链是否真的跨步持续，而不是每步被摧毁重建？
  3. SGA on/off 的噪声流是否对齐（否则 A/B 对比会被"噪声也变了"污染）？

    python tests/test_loop.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import UniPCMultistepScheduler

import DynaEdit_utils as D
from DynaEdit_utils import DynaEditWanI2V, draw_noise

ok = lambda m: print(f"  PASS  {m}")
MODEL = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
SHAPE = (1, 16, 3, 6, 8)          # (B, C, F', h, w) 缩小版
CSHAPE = (1, 20, 3, 6, 8)         # 4 mask + 16 cond


class StubTransformer:
    """确定性假速度场。必须依赖 hidden / timestep / 文本嵌入，
    否则 CFG 两支相减恒为 0，测试就发现不了顺序错误。"""
    dtype = torch.float32

    def __call__(self, hidden_states, timestep, encoder_hidden_states,
                 encoder_hidden_states_image=None, attention_kwargs=None, return_dict=True):
        z, c = hidden_states[:, :16], hidden_states[:, 16:]
        t = timestep.view(-1, 1, 1, 1, 1).float() / 1000.0
        eh = encoder_hidden_states.mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        ei = encoder_hidden_states_image.mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)
        out = torch.tanh(z * 0.5 + c.mean(1, keepdim=True)) * (1.0 + t) + eh + 0.3 * ei
        return (out,)


class StubPipe:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.transformer = StubTransformer()


def make_inputs():
    g = torch.Generator().manual_seed(7)
    r = lambda sh: torch.randn(sh, generator=g)
    return dict(
        x_src=r(SHAPE),
        cond_src=r(CSHAPE), cond_tar=r(CSHAPE),
        image_embeds_src=r((1, 5, 4)), image_embeds_tar=r((1, 5, 4)),
        prompt_embeds=r((4, 7, 4)),     # [src_neg, src_pos, tar_neg, tar_pos]
    )


def reference_flowedit(pipe, inp, T, n_avg, n_max, gs_src, gs_tar, seed):
    """从零独立写的 FlowEdit（Algorithm 1），不复用被测代码的任何循环逻辑。"""
    pipe.scheduler.set_timesteps(T, device="cpu")
    ts, sig = pipe.scheduler.timesteps, pipe.scheduler.sigmas
    n_max = T if n_max is None else n_max
    x = inp["x_src"]
    z = x.clone().to(torch.float32)

    for i, t in enumerate(ts):
        if T - i > n_max:
            continue
        ti, tim1 = sig[i].item(), sig[i + 1].item()
        acc = torch.zeros_like(x)
        for k in range(n_avg):
            w = draw_noise(x, seed, i, k)
            zs = (1 - ti) * x + ti * w
            zt = z + zs - x
            si = torch.cat([zs, inp["cond_src"]], 1)
            ti_ = torch.cat([zt, inp["cond_tar"]], 1)
            hidden = torch.cat([si, si, ti_, ti_], 0)
            img = torch.cat([inp["image_embeds_src"], inp["image_embeds_src"],
                             inp["image_embeds_tar"], inp["image_embeds_tar"]], 0)
            o = pipe.transformer(hidden_states=hidden, timestep=t.expand(4),
                                 encoder_hidden_states=inp["prompt_embeds"],
                                 encoder_hidden_states_image=img)[0].float()
            sn, sp, tn, tp = o.chunk(4)
            v_src = sn + gs_src * (sp - sn)
            v_tar = tn + gs_tar * (tp - tn)
            acc = acc + (v_tar - v_src) / n_avg
        z = z + (tim1 - ti) * acc
    return z


def test_regression():
    print("[开关全关 == FlowEdit，逐比特]")
    sch = UniPCMultistepScheduler.from_pretrained(MODEL, subfolder="scheduler")
    pipe, inp = StubPipe(sch), make_inputs()

    for T, n_avg, n_max, mb in [(10, 1, None, 4), (10, 1, None, 1), (10, 1, None, 2),
                                (10, 1, 6, 4), (4, 1, None, 2), (8, 3, None, 4)]:
        got = DynaEditWanI2V(
            pipe, inp["x_src"], inp["cond_src"], inp["cond_tar"],
            inp["image_embeds_src"], inp["image_embeds_tar"], inp["prompt_embeds"],
            T_steps=T, n_avg=n_avg, n_max=n_max,
            src_guidance_scale=2.5, tar_guidance_scale=4.5,
            use_sga=False, use_anc=False, noise_seed=123, micro_batch=mb,
        )
        ref = reference_flowedit(pipe, inp, T, n_avg, n_max, 2.5, 4.5, seed=123)
        d = (got - ref).abs().max().item()
        tol = 0.0 if n_avg == 1 else 1e-6       # n_avg>1 只差在求和顺序
        assert d <= tol, (T, n_avg, n_max, mb, d)
        ok(f"T={T} n_avg={n_avg} n_max={n_max} micro_batch={mb} -> max|diff| = {d:.3e}")

    print("  （micro_batch 不改变数值，n_max 截断行为与参考一致）")


def test_anc_chains_persist():
    print("[ANC 噪声链跨步持续]")
    sch = UniPCMultistepScheduler.from_pretrained(MODEL, subfolder="scheduler")
    pipe, inp = StubPipe(sch), make_inputs()

    stds, n_new = [], 0
    real_update, real_draw = D.anc_update, D.draw_noise

    def spy_update(w, wn, a):
        out = real_update(w, wn, a)
        stds.append(out.std().item())
        return out

    def spy_draw(x, seed, step, branch, salt=0):
        nonlocal n_new
        if salt == 1:
            n_new += 1
        return real_draw(x, seed, step, branch, salt)

    D.anc_update, D.draw_noise = spy_update, spy_draw
    try:
        # 这正是 review 指出会退化成全零张量的配置：ANC 开、SGA 关、n_avg>1
        DynaEditWanI2V(
            pipe, inp["x_src"], inp["cond_src"], inp["cond_tar"],
            inp["image_embeds_src"], inp["image_embeds_tar"], inp["prompt_embeds"],
            T_steps=50, n_avg=5, use_sga=False, use_anc=True, noise_seed=1,
        )
    finally:
        D.anc_update, D.draw_noise = real_update, real_draw

    assert len(stds) == 50 * 5, len(stds)
    lo, hi = min(stds), max(stds)
    assert lo > 0.9, f"出现了被打回原形的噪声链，最小 std={lo}"
    assert hi < 1.1, hi
    ok(f"250 次 Eq.7 更新后 w~ 的 std 全在 [{lo:.4f}, {hi:.4f}]，无退化、无全零")
    assert n_new == 5, f"链被重建了 {n_new} 次，应只在第一步创建 5 条"
    ok(f"链只在第一步创建 {n_new} 条，之后全程复用（修复前每步都会重建 4 条）")

    # SGA 开时 n_i 从 5 掉到 1，此时才允许收缩
    stds.clear(); n_new = 0
    D.anc_update, D.draw_noise = spy_update, spy_draw
    try:
        DynaEditWanI2V(
            pipe, inp["x_src"], inp["cond_src"], inp["cond_tar"],
            inp["image_embeds_src"], inp["image_embeds_tar"], inp["prompt_embeds"],
            T_steps=20, use_sga=True, use_anc=True, tau=0.001, noise_seed=1,
        )
    finally:
        D.anc_update, D.draw_noise = real_update, real_draw
    assert len(stds) == 5 * 3 + 1 * 17, len(stds)
    assert min(stds) > 0.9, min(stds)
    assert n_new == 5, n_new
    ok(f"SGA+ANC: 前 3 步 5 条链、之后收缩到 1 条，std 最小 {min(stds):.4f}")


def test_ab_noise_alignment():
    print("[SGA on/off 的噪声流对齐]")
    x = torch.zeros(SHAPE)
    for i in [0, 1, 2, 7]:
        a = draw_noise(x, 42, i, 0)          # use_sga=False 时只有 j=0
        b = draw_noise(x, 42, i, 0)          # use_sga=True  时的 j=0 分支
        assert torch.equal(a, b)
        assert not torch.equal(a, draw_noise(x, 42, i, 1))
        assert not torch.equal(a, draw_noise(x, 42, i + 1, 0))
    ok("分支 0 的噪声与该步取 1 个还是 5 个噪声无关；不同分支/不同步互相独立")
    assert not torch.equal(draw_noise(x, 1, 0, 0), draw_noise(x, 2, 0, 0))
    ok("不同 seed 给出不同噪声")
    g = draw_noise(x, None, 0, 0)
    assert g.shape == x.shape
    ok("noise_seed=None 退回全局 RNG（与原始 FlowEdit 一致）")


if __name__ == "__main__":
    test_regression()
    test_anc_chains_persist()
    test_ab_noise_alignment()
    print("\n全部通过。")
