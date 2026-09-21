"""SGA / ANC 的最小验证。纯 CPU，不需要模型权重。

    python tests/test_modules.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from DynaEdit_utils import (
    anc_schedule, anc_update, sga_aggregate, sga_schedule, sga_weights, _anc_carry,
)

torch.manual_seed(0)
SHAPE = (1, 16, 5, 6, 8)          # 缩小版 (B, C, F', h, w)
ok = lambda name: print(f"  PASS  {name}")


def test_sga():
    print("[SGA]")
    x_src = torch.randn(SHAPE)
    z_edit = torch.randn(SHAPE)
    t_i = 0.4

    # v_stack 的构造方式必须与主循环一致：torch.stack(v_list, 0) -> (n, *SHAPE)
    stack = lambda n: torch.stack([torch.randn(SHAPE) for _ in range(n)], dim=0)

    # 1) n=1 时必须恒等于唯一那个 V（softmax 单元素 = 1）
    v1 = stack(1)
    v_bar, w = sga_aggregate(z_edit, v1, x_src, t_i, tau=1.0)
    assert torch.allclose(w, torch.ones(1)), w
    assert torch.equal(v_bar, v1[0]), (v_bar - v1[0]).abs().max()
    ok("n=1 -> V_bar == V_1，权重 == 1")

    # 2) 权重归一
    v = stack(5)
    w = sga_weights(z_edit, v, x_src, t_i, tau=1.0)
    assert abs(w.sum().item() - 1.0) < 1e-6, w.sum()
    assert (w >= 0).all()
    ok("权重非负且和为 1")

    # 3) 论文字面式 (Eq.4-6) 与等价式 sum_j s_j V_j 数值一致。
    #    误差随 1/t_i 增长，所以必须扫到实际会用到的最小 t_i：
    #    T=50 / flow_shift=3 时最小 sigma 是 0.0576。
    errs = {}
    for tt in [0.5, 0.0576, 0.01, 1e-3]:
        a, _ = sga_aggregate(z_edit, v, x_src, tt, tau=1.0, literal=True)
        b, _ = sga_aggregate(z_edit, v, x_src, tt, tau=1.0, literal=False)
        errs[tt] = ((a - b).abs().max() / b.abs().max()).item()
    assert errs[0.0576] < 1e-4, errs          # 实际会用到的最小 t_i 必须安全
    assert errs[1e-3] > errs[0.5]             # 确认误差确实随 t_i 变小而放大
    ok("Eq.4-6 字面式 == 加权平均等价式，rel err @ t_i=" +
       ", ".join(f"{k:g}:{v:.1e}" for k, v in errs.items()))

    # 4) tau -> 0 退化为硬选择（one-hot），且选中 cosine 最大者
    w_hard = sga_weights(z_edit, v, x_src, t_i, tau=1e-6)
    z_proj = z_edit.unsqueeze(0) - t_i * v
    sims = F.cosine_similarity(z_proj.reshape(5, -1), x_src.reshape(1, -1), dim=1)
    assert w_hard.max() > 0.999, w_hard
    assert int(w_hard.argmax()) == int(sims.argmax())
    v_hard, _ = sga_aggregate(z_edit, v, x_src, t_i, tau=1e-6)
    assert torch.allclose(v_hard, v[int(sims.argmax())], atol=1e-4)
    ok("tau->0 退化为硬选择，选中与源视频 cosine 最大的那条")

    # 5) tau 很大 -> 趋近均匀平均（即原 FlowEdit 的行为）
    v_soft, w_soft = sga_aggregate(z_edit, v, x_src, t_i, tau=1e6)
    assert (w_soft - 0.2).abs().max() < 1e-3, w_soft
    assert (v_soft - v.mean(0)).abs().max() < 1e-3
    ok("tau->inf 退化为均匀平均（== 原 FlowEdit 的 n_avg 平均）")

    # 6) schedule: 前 3 步 5 个噪声，之后 1 个
    got = [sga_schedule(i) for i in range(6)]
    assert got == [5, 5, 5, 1, 1, 1], got
    ok(f"n_SGA schedule = {got}")


def test_anc():
    print("[ANC]")
    # 1) a_t 的三个锚点
    assert abs(anc_schedule(1.00) - 0.0) < 1e-9
    assert abs(anc_schedule(0.25) - 1.0) < 1e-9
    assert abs(anc_schedule(0.10) - 1.0) < 1e-9          # t < 0.25 保持 1
    assert abs(anc_schedule(0.625) - 0.5) < 1e-9         # 对 t 线性
    assert all(anc_schedule(t) <= anc_schedule(t - 0.05) + 1e-12 for t in [1.0, 0.8, 0.6, 0.4])
    ok("a_t: t=1->0, t=0.25->1, t<0.25 保持 1, 对 t 线性且单调递增")

    # 2) 边界: a=0 纯新噪声, a=1 完全复用
    w_old, w_new = torch.randn(SHAPE), torch.randn(SHAPE)
    assert torch.allclose(anc_update(w_old, w_new, 0.0), w_new)
    assert torch.allclose(anc_update(w_old, w_new, 1.0), w_old)
    ok("a=0 -> 纯新噪声；a=1 -> 完全复用上一步")

    # 3) 初始化 w~=0 且 a_{t_N}=0 -> 第一步就是纯新噪声
    w0 = torch.zeros(SHAPE)
    assert torch.allclose(anc_update(w0, w_new, anc_schedule(1.0)), w_new)
    ok("Alg.2 L3 初始化 w~=0 配 a_{t_N}=0 -> 第一步退化为纯 i.i.d. 噪声")

    # 4) 保方差：迭代整条 schedule 后边缘分布仍是 N(0,I)
    big = (1, 16, 8, 30, 52)
    w = torch.zeros(big)
    ts = torch.linspace(1.0, 0.0, 50)
    for t in ts:
        w = anc_update(w, torch.randn(big), anc_schedule(float(t)))
    m, s = w.mean().item(), w.std().item()
    assert abs(m) < 0.01 and abs(s - 1.0) < 0.01, (m, s)
    ok(f"跑完 50 步 schedule 后 w~ 仍是 N(0,I)（mean={m:+.4f}, std={s:.4f}）")

    # 5) ANC 确实制造了跨步相关，且相关性随 t 下降而升高
    def run(use_anc):
        w, cors = torch.zeros(big), []
        prev = None
        for t in ts:
            w = anc_update(w, torch.randn(big), anc_schedule(float(t))) if use_anc else torch.randn(big)
            if prev is not None:
                cors.append(F.cosine_similarity(w.flatten(), prev.flatten(), dim=0).item())
            prev = w.clone()
        return cors

    c_anc, c_iid = run(True), run(False)
    assert max(abs(c) for c in c_iid) < 0.02, max(c_iid)
    early, late = sum(c_anc[:10]) / 10, sum(c_anc[-10:]) / 10
    assert early < late and late > 0.9, (early, late)
    ok(f"i.i.d. 相邻步相关≈0（max |cos|={max(abs(c) for c in c_iid):.4f}）；"
       f"ANC 前段 {early:.3f} -> 后段 {late:.3f}，递增")

    # 6) n_SGA 从 5 掉到 1 时的噪声保留策略
    noises = [torch.randn(SHAPE) for _ in range(5)]
    weights = torch.tensor([0.1, 0.1, 0.6, 0.1, 0.1])
    assert torch.equal(_anc_carry(noises, weights, "argmax")[0], noises[2])
    assert torch.equal(_anc_carry(noises, weights, "first")[0], noises[0])
    mixed = _anc_carry(noises, weights, "weighted")
    assert len(mixed) == 1
    # 独立噪声加权和的 std 是 sqrt(sum w^2) < 1，必须重新归一回 1
    big_noises = [torch.randn(1, 16, 8, 30, 52) for _ in range(5)]
    w5 = torch.tensor([0.24, 0.082, 0.043, 0.011, 0.624])
    sd = _anc_carry(big_noises, w5, "weighted")[0].std().item()
    assert abs(sd - 1.0) < 0.02, f"weighted 混合后 std={sd}，未保方差"
    ok(f"anc_carry: argmax / first 正确；weighted 保方差（std={sd:.4f}，未归一时应为 "
       f"{float(w5.pow(2).sum().sqrt()):.4f}）")


def test_offswitch_equivalence():
    """开关全关的等价性【不在这里测】。

    本文件只测纯函数。"关掉开关后主循环等于 FlowEdit"是整个仓库最重要的保证，
    必须对【主循环本身】做端到端比对，否则就是在测 torch 而不是测自己的代码。
    见 tests/test_loop.py::test_regression（stub transformer + 真 UniPC scheduler，
    与从零独立写的 Algorithm 1 参考实现逐比特比对）。
    """
    print("[开关全关等价性]")
    ok("已移到 tests/test_loop.py::test_regression（端到端逐比特比对）")


if __name__ == "__main__":
    test_sga()
    test_anc()
    test_offswitch_equivalence()
    print("\n全部通过。")
