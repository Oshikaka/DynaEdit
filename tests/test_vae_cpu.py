"""用真实 WAN VAE + CLIP（CPU）验证 I2V 管线的形状与归一化往返。

不加载 14B transformer，所以登录节点就能跑，几分钟内出结果。
验证的正是最容易错、又只有真模型才能确认的那几件事：
  - x_src / condition 的形状，以及拼起来是不是恰好 36 通道（transformer 的 in_channels）
  - latent 帧数换算 (F-1)//4 + 1
  - VAE 归一化 encode/decode 往返

    HF_HUB_OFFLINE=1 python tests/test_vae_cpu.py [帧数]
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

from DynaEdit_utils import decode_latent_to_video, encode_video_to_latent, prepare_i2v_condition
from run_script import load_video

MODEL = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
ok = lambda m: print(f"  PASS  {m}")


def main():
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    H, W = 480, 832
    assert (n_frames - 1) % 4 == 0

    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)
    # 只要 VAE + CLIP；transformer/T5 传 None 跳过加载（32 GB + 11 GB 不必要）
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL, vae=vae, transformer=None, text_encoder=None, torch_dtype=torch.float32)
    print(f"  载入 VAE+CLIP 用时 {time.time() - t0:.1f}s")

    video, fps = load_video("videos/boat_832.mp4")
    video = video[:n_frames]
    dev = torch.device("cpu")

    t0 = time.time()
    x_src = encode_video_to_latent(pipe, video, H, W, device=dev)
    cond, img_emb = prepare_i2v_condition(pipe, video[0], n_frames, H, W, device=dev)
    print(f"  编码 {n_frames} 帧用时 {time.time() - t0:.1f}s")

    # --- 形状 ---
    f_lat = (n_frames - 1) // 4 + 1
    assert x_src.shape == (1, 16, f_lat, H // 8, W // 8), x_src.shape
    ok(f"x_src {tuple(x_src.shape)}  ({n_frames} 帧 -> {f_lat} latent 帧, 480x832 -> 60x104)")
    assert cond.shape == (1, 20, f_lat, H // 8, W // 8), cond.shape
    ok(f"condition {tuple(cond.shape)}  (4 时间 mask + 16 首帧 latent)")

    merged = torch.cat([x_src, cond], dim=1)
    in_ch = 36
    assert merged.shape[1] == in_ch, merged.shape
    ok(f"拼接后 {merged.shape[1]} 通道 == transformer 的 in_channels ({in_ch})")

    assert img_emb.shape == (1, 257, 1280), img_emb.shape
    ok(f"CLIP image_embeds {tuple(img_emb.shape)}")

    # --- condition 的语义：mask 首帧为 1、其余为 0 ---
    mask = cond[:, :4]
    assert mask[:, :, 0].min() > 0.99, mask[:, :, 0].min()
    if f_lat > 1:
        assert mask[:, :, 1:].abs().max() < 1e-6, mask[:, :, 1:].abs().max()
    ok("时间 mask: 第 0 个 latent 帧全 1，其余全 0（只条件化首帧）")

    # --- 归一化往返 ---
    t0 = time.time()
    rec = decode_latent_to_video(pipe, x_src, output_type="np", device=dev)[0]
    print(f"  解码用时 {time.time() - t0:.1f}s")
    import numpy as np
    src = np.stack([np.asarray(f, dtype=np.float32) / 255.0 for f in video])
    src = src[:, :H, :W]
    err = np.abs(rec[:len(src)] - src).mean()
    assert rec.shape[0] == n_frames, rec.shape
    assert err < 0.06, err        # VAE 有损，只要不是归一化搞反（那会是 0.3~0.5）
    ok(f"encode->decode 往返帧数 {rec.shape[0]}，平均绝对误差 {err:.4f}（VAE 有损，归一化方向正确）")

    # 归一化搞反会是什么样，给个对照
    bad = np.abs(rec[:len(src)] - src[::-1]).mean()
    print(f"  参照：与时序颠倒的源比 {bad:.4f}（说明 {err:.4f} 确实是重建而非巧合）")

    print("\n全部通过。")


if __name__ == "__main__":
    main()
