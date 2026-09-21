"""DynaEdit 入口。配置风格沿用图像版 FlowEdit 的 run_script.py：
超参走 <exp>.yaml，编辑内容走 edits.yaml。

    python run_script.py --exp_yaml configs/wan_i2v_exp.yaml
"""

import argparse
import gc
import os
import random

import imageio
import numpy as np
import torch
import yaml
from PIL import Image

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanImageToVideoPipeline
from diffusers.utils import export_to_video

from DynaEdit_utils import (
    DynaEditWanI2V,
    decode_latent_to_video,
    encode_src_tar_prompts,
    encode_video_to_latent,
    prepare_i2v_condition,
)


def load_video(path):
    """mp4 -> (List[PIL.Image], fps)"""
    reader = imageio.get_reader(path)
    fps = reader.get_meta_data()["fps"]
    frames = [Image.fromarray(f) for f in reader]
    reader.close()
    return frames, fps


def resolve_negative_prompt(exp):
    """字面 negative_prompt 优先；否则查 configs/negative_prompts.yaml 里的预设。"""
    if "negative_prompt" in exp and exp["negative_prompt"] is not None:
        return exp["negative_prompt"]
    name = exp.get("negative_prompt_preset", "none")
    with open("configs/negative_prompts.yaml") as f:
        presets = yaml.safe_load(f)
    if name not in presets:
        raise ValueError(f"未知的 negative_prompt_preset: {name}；可选 {sorted(presets)}")
    return presets[name]


def free(*modules, device="cpu"):
    for m in modules:
        m.to(device)
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--exp_yaml", type=str, default="configs/wan_i2v_exp.yaml")
    parser.add_argument("--items", type=str, default=None,
                        help="只跑 edits.yaml 里这些 name（逗号分隔）。一条视频一个作业时用")
    parser.add_argument("--exps", type=str, default=None,
                        help="只跑这些 exp_name（逗号分隔）。一个 yaml 放多组实验时用")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device_number}" if torch.cuda.is_available() else "cpu")

    with open(args.exp_yaml) as f:
        raw = yaml.load(f, Loader=yaml.FullLoader)
    # 两种格式都支持：
    #   纯列表（图像版 FlowEdit 的风格，阶段 1 在用）
    #   {defaults: {...}, experiments: [{...}]}（扫参时避免大段重复）
    if isinstance(raw, dict):
        defaults = raw.get("defaults", {})
        exp_configs = [{**defaults, **e} for e in raw["experiments"]]
    else:
        exp_configs = raw

    if args.exps:
        want = [x.strip() for x in args.exps.split(",")]
        known = {e["exp_name"] for e in exp_configs}
        missing = set(want) - known
        if missing:
            raise ValueError(f"--exps 里这些 exp_name 不存在: {sorted(missing)}；可选 {sorted(known)}")
        exp_configs = [e for e in exp_configs if e["exp_name"] in want]

    model_ids = {e["model_id"] for e in exp_configs}
    if len(model_ids) > 1:
        raise ValueError(f"一个 yaml 里只能用一个 model_id，当前有 {model_ids}")
    model_id = model_ids.pop()
    # WAN 的 VAE 必须 fp32，bf16 下会出 artifact；transformer / T5 走 bf16
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    base_scheduler_config = dict(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)

    for exp in exp_configs:
        # flow_shift 会改变每个 sigma，进而改变每个 t_i 和 a_t，所以必须按 exp 重建
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            base_scheduler_config, flow_shift=exp.get("flow_shift", 3.0)
        )
        seed = exp["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        with open(exp["dataset_yaml"]) as f:
            dataset = yaml.load(f, Loader=yaml.FullLoader)
        if args.items:
            want = [x.strip() for x in args.items.split(",")]
            known = {d["name"] for d in dataset}
            missing = set(want) - known
            if missing:
                raise ValueError(f"--items 里这些 name 不存在: {sorted(missing)}；可选 {sorted(known)}")
            dataset = [d for d in dataset if d["name"] in want]

        for item in dataset:
            # 逐条覆盖：论文 App.D.2 就是按每个例子分配不同的超参组。
            # 扫参时要显式控制超参，必须能关掉覆盖，否则被覆盖的那条不会参与扫描。
            cfg = dict(exp)
            if not exp.get("ignore_overrides", False):
                cfg.update(item.get("overrides", {}))

            video, fps = load_video(item["input_video"])
            n_frames = cfg.get("num_frames") or len(video)
            # 这两个检查必须在 VAE/T5 编码之前做：否则要等几十秒的 fp32 VAE 跑完，
            # 才会在 diffusers 内部炸出一个看不懂的 reshape 报错。
            if (n_frames - 1) % 4 != 0:
                raise ValueError(
                    f"{item['name']}: WAN 要求帧数为 4k+1，num_frames={n_frames} 不合法")
            if len(video) < n_frames:
                raise ValueError(
                    f"{item['name']}: 视频只有 {len(video)} 帧，不够 num_frames={n_frames}")
            video = video[:n_frames]
            height = cfg.get("height", 480)
            width = cfg.get("width", 832)

            first_frame_src = video[0]
            tar_ff_path = item.get("target_first_frame")
            first_frame_tar = Image.open(tar_ff_path).convert("RGB") if tar_ff_path else first_frame_src

            # --- 阶段 1/3: 文本编码（T5 上卡，用完下卡）---
            pipe.text_encoder.to(device)
            prompt_embeds = encode_src_tar_prompts(
                pipe, item["source_prompt"], item["target_prompt"],
                resolve_negative_prompt(cfg), device=device,
            )
            free(pipe.text_encoder)

            # --- 阶段 2/3: 视觉编码（VAE + CLIP 上卡，用完下卡）---
            pipe.vae.to(device)
            pipe.image_encoder.to(device)
            # 显存峰值其实在这里而不是 transformer：fp32 的 WAN VAE 要一次吃下
            # 整段 832x480xF 的视频。单卡 64 GB 上吃紧就开 tiling。
            if cfg.get("vae_tiling", False):
                pipe.vae.enable_tiling()
            x_src = encode_video_to_latent(pipe, video, height, width, device=device)
            cond_src, img_emb_src = prepare_i2v_condition(
                pipe, first_frame_src, n_frames, height, width, device=device)
            if tar_ff_path:
                cond_tar, img_emb_tar = prepare_i2v_condition(
                    pipe, first_frame_tar, n_frames, height, width, device=device)
            else:
                # 没有 edited first frame 时 f_tar == f_src，再编码一次是逐字节重复劳动，
                # 而 fp32 WAN VAE 的编码正是显存峰值所在
                cond_tar, img_emb_tar = cond_src, img_emb_src
            if cfg.get("vae_tiling", False):
                pipe.vae.disable_tiling()
            free(pipe.vae, pipe.image_encoder)
            print(f"[shape] x_src={tuple(x_src.shape)} cond={tuple(cond_src.shape)} "
                  f"img_emb={tuple(img_emb_src.shape)} frames={n_frames}", flush=True)

            # --- 阶段 3/3: 编辑循环（transformer 上卡）---
            pipe.transformer.to(device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)
            z_edit = DynaEditWanI2V(
                pipe, x_src, cond_src, cond_tar, img_emb_src, img_emb_tar, prompt_embeds,
                T_steps=cfg["T_steps"],
                n_avg=cfg["n_avg"],
                src_guidance_scale=cfg["src_guidance_scale"],
                tar_guidance_scale=cfg["tar_guidance_scale"],
                n_max=cfg.get("n_max"),
                use_sga=cfg.get("use_sga", False),
                tau=cfg.get("tau", 0.001),
                sga_n_hi=cfg.get("sga_n_hi", 5),
                sga_k=cfg.get("sga_k", 3),
                sga_literal=cfg.get("sga_literal", False),
                use_anc=cfg.get("use_anc", False),
                anc_t_full=cfg.get("anc_t_full", 0.25),
                anc_carry=cfg.get("anc_carry", "argmax"),
                noise_seed=None if cfg.get("global_rng", False) else seed,
                micro_batch=cfg.get("micro_batch", 4),
                verbose=True,
            )
            if torch.cuda.is_available():
                print(f"[mem] 编辑循环峰值显存 {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB "
                      f"(micro_batch={cfg.get('micro_batch', 4)}, {n_frames} 帧)", flush=True)
            free(pipe.transformer)

            # --- 解码 ---
            pipe.vae.to(device)
            frames = decode_latent_to_video(pipe, z_edit, output_type="np", device=device)[0]
            free(pipe.vae)

            tag = (f"sga{int(cfg.get('use_sga', False))}_anc{int(cfg.get('use_anc', False))}"
                   f"_cfg{cfg['src_guidance_scale']}-{cfg['tar_guidance_scale']}"
                   f"_tau{cfg.get('tau', 0.001)}_T{cfg['T_steps']}"
                   f"_neg{cfg.get('negative_prompt_preset', 'none')}_seed{seed}")
            save_dir = os.path.join("outputs", cfg["exp_name"], item["name"])
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"{tag}.mp4")
            export_to_video(frames, out_path, fps=int(round(fps)))

            with open(os.path.join(save_dir, f"{tag}.txt"), "w") as f:
                f.write(f"source_prompt: {item['source_prompt']}\n")
                f.write(f"target_prompt: {item['target_prompt']}\n")
                f.write(f"target_first_frame: {tar_ff_path or '(源视频首帧)'}\n")
                f.write(f"negative_prompt: {resolve_negative_prompt(cfg)!r}\n")
                f.write(f"overrides: {item.get('overrides', {})}\n")
                for k, v in cfg.items():          # cfg 含逐条覆盖，exp 不含
                    f.write(f"{k}: {v}\n")
            print(f"[saved] {out_path}", flush=True)

    print("Done")


if __name__ == "__main__":
    main()
