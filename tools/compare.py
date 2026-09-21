"""多路对比：源视频 + 若干变体，一次出对比表和叠放的抽帧图 / 时空切片图。

    python tools/compare.py videos/blackswan_832.mp4 \
        base=outputs/.../p2_base/....mp4 \
        sga001=outputs/.../p2_sga_t0.001/....mp4 \
        -o outputs/compare/swan

指标含义见 tools/analyze.py。判读要点：
  - SGA 要修的是【低频错位】-> 看 corr 是否上升
  - 顺带别把画质做坏 -> 看空间高频保留是否不降反升
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 必须在 import analyze 之前

import numpy as np
from PIL import Image, ImageDraw

from analyze import (grid, jitter, lowfreq_align, read, save_img, sharpness,
                     sharpness_trend, st_slice)


def label(img, text, h=28):
    """在图像上方贴一条黑底白字的标签。"""
    arr = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    out = Image.new("RGB", (arr.shape[1], arr.shape[0] + h), (0, 0, 0))
    out.paste(Image.fromarray(arr), (0, h))
    ImageDraw.Draw(out).text((8, 7), text, fill=(255, 255, 255))
    return np.asarray(out, dtype=np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("variants", nargs="+", help="形如 标签=路径")
    ap.add_argument("-o", "--outdir", default="outputs/compare")
    ap.add_argument("--row", type=float, default=0.5)
    ap.add_argument("--nframes", type=int, default=6, help="抽帧图每行几帧")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    a = read(args.src)
    items = []
    for v in args.variants:
        if "=" not in v:
            raise SystemExit(f"变体要写成 标签=路径，收到: {v}")
        k, p = v.split("=", 1)
        items.append((k, read(p)))
    n = min([len(a)] + [len(b) for _, b in items])

    sa = sharpness(a[:n])
    ta = sharpness_trend(a[:n])
    ja, _, _ = jitter(a[:n])

    print(f"源: {os.path.basename(args.src)}  对齐比较前 {n} 帧")
    print()
    # 趋势要看【与源的距离】而不是大小：低于源=越来越糊，高于源=崩成噪点，两边都是退化
    print(f"{'变体':<16} {'空间高频保留':>12} {'趋势':>8} {'偏离源':>8} {'时间粗糙度比':>12} {'低频运动相关':>12}")
    print(f"{'源(自比)':<16} {1.0:>12.2f} {ta:>8.3f} {0.0:>8.3f} {1.0:>12.2f} {1.0:>12.4f}")
    rows = []
    for k, b in items:
        sb = sharpness(b[:n])
        jb, _, _ = jitter(b[:n])
        corr = lowfreq_align(a[:n], b[:n])
        tb = sharpness_trend(b[:n])
        rows.append((k, sb / (sa + 1e-8), jb / (ja + 1e-8), corr, abs(tb - ta)))
        print(f"{k:<16} {sb / (sa + 1e-8):>12.2f} {tb:>8.3f} {abs(tb - ta):>8.3f} "
              f"{jb / (ja + 1e-8):>12.2f} {corr:>12.4f}")

    if len(rows) > 1:
        print()
        print(f"低频对齐最好: {max(rows, key=lambda r: r[3])[0]}")
        print(f"高频保留最好: {max(rows, key=lambda r: r[1])[0]}")
        print(f"时间稳定最好: {min(rows, key=lambda r: r[4])[0]}  (趋势最接近源)")

    stack_g = [label(grid(a[:n], args.nframes), "source")]
    stack_s = [label(st_slice(a[:n], args.row), "source")]
    for k, b in items:
        stack_g.append(label(grid(b[:n], args.nframes), k))
        stack_s.append(label(st_slice(b[:n], args.row), k))

    base = os.path.basename(args.outdir.rstrip("/")) or "cmp"
    save_img(np.concatenate(stack_g, 0), f"{args.outdir}/{base}_grid.png")
    save_img(np.concatenate(stack_s, 0), f"{args.outdir}/{base}_slice.png")
    print(f"\n图: {args.outdir}/{base}_grid.png 与 {base}_slice.png")


if __name__ == "__main__":
    main()
