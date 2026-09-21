"""对比源视频与编辑结果，量化论文 Fig.2 / Fig.3 描述的两个现象。

    python tools/analyze.py videos/boat_832.mp4 outputs/.../xxx.mp4 -o outputs/analysis

产出：
  1. <name>_slice.png   时空切片 (x-t)。论文 Fig.3 用它看低频错位：
                        源与编辑的条纹走向不一致 = 相机运动/物体轨迹被无谓改变。
  2. <name>_grid.png    等间隔抽帧拼图，肉眼看编辑力与画质。
  3. stdout 两个指标（见下）。

⚠️ 指标是启发式的，不是论文用的评测（论文用 Gemini VLM 打 1-5 分，见阶段 4）。
   它们只用来回答"现象有没有复现"，不用来比较方法优劣。
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def read(path):
    r = imageio.get_reader(path)
    frames = np.stack([f for f in r]).astype(np.float32) / 255.0   # (F, H, W, 3)
    r.close()
    return frames


def st_slice(v, frac=0.5):
    """取固定一行像素，沿时间堆叠 -> (F, W, 3)。静止相机下条纹应竖直。"""
    return v[:, int(v.shape[1] * frac), :, :]


def jitter(v):
    """高频抖动：时间二阶差分能量。

    平滑运动的二阶差分接近 0；逐帧乱跳的高频抖动会把它顶起来。
    除以一阶差分能量做归一，避免"运动大 = 抖动大"的假象。
    """
    g = v.mean(-1)                                   # 转灰度，忽略色彩
    d1 = np.abs(np.diff(g, axis=0)).mean()
    d2 = np.abs(np.diff(g, n=2, axis=0)).mean()
    return d2 / (d1 + 1e-8), d1, d2


def sharpness(v):
    """空间高频能量（梯度幅值均值）。

    论文 n_max=N 基线的退化主要表现为【模糊/涂抹】，不是逐帧乱跳。
    模糊会同时压低时间一阶和二阶差分，所以 jitter() 那个归一化比值看不出来 ——
    必须直接量空间高频的丢失。
    """
    g = v.mean(-1)
    gx = np.abs(np.diff(g, axis=2)).mean()
    gy = np.abs(np.diff(g, axis=1)).mean()
    return (gx + gy) / 2


def sharpness_trend(v, k=3):
    """后 1/3 与前 1/3 的空间高频能量之比。

    论文对缺失 ANC 的描述是"旗子到后面就化掉了"——内容随时间【逐渐】溶解。
    这类退化在全片平均的 sharpness 上会被前段的清晰帧稀释，必须看趋势。
    源视频应接近 1.0；越小说明越往后越糊。
    """
    n = len(v) // k
    return sharpness(v[-n:]) / (sharpness(v[:n]) + 1e-8)


def lowfreq_align(a, b, block=16):
    """低频对齐：把两段视频重度降采样后，比较各自的【运动模式】。

    对每段视频算逐帧低频差分 (F-1, h, w)，展平成时间序列后算相关系数。
    相机运动、大物体轨迹被改变 -> 相关系数掉下来。
    """
    def lf(v):
        f, h, w, _ = v.shape
        h2, w2 = h // block, w // block
        g = v.mean(-1)[:, : h2 * block, : w2 * block]
        g = g.reshape(f, h2, block, w2, block).mean((2, 4))      # 块平均 = 低通
        return np.diff(g, axis=0).reshape(f - 1, -1)

    x, y = lf(a), lf(b)
    n = min(len(x), len(y))
    x, y = x[:n].ravel(), y[:n].ravel()
    x, y = x - x.mean(), y - y.mean()
    return float((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-8))


def save_img(arr, path):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path)


def grid(v, n=6):
    idx = np.linspace(0, len(v) - 1, n).astype(int)
    return np.concatenate([v[i] for i in idx], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("edit")
    ap.add_argument("-o", "--outdir", default="outputs/analysis")
    ap.add_argument("--row", type=float, default=0.5, help="时空切片取第几行（0-1）")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    a, b = read(args.src), read(args.edit)
    name = os.path.splitext(os.path.basename(args.edit))[0]
    n = min(len(a), len(b))
    print(f"源 {a.shape}  编辑 {b.shape}  对齐比较前 {n} 帧")

    # 时空切片：上下拼，方便直接比条纹走向
    sl = np.concatenate([st_slice(a[:n], args.row), st_slice(b[:n], args.row)], axis=0)
    save_img(sl, f"{args.outdir}/{name}_slice.png")
    save_img(np.concatenate([grid(a[:n]), grid(b[:n])], axis=0), f"{args.outdir}/{name}_grid.png")

    ja, d1a, d2a = jitter(a[:n])
    jb, d1b, d2b = jitter(b[:n])
    sa, sb = sharpness(a[:n]), sharpness(b[:n])
    ta, tb = sharpness_trend(a[:n]), sharpness_trend(b[:n])
    corr = lowfreq_align(a[:n], b[:n])

    print()
    print("空间高频能量（梯度幅值，越小越糊）  ← 论文基线的主要退化形式")
    print(f"    源     {sa:.5f}")
    print(f"    编辑   {sb:.5f}   保留 {sb / (sa + 1e-8):.2f}x  "
          f"{'<- 明显模糊' if sb / (sa + 1e-8) < 0.85 else ''}")
    print("高频随时间的趋势（后1/3 ÷ 前1/3，<1 = 越往后越糊）  ← ANC 要修的形态")
    print(f"    源     {ta:.3f}")
    print(f"    编辑   {tb:.3f}   {'<- 内容随时间溶解' if tb < 0.85 * ta else ''}")
    print("时间粗糙度（二阶差分 / 一阶差分）")
    print(f"    源     {ja:.4f}   (d1={d1a:.5f} d2={d2a:.5f})")
    print(f"    编辑   {jb:.4f}   (d1={d1b:.5f} d2={d2b:.5f})   比值 {jb / (ja + 1e-8):.2f}x")
    print("低频运动对齐（块平均后的帧间差分相关，1=完全一致，0=无关）")
    print(f"    corr   {corr:+.4f}   {'<- 低频错位明显' if corr < 0.7 else ''}")
    print()
    print(f"图已存到 {args.outdir}/{name}_slice.png 与 _grid.png")
    print("判读：复现论文 Fig.2 的基线现象 = 画面确实按 target prompt 改了（有编辑力），")
    print("      但空间高频能量明显丢失（模糊/涂抹），且 corr 明显低于 1（低频错位）。")
    print("      注意时间粗糙度对【模糊】这种退化不敏感，甚至会反向 —— 模糊会同时压低 d1 和 d2。")


if __name__ == "__main__":
    main()
