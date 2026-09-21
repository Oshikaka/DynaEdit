"""把浏览器解不了的 mp4 转成 H.264，不动原文件。

    python tools/to_h264.py videos/*.mp4                 # 输出到 videos/web/
    python tools/to_h264.py videos/bear_832.mp4 -o /tmp   # 指定输出目录

起因：videos/ 里 bear / blackswan / boat / snowboard 是 mp4v（MPEG-4 Part 2）编的，
Chrome / Safari / Firefox / VS Code 预览都只解 H.264、VP8/9、AV1，碰到 mp4v 直接报
加载失败。jeep / rabbit / sea_turtle 是 avc1（H.264），所以那三个能放。

**原文件一律保留**：videos/ 下的是采样管线的输入，转码是有损的，换掉会让之后的
实验和 doc/phase1-3 已记录的结果对不上（phase3 还依赖逐比特可复现做校验）。
所以这里只产出一份给浏览器看的副本。
"""

import argparse
import os
import shutil
import subprocess
import struct
import sys

# 浏览器能解的视频 fourcc。hvc1/hev1（HEVC）只有 Safari 和带硬解的 Chrome 能放，
# 不算安全档，所以不列进来，碰到一样转。
BROWSER_SAFE = {"avc1", "av01", "vp09", "vp08"}


def probe_codec(path):
    """扒 mp4 的 stsd box，返回视频轨的 fourcc（如 avc1 / mp4v）。非 mp4 返回 None。"""
    size = os.path.getsize(path)

    def walk(f, end):
        while f.tell() < end:
            pos = f.tell()
            hdr = f.read(8)
            if len(hdr) < 8:
                return None
            box, typ = struct.unpack(">I4s", hdr)
            typ = typ.decode("latin1", "replace")
            if box == 1:
                box = struct.unpack(">Q", f.read(8))[0]
            if box == 0:
                box = end - pos
            if box < 8:
                return None
            if typ == "stsd":
                f.seek(pos + 16)                     # box 头 8 + ver/flags 4 + entry_count 4
                return f.read(8)[4:8].decode("latin1", "replace")
            if typ in ("moov", "trak", "mdia", "minf", "stbl"):
                got = walk(f, pos + box)
                if got:
                    return got
            f.seek(pos + box)
        return None

    with open(path, "rb") as f:
        return walk(f, size)


def find_ffmpeg(explicit=None):
    """依次试：显式指定 -> FFMPEG_BINARY -> PATH -> imageio-ffmpeg 自带的那个。"""
    for cand in (explicit, os.environ.get("FFMPEG_BINARY"), shutil.which("ffmpeg")):
        if cand and os.path.exists(cand):
            return cand
    try:                                             # 装了 imageio-ffmpeg 就有现成的静态包
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def transcode(src, dst, ffmpeg, crf=16):
    """转 H.264 / yuv420p。crf 16 对已经压过一道的素材接近视觉无损。"""
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", src,
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p",                       # 4:2:0，缺了它 Safari 不放
        "-profile:v", "high", "-level", "4.0",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # H.264 要求宽高偶数
        "-movflags", "+faststart",                   # moov 提到文件头，边下边播
        "-an", dst,
    ]
    subprocess.run(cmd, check=True)


def ensure_h264(src, dst, ffmpeg=None, crf=16):
    """已经是浏览器能解的就直接复制，否则转码。返回 (是否转码过, 编码名)。

    转不了（没 ffmpeg）就抛 RuntimeError，交给调用方决定是降级还是报错。
    """
    codec = probe_codec(src)
    if codec in BROWSER_SAFE:
        shutil.copyfile(src, dst)
        return False, codec
    ff = find_ffmpeg(ffmpeg)
    if not ff:
        raise RuntimeError(
            f"{os.path.basename(src)} 是 {codec}，浏览器解不了，但找不到 ffmpeg 来转。"
            "装 imageio-ffmpeg，或用 --ffmpeg 指路径。")
    transcode(src, dst, ff, crf)
    return True, codec


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--outdir", default="videos/web")
    ap.add_argument("--ffmpeg", help="ffmpeg 可执行文件路径")
    ap.add_argument("--crf", type=int, default=16, help="越小越清晰、文件越大（默认 16）")
    ap.add_argument("--force", action="store_true", help="已经是 H.264 的也重转一遍")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    done = 0
    for src in args.inputs:
        codec = probe_codec(src)
        dst = os.path.join(args.outdir, os.path.basename(src))
        if os.path.abspath(dst) == os.path.abspath(src):
            print(f"  跳过 {src}：输出会覆盖原文件", file=sys.stderr)
            continue
        if codec in BROWSER_SAFE and not args.force:
            print(f"  {os.path.basename(src):22} {codec}  浏览器可放，不动")
            continue
        ff = find_ffmpeg(args.ffmpeg)
        if not ff:
            raise SystemExit("找不到 ffmpeg。pip install imageio-ffmpeg，或用 --ffmpeg 指路径")
        transcode(src, dst, ff, args.crf)
        a, b = os.path.getsize(src) / 2**20, os.path.getsize(dst) / 2**20
        print(f"  {os.path.basename(src):22} {codec} -> avc1  {a:.1f} -> {b:.1f} MB  {dst}")
        done += 1
    print(f"\n转了 {done} 个，原文件原样保留在原处。")


if __name__ == "__main__":
    main()
