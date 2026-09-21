"""把 outputs/ 里的结果打成一个可直接部署的静态站点，每个 case 一行并排对比。

    python tools/build_site.py --columns source outputs/p2_base -o docs
    python tools/build_site.py --columns outputs/p2_base outputs/p3_full
    python tools/build_site.py --columns source outputs/p2_base outputs/p3_full --zip

--columns 每一项是 `source`（videos/ 里的原视频）或一个实验目录，从左到右排。
想改栏目标题就写成 `Label=outputs/p2_base`，不写则按开关自动命名。

产物是纯静态的（index.html + 若干 mp4），GitHub Pages / Netlify 都能直接发。
prompt 和超参不写死在模板里，全部从每个 mp4 旁边的同名 .txt 读——那是跑的时候
落下来的真实配置，跟页面永远对得上。源视频路径从 configs/edits.yaml 取。
"""

import argparse
import html
import os
import shutil
import sys
import zipfile
from datetime import datetime
from difflib import SequenceMatcher

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml

from to_h264 import ensure_h264

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_sidecar(path):
    """读 mp4 旁边的 .txt 配置，返回 dict。格式是每行 `key: value`。"""
    meta = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if ": " not in line:
                continue
            k, v = line.rstrip("\n").split(": ", 1)
            meta[k.strip()] = v.strip()
    return meta


def scan(exp_dir):
    """扫一个实验目录，返回 {case_name: (mp4 路径, meta dict)}。"""
    out = {}
    if not os.path.isdir(exp_dir):
        raise SystemExit(f"实验目录不存在: {exp_dir}")
    for case in sorted(os.listdir(exp_dir)):
        d = os.path.join(exp_dir, case)
        if not os.path.isdir(d):
            continue
        mp4 = [f for f in sorted(os.listdir(d)) if f.endswith(".mp4")]
        if not mp4:
            continue
        p = os.path.join(d, mp4[0])
        txt = p[:-4] + ".txt"
        out[case] = (p, read_sidecar(txt) if os.path.exists(txt) else {})
    return out


def source_videos(dataset_yaml):
    """从 edits.yaml 取每个 case 的源视频路径和 prompt。"""
    with open(os.path.join(ROOT, dataset_yaml), encoding="utf-8") as f:
        items = yaml.safe_load(f)
    return {it["name"]: it for it in items}


def switches(meta):
    """把 use_sga / use_anc 翻成人看的模块名。"""
    on = [n for n, k in (("SGA", "use_sga"), ("ANC", "use_anc"))
          if meta.get(k, "False") == "True"]
    return " + ".join(on) if on else None


def diff_mark(src, tar):
    """把 target prompt 相对 source prompt 改动的部分用 <mark> 标出来。

    词级 diff。编辑既可能是句尾追加、也可能是中间插一句顺带把原来的尾巴删掉
    （swan 就是这种），所以用 difflib 逐块对，不能只做掐头去尾。
    """
    def norm(w):                                 # 比较时忽略标点大小写，不然
        return w.strip(".,;:!?\"'").lower()      # 句末的「.」和「,」会让整段失配

    a, b = src.split(), tar.split()
    ops = SequenceMatcher(None, [norm(w) for w in a], [norm(w) for w in b]).get_opcodes()
    out = []
    for tag, _, _, j1, j2 in ops:
        if tag == "delete":                      # 源里被删掉的词不在 target 里，无需呈现
            continue
        seg = html.escape(" ".join(b[j1:j2]))
        out.append(seg if tag == "equal" else f"<mark>{seg}</mark>")
    return " ".join(x for x in out if x)


def badges(meta):
    """从配置里挑几个真正影响结果的超参，做成角标。"""
    out = []
    src_g, tar_g = meta.get("src_guidance_scale"), meta.get("tar_guidance_scale")
    if src_g and tar_g:
        out.append(f"CFG {src_g} → {tar_g}")
    for key, fmt in (("tau", "τ = {}"), ("T_steps", "T = {}"), ("seed", "seed {}")):
        if meta.get(key):
            out.append(fmt.format(meta[key]))
    if meta.get("width") and meta.get("height"):
        out.append(f'{meta["width"]}×{meta["height"]}')
    return out


def make_columns(specs):
    """把 --columns 的每一项解析成一栏。`Label=spec` 可覆盖自动命名。"""
    cols = []
    for spec in specs:
        label = None
        if "=" in spec:
            label, spec = spec.split("=", 1)
        if spec == "source":
            cols.append(dict(kind="source", slug="source", spec=spec, items=None,
                             label=label or "Source",
                             sub="original input video"))
            continue
        items = scan(os.path.join(ROOT, spec))
        if not items:
            raise SystemExit(f"{spec} 里没有带 mp4 的 case 目录")
        meta = next(iter(items.values()))[1]          # 同一组实验的开关是一样的
        sw = switches(meta)
        cols.append(dict(kind="dyna" if sw else "base", slug=os.path.basename(spec.rstrip("/")),
                         spec=spec, items=items,
                         label=label or ("DynaEdit" if sw else "FlowEdit baseline"),
                         sub=sw or "plain FlowEdit · SGA off · ANC off"))
    return cols


def pane(col, rel):
    return f"""        <figure class="pane">
          <figcaption class="pane-label"><span class="dot dot-{col['kind']}"></span>{html.escape(col['label'])}<em>{html.escape(col['sub'])}</em></figcaption>
          <video src="{rel}#t=0.001" muted loop playsinline preload="metadata"></video>
        </figure>"""


def build_case(case, meta, cols, rels):
    src_p = meta.get("source_prompt", "")
    tar_p = meta.get("target_prompt", "")
    tag = " ".join(f'<span class="badge">{html.escape(b)}</span>' for b in badges(meta))
    panes = "\n".join(pane(c, rels[c["slug"]]) for c in cols)
    return f"""    <section class="case" id="{html.escape(case)}">
      <header class="case-head">
        <h2>{html.escape(case.replace("_", " "))}</h2>
        <div class="badges">{tag}</div>
      </header>
      <div class="prompts">
        <div class="prompt">
          <span class="tag">source prompt</span>
          <p>{html.escape(src_p)}</p>
        </div>
        <div class="prompt prompt-target">
          <span class="tag tag-target">target prompt</span>
          <p>{diff_mark(src_p, tar_p)}</p>
        </div>
      </div>
      <div class="panes">
{panes}
      </div>
      <div class="ctl">
        <button data-act="toggle">Play / Pause</button>
        <button data-act="restart">Restart</button>
        <button data-act="step" data-dir="-1" title="Step back one frame">◀</button>
        <button data-act="step" data-dir="1" title="Step forward one frame">▶</button>
        <input class="seek" type="range" min="0" max="1000" value="0" step="1" aria-label="Seek">
        <span class="time">0.00s</span>
        <button data-act="rate" data-rate="0.5">0.5×</button>
        <button data-act="rate" data-rate="1" class="on">1×</button>
      </div>
    </section>"""


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<meta name="description" content="__SUB__">
<style>
:root {
  --bg: #0d1117; --panel: #151b23; --line: #262d38;
  --fg: #e6edf3; --dim: #9aa6b2; --faint: #6b7684;
  --source: #58a6ff; --base: #f0883e; --dyna: #3fb950;
  --mark: rgba(63,185,80,.18); --mark-line: rgba(63,185,80,.55);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1440px; margin: 0 auto; padding: 0 24px 72px; }

header.top { padding: 56px 0 28px; border-bottom: 1px solid var(--line); margin-bottom: 8px; }
header.top h1 { margin: 0 0 10px; font-size: 30px; letter-spacing: -.02em; }
header.top p { margin: 0; color: var(--dim); max-width: 80ch; }
header.top p + p { margin-top: 8px; }
header.top code { background: var(--panel); padding: 1px 6px; border-radius: 5px;
  font-size: 13px; color: var(--fg); }
a { color: var(--source); }

.toolbar { position: sticky; top: 0; z-index: 20; display: flex; flex-wrap: wrap;
  gap: 10px; align-items: center; padding: 12px 0;
  background: rgba(13,17,23,.92); backdrop-filter: blur(8px);
  border-bottom: 1px solid var(--line); margin-bottom: 28px; }
.toolbar .spacer { flex: 1; }
.toolbar label { display: inline-flex; align-items: center; gap: 7px;
  color: var(--dim); font-size: 13px; cursor: pointer; user-select: none; }
.toolbar input[type=checkbox] { accent-color: var(--dyna); width: 15px; height: 15px; }

button {
  background: var(--panel); color: var(--fg); border: 1px solid var(--line);
  padding: 6px 13px; border-radius: 7px; font-size: 13px; cursor: pointer;
  font-family: inherit; transition: background .12s, border-color .12s;
}
button:hover { background: #1d242e; border-color: #38404d; }
button.on { border-color: var(--dyna); color: var(--dyna); }

.case { border: 1px solid var(--line); border-radius: 14px; background: var(--panel);
  padding: 22px; margin-bottom: 28px; }
.case-head { display: flex; flex-wrap: wrap; gap: 12px; align-items: baseline;
  margin-bottom: 16px; }
.case-head h2 { margin: 0; font-size: 20px; letter-spacing: -.01em; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; }
.badge { font-size: 11.5px; color: var(--dim); border: 1px solid var(--line);
  border-radius: 999px; padding: 2px 9px; background: #10161e;
  font-variant-numeric: tabular-nums; }

.prompts { display: grid; gap: 10px; margin-bottom: 18px; }
.prompt { display: grid; grid-template-columns: 124px 1fr; gap: 14px;
  align-items: start; background: #10161e; border: 1px solid var(--line);
  border-radius: 10px; padding: 11px 14px; }
.prompt p { margin: 0; font-size: 14px; }
.prompt-target p { color: var(--fg); }
.prompt:not(.prompt-target) p { color: var(--dim); }
.tag { font-size: 11px; letter-spacing: .06em; text-transform: uppercase;
  color: var(--faint); padding-top: 3px; }
.tag-target { color: var(--dyna); }
mark { background: var(--mark); color: var(--fg); border-bottom: 1px solid var(--mark-line);
  border-radius: 3px; padding: 0 2px; }

.panes { display: grid; gap: 14px;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }
.pane { margin: 0; }
.pane-label { display: flex; align-items: center; gap: 8px; font-size: 13px;
  font-weight: 600; margin-bottom: 8px; }
.pane-label em { font-style: normal; font-weight: 400; color: var(--faint);
  font-size: 12px; }
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.dot-source { background: var(--source); }
.dot-base { background: var(--base); }
.dot-dyna { background: var(--dyna); }
video { width: 100%; display: block; border-radius: 9px; background: #000;
  border: 1px solid var(--line); aspect-ratio: 832 / 480; }

.ctl { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 14px; }
.ctl .time { font-size: 12px; color: var(--faint); font-variant-numeric: tabular-nums;
  min-width: 52px; }
.seek { flex: 1; min-width: 160px; accent-color: var(--dyna); height: 4px; cursor: pointer; }

footer { border-top: 1px solid var(--line); margin-top: 40px; padding-top: 20px;
  color: var(--faint); font-size: 12.5px; }
footer code { color: var(--dim); }
@media (max-width: 640px) {
  .wrap { padding: 0 14px 48px; }
  .prompt { grid-template-columns: 1fr; gap: 4px; }
  header.top { padding-top: 34px; }
}
</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <h1>__TITLE__</h1>
  <p>__SUB__</p>
  <p>__INTRO__</p>
</header>

<div class="toolbar">
  <button id="all-play">Play all</button>
  <button id="all-pause">Pause all</button>
  <button id="all-restart">Restart all</button>
  <div class="spacer"></div>
  <label><input type="checkbox" id="opt-auto" checked> Autoplay when scrolled into view</label>
</div>

__CASES__

<footer>
  <p>__FOOTCOLS__</p>
  <p>Base model <code>__MODEL__</code>. Generated by <code>tools/build_site.py</code> on __DATE__;
     prompts and hyper-parameters are read from the <code>.txt</code> run record written next to each mp4.</p>
</footer>
</div>

<script>
const cases = [...document.querySelectorAll(".case")];

function vids(card)   { return [...card.querySelectorAll("video")]; }
function leader(card) { return vids(card)[0]; }
function play(card)   { vids(card).forEach(v => v.play().catch(() => {})); }
function pause(card)  { vids(card).forEach(v => v.pause()); }
function seek(card, t){ vids(card).forEach(v => { v.currentTime = t; }); }

cases.forEach(card => {
  const ctl  = card.querySelector(".ctl");
  const bar  = card.querySelector(".seek");
  const time = card.querySelector(".time");

  ctl.addEventListener("click", e => {
    const b = e.target.closest("button");
    if (!b) return;
    const act = b.dataset.act;
    if (act === "toggle")  { leader(card).paused ? play(card) : pause(card); }
    if (act === "restart") { seek(card, 0); play(card); }
    if (act === "step") {
      pause(card);
      // Wan renders at 16fps, so one frame is 1/16 s
      seek(card, Math.max(0, leader(card).currentTime + (+b.dataset.dir) / 16));
    }
    if (act === "rate") {
      const r = +b.dataset.rate;
      vids(card).forEach(v => { v.playbackRate = r; });
      ctl.querySelectorAll('[data-act=rate]').forEach(x => x.classList.toggle("on", x === b));
    }
  });

  bar.addEventListener("input", () => {
    const l = leader(card);
    if (l.duration) seek(card, bar.value / 1000 * l.duration);
  });

  card.addEventListener("timeupdate", e => {
    const l = leader(card);
    if (e.target !== l || !l.duration) return;
    bar.value = l.currentTime / l.duration * 1000;
    time.textContent = l.currentTime.toFixed(2) + "s";
    // The streams decode independently and drift apart; pull followers back past 80ms
    vids(card).slice(1).forEach(v => {
      if (Math.abs(v.currentTime - l.currentTime) > 0.08) v.currentTime = l.currentTime;
    });
  }, true);
});

document.getElementById("all-play").onclick    = () => cases.forEach(play);
document.getElementById("all-pause").onclick   = () => cases.forEach(pause);
document.getElementById("all-restart").onclick = () => cases.forEach(c => { seek(c, 0); play(c); });

const optAuto = document.getElementById("opt-auto");
const io = new IntersectionObserver(entries => {
  if (!optAuto.checked) return;
  entries.forEach(en => en.isIntersecting ? play(en.target) : pause(en.target));
}, { threshold: 0.35 });
function watch() { cases.forEach(c => optAuto.checked ? io.observe(c) : io.unobserve(c)); }
optAuto.onchange = () => { watch(); if (!optAuto.checked) cases.forEach(pause); };
watch();
</script>
</body>
</html>
"""


def describe(col):
    return (f'<strong style="color:var(--{col["kind"]})">{html.escape(col["label"])}</strong>'
            f' ({html.escape(col["sub"])})')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--columns", nargs="+", default=["source", "outputs/p2_base"],
                    help="从左到右每一栏：`source` 或实验目录，可写 `Label=路径`")
    ap.add_argument("--dataset-yaml", default="configs/edits.yaml")
    ap.add_argument("-o", "--outdir", default="docs", help="GitHub Pages 默认认 docs/")
    ap.add_argument("--title", default="DynaEdit · Video Editing Results")
    ap.add_argument("--cases", nargs="*", help="只出这几个 case（默认全部，按给定顺序）")
    ap.add_argument("--ffmpeg", help="ffmpeg 路径。只有碰到浏览器解不了的编码才用得上")
    ap.add_argument("--zip", action="store_true",
                    help="额外打一个 zip。index.html 在压缩包根目录，可直接拖到 netlify.com/drop")
    args = ap.parse_args()

    cols = make_columns(args.columns)
    exp_cols = [c for c in cols if c["items"] is not None]
    if not exp_cols:
        raise SystemExit("至少要有一栏是实验目录，不然没有 prompt 和超参可显示")
    dataset = source_videos(args.dataset_yaml)

    names = args.cases or sorted(set.intersection(*(set(c["items"]) for c in exp_cols)))
    for c in exp_cols:
        miss = [n for n in names if n not in c["items"]]
        if miss:
            raise SystemExit(f"{c['spec']} 里缺 case: {', '.join(miss)}")
    if any(c["kind"] == "source" for c in cols):
        miss = [n for n in names if not dataset.get(n, {}).get("input_video")]
        if miss:
            raise SystemExit(f"{args.dataset_yaml} 里这些 case 没有 input_video: {', '.join(miss)}")
    if not names:
        raise SystemExit("各栏没有共同的 case")

    out = os.path.join(ROOT, args.outdir)
    vdir = os.path.join(out, "videos")
    shutil.rmtree(vdir, ignore_errors=True)
    os.makedirs(vdir, exist_ok=True)

    blocks = []
    for case in names:
        rels = {}
        for c in cols:
            src = (os.path.join(ROOT, dataset[case]["input_video"])
                   if c["kind"] == "source" else c["items"][case][0])
            name = f'{case}__{c["slug"]}.mp4'
            try:
                # 浏览器只解 H.264 / VP9 / AV1。videos/ 里曾经有 mp4v 的片子，
                # 真碰上了就地转一份进站点，原文件不动。
                converted, codec = ensure_h264(src, os.path.join(vdir, name), args.ffmpeg)
            except RuntimeError as e:
                raise SystemExit(str(e))
            if converted:
                print(f"  {case}/{c['slug']}: {codec} -> avc1（原文件未改动）")
            rels[c["slug"]] = "videos/" + name
        meta = exp_cols[0]["items"][case][1]
        blocks.append(build_case(case, meta, cols, rels))
        print(f"  {case}: " + " | ".join(c["label"] for c in cols))

    intro = "Left to right: " + ", ".join(describe(c) for c in cols) + "."
    if len(exp_cols) >= 2:
        intro += (" All generated columns use the same seed, step count and guidance scales, "
                  "so any difference between them comes only from the modules named above.")
    intro += (" In each target prompt, the <mark>highlighted</mark> span is what was added"
              " relative to the source prompt, which describes the original clip.")
    sub = (f"{len(names)} edits on Wan2.1-I2V, compared side by side. Training-free:"
           " no fine-tuning, no inversion, no changes to attention or the transformer forward.")
    foot = " · ".join(
        f'{html.escape(c["label"])}: <code>'
        + html.escape(os.path.dirname(dataset[names[0]]["input_video"]) + "/"
                      if c["kind"] == "source" else c["spec"]) + "</code>"
        for c in cols)

    page = (PAGE.replace("__TITLE__", html.escape(args.title))
                .replace("__SUB__", html.escape(sub))
                .replace("__INTRO__", intro)
                .replace("__CASES__", "\n".join(blocks))
                .replace("__FOOTCOLS__", foot)
                .replace("__MODEL__", html.escape(exp_cols[0]["items"][names[0]][1]
                                                  .get("model_id", "—")))
                .replace("__DATE__", datetime.now().strftime("%Y-%m-%d")))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    open(os.path.join(out, ".nojekyll"), "w").close()   # GitHub Pages 别拿 jekyll 处理

    mb = sum(os.path.getsize(os.path.join(vdir, f)) for f in os.listdir(vdir)) / 2**20
    rel = os.path.relpath(out, ROOT)
    print(f"\n生成 {rel}/index.html（{len(names)} 组 × {len(cols)} 栏，视频共 {mb:.1f} MB）")
    print(f"本地预览: cd {rel} && python3 -m http.server 8000")

    if args.zip:
        zpath = out.rstrip("/") + ".zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for d, _, files in os.walk(out):
                for f in files:
                    full = os.path.join(d, f)
                    z.write(full, os.path.relpath(full, out))   # index.html 必须在包根
        print(f"打包: {os.path.relpath(zpath, ROOT)}（{os.path.getsize(zpath) / 2**20:.1f} MB）")


if __name__ == "__main__":
    main()
