"""把 outputs/ 里的 baseline / DynaEdit 结果打成一个可直接部署的静态站点。

    python tools/build_site.py                          # 默认 p2_base vs p3_full
    python tools/build_site.py --baseline outputs/p4_fe_hi --dynaedit outputs/p4_g4
    cd site && python3 -m http.server 8000              # 本地预览
    python tools/build_site.py --zip                    # 顺手打包，拖到 netlify.com/drop 即上线

产物 site/ 是纯静态的（只有 index.html + 若干 mp4），丢到 GitHub Pages /
Netlify / 任何能发静态文件的地方都能跑，不需要后端。

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

import yaml

import to_h264

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB_COPIES = os.path.join(ROOT, "videos", "web")   # to_h264.py 转好的浏览器版


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


def copy_playable(src, dst):
    """把视频放进站点，顺手保证编码是浏览器能解的。

    videos/ 里 bear / blackswan / boat / snowboard 是 mp4v（MPEG-4 Part 2）编的，
    浏览器一律解不了，页面上那一栏就是「加载出错」。原文件是采样管线的输入不能动
    （见 tools/to_h264.py），所以优先拿 videos/web/ 里转好的那份，没有就现转，
    实在转不了才原样拷贝并吼一声——站点照样出得来，只是那一栏放不了。
    """
    web = os.path.join(WEB_COPIES, os.path.basename(src))
    if os.path.exists(web) and to_h264.probe_codec(web) in to_h264.BROWSER_SAFE:
        shutil.copyfile(web, dst)
        return
    try:
        to_h264.ensure_h264(src, dst)
    except RuntimeError as e:
        shutil.copyfile(src, dst)
        print(f"  ! {e}\n    先原样放进去了，页面上这一栏会加载失败。", file=sys.stderr)


def source_videos(dataset_yaml):
    """从 edits.yaml 取每个 case 的源视频路径。"""
    with open(os.path.join(ROOT, dataset_yaml), encoding="utf-8") as f:
        items = yaml.safe_load(f)
    return {it["name"]: it.get("input_video") for it in items}


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


def switches(meta):
    """把 use_sga / use_anc 翻成人看的模块名。"""
    on = [n for n, k in (("SGA", "use_sga"), ("ANC", "use_anc"))
          if meta.get(k, "False") == "True"]
    return " + ".join(on) if on else "both switches off"


def pane(kind, label, sub, rel_src):
    return f"""        <figure class="pane pane-{kind}">
          <figcaption class="pane-label"><span class="dot dot-{kind}"></span>{html.escape(label)}<em>{html.escape(sub)}</em></figcaption>
          <video data-role="{kind}" src="{rel_src}#t=0.001" muted loop playsinline preload="metadata"></video>
        </figure>"""


def build_case(case, meta, rel, show_src):
    src_p = meta.get("source_prompt", "")
    tar_p = meta.get("target_prompt", "")
    tag = " ".join(f'<span class="badge">{html.escape(b)}</span>' for b in badges(meta))
    panes = []
    if show_src:
        panes.append(pane("source", "Source", "input video", rel["source"]))
    panes.append(pane("base", "Baseline", "FlowEdit · both switches off", rel["base"]))
    panes.append(pane("dyna", "DynaEdit", switches(meta), rel["dyna"]))
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
{chr(10).join(panes)}
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
  --base: #f0883e; --dyna: #3fb950; --source: #58a6ff;
  --mark: rgba(63,185,80,.18); --mark-line: rgba(63,185,80,.55);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
        "Hiragino Sans GB", "Microsoft YaHei", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1440px; margin: 0 auto; padding: 0 24px 72px; }

header.top { padding: 56px 0 28px; border-bottom: 1px solid var(--line); margin-bottom: 8px; }
header.top h1 { margin: 0 0 10px; font-size: 30px; letter-spacing: -.02em; }
header.top p { margin: 0; color: var(--dim); max-width: 78ch; }
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
.pane-source { display: none; }
body.show-source .pane-source { display: block; }
.pane-label { display: flex; align-items: center; gap: 8px; font-size: 13px;
  font-weight: 600; margin-bottom: 8px; }
.pane-label em { font-style: normal; font-weight: 400; color: var(--faint);
  font-size: 12px; }
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.dot-base { background: var(--base); } .dot-dyna { background: var(--dyna); }
.dot-source { background: var(--source); }
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
  <p>Left is the <strong style="color:var(--base)">Baseline</strong> (video FlowEdit,
     <code>use_sga=false, use_anc=false</code>); right is
     <strong style="color:var(--dyna)">DynaEdit</strong> — same seed, same CFG, same step count,
     with only SGA and ANC switched on. Every other hyper-parameter is identical on both sides,
     so any difference comes from those two modules alone. In each target prompt, the
     <mark>highlighted</mark> span is what was added relative to the source prompt.</p>
</header>

<div class="toolbar">
  <button id="all-play">Play all</button>
  <button id="all-pause">Pause all</button>
  <button id="all-restart">Restart all</button>
  <div class="spacer"></div>
  <label><input type="checkbox" id="opt-source"> Show source video</label>
  <label><input type="checkbox" id="opt-auto" checked> Autoplay when scrolled into view</label>
</div>

__CASES__

<footer>
  <p>Baseline: <code>__BASE_DIR__</code> · DynaEdit: <code>__DYNA_DIR__</code> · base model <code>__MODEL__</code></p>
  <p>Generated by <code>tools/build_site.py</code> on __DATE__. Prompts and hyper-parameters are read
     from the <code>.txt</code> run record written next to each mp4.</p>
</footer>
</div>

<script>
const cases = [...document.querySelectorAll(".case")];

function vids(card) {
  // Only the visible panes count: an unchecked source video stays out of the sync group
  // instead of quietly playing in the background.
  return [...card.querySelectorAll("video")].filter(v => v.offsetParent !== null);
}
function leader(card) { return vids(card)[0]; }

function play(card)  { vids(card).forEach(v => v.play().catch(() => {})); }
function pause(card) { [...card.querySelectorAll("video")].forEach(v => v.pause()); }
function seek(card, t) {
  [...card.querySelectorAll("video")].forEach(v => { v.currentTime = t; });
}

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
      const l = leader(card);
      // Wan renders at 16fps, so one frame is 1/16 s
      seek(card, Math.max(0, l.currentTime + (+b.dataset.dir) / 16));
    }
    if (act === "rate") {
      const r = +b.dataset.rate;
      [...card.querySelectorAll("video")].forEach(v => { v.playbackRate = r; });
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

const optSource = document.getElementById("opt-source");
optSource.onchange = () => {
  document.body.classList.toggle("show-source", optSource.checked);
  // A source video that just appeared has to catch up, or it shows up black
  cases.forEach(c => { const l = leader(c); seek(c, l.currentTime); if (!l.paused) play(c); });
};

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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", default="outputs/p2_base", help="baseline 实验目录")
    ap.add_argument("--dynaedit", default="outputs/p3_full", help="DynaEdit 实验目录")
    ap.add_argument("--dataset-yaml", default="configs/edits.yaml", help="取源视频路径用")
    ap.add_argument("-o", "--outdir", default="site")
    ap.add_argument("--title", default="DynaEdit · Video Editing Results")
    ap.add_argument("--cases", nargs="*", help="只出这几个 case（默认全部，按给定顺序）")
    ap.add_argument("--zip", action="store_true",
                    help="额外打一个 zip。index.html 在压缩包根目录，可直接拖到 netlify.com/drop")
    args = ap.parse_args()

    base = scan(os.path.join(ROOT, args.baseline))
    dyna = scan(os.path.join(ROOT, args.dynaedit))
    srcs = source_videos(args.dataset_yaml)

    names = args.cases or [c for c in dyna if c in base]
    missing = [c for c in names if c not in base or c not in dyna]
    if missing:
        raise SystemExit(f"两边都得有才能对比，缺: {', '.join(missing)}")
    if not names:
        raise SystemExit("两个目录没有共同的 case")

    out = os.path.join(ROOT, args.outdir)
    vdir = os.path.join(out, "videos")
    shutil.rmtree(vdir, ignore_errors=True)
    os.makedirs(vdir, exist_ok=True)

    blocks, model, have_src = [], "", True
    for case in names:
        rel = {}
        for kind, src in (("base", base[case][0]), ("dyna", dyna[case][0])):
            name = f"{case}__{kind}.mp4"
            copy_playable(src, os.path.join(vdir, name))
            rel[kind] = "videos/" + name
        sv = srcs.get(case)
        if sv and os.path.exists(os.path.join(ROOT, sv)):
            name = f"{case}__source.mp4"
            copy_playable(os.path.join(ROOT, sv), os.path.join(vdir, name))
            rel["source"] = "videos/" + name
        else:
            have_src = False
            print(f"  ! {case} 没找到源视频，跳过源视频那一栏", file=sys.stderr)
        meta = dyna[case][1]
        model = model or meta.get("model_id", "")
        blocks.append(build_case(case, meta, rel, "source" in rel))
        print(f"  {case}: baseline {switches(base[case][1])} | dynaedit {switches(meta)}")

    sub = (f"{len(names)} edits, compared side by side. The baseline is video FlowEdit; "
           "the right-hand column adds SGA (similarity-guided aggregation) and "
           "ANC (annealed noise correlation).")
    page = (PAGE.replace("__TITLE__", html.escape(args.title))
                .replace("__SUB__", html.escape(sub))
                .replace("__CASES__", "\n".join(blocks))
                .replace("__BASE_DIR__", html.escape(args.baseline))
                .replace("__DYNA_DIR__", html.escape(args.dynaedit))
                .replace("__MODEL__", html.escape(model or "—"))
                .replace("__DATE__", datetime.now().strftime("%Y-%m-%d %H:%M")))
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    open(os.path.join(out, ".nojekyll"), "w").close()   # GitHub Pages 别拿 jekyll 处理

    mb = sum(os.path.getsize(os.path.join(vdir, f)) for f in os.listdir(vdir)) / 2**20
    print(f"\n生成 {os.path.relpath(out, ROOT)}/index.html"
          f"（{len(names)} 组，{'含' if have_src else '部分缺'}源视频，视频共 {mb:.1f} MB）")
    print(f"本地预览: cd {os.path.relpath(out, ROOT)} && python3 -m http.server 8000")

    if args.zip:
        zpath = out.rstrip("/") + ".zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for d, _, files in os.walk(out):
                for f in files:
                    full = os.path.join(d, f)
                    z.write(full, os.path.relpath(full, out))   # index.html 必须在包根
        print(f"打包: {os.path.relpath(zpath, ROOT)}"
              f"（{os.path.getsize(zpath) / 2**20:.1f} MB）拖到 https://app.netlify.com/drop 即上线")


if __name__ == "__main__":
    main()
