"""用 Gemini 对编辑结果打分，prompt 照搬论文附录 F。

    export GEMINI_API_KEY=...
    python eval/vlm_eval.py --list-models          # 先看 key 能用哪些模型
    python eval/vlm_eval.py --outputs outputs --exps p4_g1,p4_g2,p4_g3,p4_g4,p4_fe_lo,p4_fe_hi
    python eval/vlm_eval.py --report               # 只读缓存出表，不调 API

三个维度各 1-5 分（论文 6.6）：
  content_preservation  源保真   —— 传 源视频 + 编辑视频 + 两条 prompt
  text_adherence        文本符合 —— 传 编辑视频 + target prompt
  visual_quality        视觉质量 —— 只传 编辑视频

免费层配额有限，所以：结果落盘缓存（同一 (视频,维度,模型) 只算一次）、
429/5xx 指数退避、中断后可断点续跑。
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import DIMENSIONS

CACHE = Path("eval/scores_cache.json")
DEFAULT_MODEL = "gemini-3-pro-preview"


def load_cache():
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save_cache(c):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(c, indent=2, ensure_ascii=False))


def file_key(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def upload(client, path, uploaded):
    """上传并等待处理完成。同一文件在一次运行内只上传一次。"""
    if path in uploaded:
        return uploaded[path]
    f = client.files.upload(file=path)
    for _ in range(120):
        if f.state.name == "ACTIVE":
            break
        if f.state.name == "FAILED":
            raise RuntimeError(f"上传失败: {path}")
        time.sleep(2)
        f = client.files.get(name=f.name)
    else:
        raise RuntimeError(f"上传超时: {path}")
    uploaded[path] = f
    return f


def ask(client, model, parts, retries=6):
    """带指数退避的调用。免费层很容易撞 429。"""
    delay = 10
    for attempt in range(retries):
        try:
            r = client.models.generate_content(model=model, contents=parts)
            txt = (r.text or "").strip()
            m = re.search(r"[1-5]", txt)
            if not m:
                raise ValueError(f"没解析出 1-5 的分数，模型返回: {txt[:120]!r}")
            return int(m.group()), txt
        except Exception as e:
            msg = str(e)
            transient = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "500", "503", "UNAVAILABLE", "deadline"))
            if attempt == retries - 1 or not transient:
                raise
            print(f"    重试 {attempt + 1}/{retries}（{delay}s）: {msg[:90]}", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)


def collect(outputs_dir, exps, edits_yaml, dedup=True):
    """收集待评任务。

    dedup: 按 (item, 文件名 tag) 去重。tag 编码了全部被扫维度
    （sga/anc/cfg/tau/T_steps/negative_prompt/seed），所以生效配置相同的运行
    tag 必然相同。不去重的话，同一配置的重复运行会在 VLM 统计里被当成两个
    独立样本，等于给某些格子偷偷加权。
    """
    edits = {e["name"]: e for e in yaml.safe_load(open(edits_yaml))}
    tasks, seen = [], {}
    for exp in exps:
        for item_dir in sorted(Path(outputs_dir, exp).glob("*")):
            if not item_dir.is_dir() or item_dir.name not in edits:
                continue
            for mp4 in sorted(item_dir.glob("*.mp4")):
                key = (item_dir.name, mp4.stem)
                if dedup and key in seen:
                    print(f"  [去重] {exp}/{item_dir.name} 与 {seen[key]} 生效配置相同，跳过")
                    continue
                seen[key] = f"{exp}/{item_dir.name}"
                e = edits[item_dir.name]
                tasks.append(dict(exp=exp, item=item_dir.name, video=str(mp4),
                                  source=e["input_video"],
                                  source_prompt=e["source_prompt"].strip(),
                                  target_prompt=e["target_prompt"].strip()))
    return tasks


def report(cache, model):
    rows = {}
    for k, v in cache.items():
        if v.get("model") != model:
            continue
        rows.setdefault((v["exp"], v["item"]), {})[v["dim"]] = v["score"]
    if not rows:
        print("缓存里没有该模型的结果。")
        return
    dims = ["content_preservation", "text_adherence", "visual_quality"]
    by_exp = {}
    print(f"\n{'实验':<12}{'视频':<18}" + "".join(f"{d[:10]:>13}" for d in dims) + f"{'均值':>8}")
    for (exp, item), d in sorted(rows.items()):
        vals = [d.get(x) for x in dims]
        if None in vals:
            print(f"{exp:<12}{item:<18}" + "".join(f"{v if v is not None else '-':>13}" for v in vals) + f"{'(不全)':>8}")
            continue
        by_exp.setdefault(exp, []).append(vals)
        print(f"{exp:<12}{item:<18}" + "".join(f"{v:>13}" for v in vals) + f"{sum(vals) / 3:>8.2f}")
    print(f"\n{'实验均值':<30}" + "".join(f"{d[:10]:>13}" for d in dims) + f"{'总均值':>8}")
    for exp, vs in sorted(by_exp.items()):
        m = [sum(x[i] for x in vs) / len(vs) for i in range(3)]
        print(f"{exp:<30}" + "".join(f"{x:>13.2f}" for x in m) + f"{sum(m) / 3:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--exps", default="p4_g1,p4_g2,p4_g3,p4_g4,p4_fe_lo,p4_fe_hi")
    ap.add_argument("--edits", default="configs/edits.yaml")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--report", action="store_true", help="只读缓存出表，不调 API")
    ap.add_argument("--sleep", type=float, default=4.0, help="每次调用后的间隔，免费层防限流")
    ap.add_argument("--no-dedup", action="store_true",
                    help="不按生效配置去重（默认去重，避免重复运行在统计里被加权）")
    args = ap.parse_args()

    cache = load_cache()
    if args.report:
        report(cache, args.model)
        return

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("未找到 GEMINI_API_KEY（或 GOOGLE_API_KEY）。\n"
                 "  export GEMINI_API_KEY=...\n"
                 "然后重跑。只想看已缓存的结果用 --report。")

    from google import genai
    client = genai.Client(api_key=key)

    if args.list_models:
        for m in client.models.list():
            acts = getattr(m, "supported_actions", None) or []
            if not acts or "generateContent" in acts:
                print(f"  {m.name}")
        return

    tasks = collect(args.outputs, args.exps.split(","), args.edits, dedup=not args.no_dedup)
    print(f"待评视频 {len(tasks)} 条 × 3 维度 = {len(tasks) * 3} 次调用，模型 {args.model}")
    uploaded, done, skipped = {}, 0, 0

    for t in tasks:
        vkey = file_key(t["video"])
        for dim, tmpl in DIMENSIONS.items():
            ck = f"{vkey}|{dim}|{args.model}"
            if ck in cache:
                skipped += 1
                continue
            instr = tmpl.format(source_prompt=t["source_prompt"], target_prompt=t["target_prompt"])
            if dim == "content_preservation":
                parts = ["SOURCE_VIDEO:", upload(client, t["source"], uploaded),
                         "TARGET_VIDEO:", upload(client, t["video"], uploaded), instr]
            else:
                parts = ["VIDEO:", upload(client, t["video"], uploaded), instr]
            print(f"  [{t['exp']}/{t['item']}] {dim} ...", end=" ", flush=True)
            score, raw = ask(client, args.model, parts)
            print(score, flush=True)
            cache[ck] = dict(exp=t["exp"], item=t["item"], dim=dim, score=score,
                             model=args.model, video=t["video"], raw=raw[:200])
            save_cache(cache)
            done += 1
            time.sleep(args.sleep)

    print(f"\n新评 {done} 次，命中缓存 {skipped} 次。")
    report(cache, args.model)


if __name__ == "__main__":
    main()
