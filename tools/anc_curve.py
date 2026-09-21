"""从运行日志里抽出 ANC 诊断，按 step 对齐两组（ANC 开 / 关），复现论文 Fig.11。

    python tools/anc_curve.py <ANC开的日志> <ANC关的日志>
"""
import re, sys, math

PAT = re.compile(r"\[corr\] step=(\d+) t=([\d.]+) a_t=([\d.]+) cos_V=([-+\d.naN]+) cos_w=([-+\d.naN]+)")


def parse(path):
    out = {}
    for line in open(path, errors="ignore"):
        for m in PAT.finditer(line):
            s, t, a, cv, cw = m.groups()
            f = lambda x: float("nan") if "nan" in x.lower() else float(x)
            out[int(s)] = (float(t), float(a), f(cv), f(cw))
    return out


on, off = parse(sys.argv[1]), parse(sys.argv[2])
steps = sorted(set(on) & set(off))
print(f"{'step':>4} {'t':>7} {'a_t':>6} {'√a_t':>7} | {'cos_w 开':>9} {'cos_w 关':>9} | {'cos_V 开':>9} {'cos_V 关':>9}")
dv = []
for s in steps:
    t, a, cv1, cw1 = on[s]
    _, _, cv0, cw0 = off[s]
    if math.isnan(cv1):
        continue
    dv.append(cv1 - cv0)
    if s % 3 == 0 or s == steps[-1]:
        print(f"{s:>4} {t:>7.4f} {a:>6.3f} {math.sqrt(a):>7.4f} | {cw1:>9.4f} {cw0:>9.4f} | {cv1:>9.4f} {cv0:>9.4f}")

err = [abs(on[s][3] - math.sqrt(on[s][1])) for s in steps if not math.isnan(on[s][3])]
print()
print(f"cos_w(ANC开) 与 √a_t 的最大偏差: {max(err):.5f}   ← Eq.7 的闭式校验")
off_w = [abs(off[s][3]) for s in steps if not math.isnan(off[s][3])]
print(f"cos_w(ANC关) 的最大绝对值:       {max(off_w):.5f}   ← 应≈0（跨步 i.i.d.）")
n = len(dv)
print(f"cos_V 差值(开-关) 均值: {sum(dv)/n:+.4f}  前半 {sum(dv[:n//2])/(n//2):+.4f}  后半 {sum(dv[n//2:])/(n-n//2):+.4f}")
