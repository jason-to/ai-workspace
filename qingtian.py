"""把《晴天》钢琴谱（前 18 小节）转成音频。G 大调 4/4，Andante。
记谱: 每个音符为 "音名/时值(拍)"，和弦用 + 连接，r 为休止。
"""
import numpy as np, wave, sys

BPM = 72
SR = 44100

def m(s):  # 解析一个小节
    out = []
    for tok in s.split():
        p, d = tok.split('/')
        out.append((p, float(d)))
    tot = sum(d for _, d in out)
    assert abs(tot - 4) < 1e-6, (s, tot)
    return out

# ---------- 右手 ----------
intro_a = "r/.5 G3/.5 D4/.5 G3/.5 r/1 D4/.5 G3/.5"
intro_b = "r/1 D4/.5 G3/.5 r/.5 D4/.5 r/.5 D4/.5"
def chord_bar(c1, c2):
    # 四分 + 附点八分 + 十六分(连) + 十六分 + 八分 + 附点八分 + 十六分
    return (f"{c1}/1 {c1}/.75 {c1}/.5 {c1}/.25 {c2}/.5 {c2}/.75 {c2}/.25")
EM, GC = "E4+G4+B4", "D4+G4+B4"
verse_a = "r/.5 D5/.5 D5/.5 G4/.5 G4/1 A4/.5 B4/.5"
RH = [
    intro_a, intro_b, intro_a, intro_b,
    chord_bar(EM, EM), chord_bar(GC, GC), chord_bar(EM, EM), chord_bar(GC, GC),
    verse_a,                                                   # 故事的小黄花
    "r/.5 D5/.5 D5/.25 G4/.25 G4/.5 G4/.5 A4/.25 B4/.25 A4/.25 G4/.25 D4/.5",  # 从出生那年就飘着
    verse_a,                                                   # 童年的荡秋千
    "r/.5 B4/1 A4/.25 B4/.25 C5/.25 B4/.25 A4/.25 C5/.25 B4/.25 A4/.25 G4/.5",  # 随记忆一直晃到现在
    "D4/.5 G4/.5 G4/.5 B4/.5 C5/.5 B4/.5 A4/.5 G4/.25 A4/.25",                # re sol sol si do si la sol la
    "B4/.5 B4/.5 B4/.5 B4/.5 A4/.25 B4/.25 A4/.5 G4/1",                       # si si si si la si la sol
    "D4/.5 G4/.5 G4/.5 B4/.5 C5/.5 B4/.5 A4/.5 G4/.25 A4/.25",                # 吹着前奏望着天空我
    "B4/.5 B4/.5 B4/.5 B4/.5 A4/.25 B4/.25 A4/.5 G4/.5 F#4/.5",               # 想起花瓣试着掉落 为
    "F#4/.25 G4/.25 G4/.25 G4/.25 F#4/.75 G4/.75 G4/.25 G4/.25 G4/.25 F#4/.5 G4/.25",  # 你翘课的那一天 花落的那一
    "r/.25 G4/.25 G4/.25 G4/.25 F#4/.75 G4/.5 G4/.25 G4/.25 G4/.25 D5/.25 C5/.5 D5/.25",  # 教室的那一间 我怎么看不见
]

# ---------- 左手 ----------
lh_intro_a = "E3/2 C3/.5 D3/.25 E3/1.25"
lh_intro_b = "G2/.5 D3/1.5 G2/1 F#2/1"
lh_em_c = "E2/.5 B2/.5 E3/.5 B2/.5 C3/.5 G3/.5 C4/.5 G3/.5"
lh_g_d = "G2/.5 D3/.5 G3/.5 D3/.5 G2/.5 D3/.5 F#2/.5 D3/.5"
lh_v = "E3/.5 G3/.5 D4/.5 G3/.5 C3/.5 G3/.5 D4/.5 G3/.5"
LH = [lh_intro_a, lh_intro_b, lh_intro_a, lh_intro_b,
      lh_em_c, lh_g_d, lh_em_c, lh_g_d] + [lh_v, lh_g_d] * 5

NOTE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
def freq(n):
    semi = NOTE[n[0]] + (1 if '#' in n else 0) - (1 if n[1] == 'b' else 0)
    octv = int(n[-1])
    midi = 12 * (octv + 1) + semi
    return 440 * 2 ** ((midi - 69) / 12)

def tone(f, dur_s, vel):
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    # 简单"钢琴"音色：多次谐波 + 指数衰减 + 轻微失谐
    sig = np.zeros(n)
    for k, a in enumerate([1, .5, .3, .18, .1, .06], 1):
        decay = np.exp(-t * (1.2 + 0.9 * k) * (f / 440) ** 0.35)
        sig += a * decay * np.sin(2 * np.pi * f * k * t * (1 + 0.0004 * k))
    attack = np.minimum(1, t / 0.004)
    release = np.minimum(1, (n - np.arange(n)) / (0.08 * SR))
    return vel * sig * attack * release

def render(bars, vel, extra):
    beat = 60 / BPM
    total = len(bars) * 4 * beat + 4
    buf = np.zeros(int(SR * total))
    pos = 0.0
    for bar in bars:
        for p, d in m(bar):
            if p != 'r':
                start = int(SR * pos * beat)
                for n in p.split('+'):
                    w = tone(freq(n), d * beat + extra, vel)
                    buf[start:start + len(w)] += w
            pos += d
    return buf

rh = render(RH, 0.30, 0.35)
lh = render(LH, 0.20, 1.0)   # 左手模拟踏板，余音更长
mix = rh + lh
mix /= np.max(np.abs(mix)) * 1.1
pcm = (mix * 32767).astype(np.int16)
out = sys.argv[1]
with wave.open(out, 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print("wrote", out, len(pcm) / SR, "s")
