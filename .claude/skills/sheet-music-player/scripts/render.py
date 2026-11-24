#!/usr/bin/env python3
"""把文本乐谱（.score）渲染成 MP3（无 ffmpeg 时为 WAV）。

用法:
    python3 render.py song.score -o out/song        # 生成 out/song.mp3（无 ffmpeg 时为 .wav）
    python3 render.py song.score --midi             # 额外输出 .mid（仅在用户要求时使用）
    python3 render.py song.score --check            # 只检查每小节时值，不渲染
    python3 render.py song.score --bars 5-12        # 只渲染第 5~12 小节
    python3 render.py song.score --tempo 90         # 覆盖速度

格式说明见同目录上层的 SKILL.md。只依赖 numpy；有 ffmpeg 时输出 mp3。
"""
import argparse, re, shutil, struct, subprocess, sys, wave
from fractions import Fraction
from pathlib import Path

import numpy as np

SR = 44100
LETTER = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
SHARP_ORDER, FLAT_ORDER = "FCGDAEB", "BEADGCF"
KEY_SHARPS = {'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
              'Am': 0, 'Em': 1, 'Bm': 2, 'F#m': 3, 'C#m': 4, 'G#m': 5, 'D#m': 6, 'A#m': 7}
KEY_FLATS = {'F': 1, 'Bb': 2, 'Eb': 3, 'Ab': 4, 'Db': 5, 'Gb': 6, 'Cb': 7,
             'Dm': 1, 'Gm': 2, 'Cm': 3, 'Fm': 4, 'Bbm': 5, 'Ebm': 6, 'Abm': 7}
NOTE_RE = re.compile(r'^([A-Ga-g])(##|bb|#|b|n)?(-?\d)$')


def key_signature(key):
    if key in KEY_SHARPS:
        return {c: 1 for c in SHARP_ORDER[:KEY_SHARPS[key]]}
    if key in KEY_FLATS:
        return {c: -1 for c in FLAT_ORDER[:KEY_FLATS[key]]}
    sys.exit(f"未知调号: {key}（例: C, G, D, F, Bb, Am, Em）")


def parse_dur(s):
    try:
        return Fraction(s) if not s.startswith('.') else Fraction('0' + s)
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"无法解析时值 '{s}'")


def parse(path):
    meta = {'tempo': '72', 'time': '4/4', 'key': 'C', 'title': Path(path).stem}
    voices = []  # [{name, vel, pedal, bars: [[tokens]]}]
    cur = None
    for lineno, raw in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        line = raw.split('//', 1)[0].strip()      # 注释用 //（# 是升号）
        if not line:
            continue
        m = re.match(r'^\[(\w+)(.*)\]$', line)
        if m:
            opts = dict(kv.split('=') for kv in m.group(2).split())
            cur = {'name': m.group(1), 'vel': float(opts.get('vel', 0.25)),
                   'pedal': float(opts.get('pedal', 0.3)), 'bars': []}
            voices.append(cur)
            continue
        m = re.match(r'^(\w+)\s*:\s*(.+)$', line)
        if m and cur is None:
            meta[m.group(1).lower()] = m.group(2).strip()
            continue
        if cur is None:
            sys.exit(f"第 {lineno} 行: 音符出现在任何 [声部] 之前")
        for chunk in line.split('|'):
            toks = chunk.split()
            if toks:
                cur['bars'].append((lineno, toks))
    return meta, voices


def note_midi(name, keysig, bar_acc):
    m = NOTE_RE.match(name)
    if not m:
        raise ValueError(f"无法解析音名 '{name}'（例: C4, F#4, Bb3, En4）")
    letter, acc, octv = m.group(1).upper(), m.group(2), int(m.group(3))
    slot = (letter, octv)
    if acc:
        shift = {'#': 1, '##': 2, 'b': -1, 'bb': -2, 'n': 0}[acc]
        bar_acc[slot] = shift            # 临时记号在本小节内持续有效
    else:
        shift = bar_acc.get(slot, keysig.get(letter, 0))
    return 12 * (octv + 1) + LETTER[letter] + shift


def build_events(meta, voices, bar_range=None):
    keysig = key_signature(meta['key'])
    num, den = map(int, meta['time'].split('/'))
    bar_len = Fraction(num * 4, den)     # 以四分音符为一拍
    problems, events = [], []            # events: (voice_idx, start_beat, dur_beat, midi)
    nbars = max(len(v['bars']) for v in voices)
    lo, hi = bar_range or (1, nbars)
    for vi, v in enumerate(voices):
        pos = Fraction(0)
        pending_tie = {}                 # midi -> event index
        for bi, (lineno, toks) in enumerate(v['bars'], 1):
            bar_acc, total = {}, Fraction(0)
            for tok in toks:
                tie = tok.endswith('~')
                tok = tok.rstrip('~')
                try:
                    pitch, d = tok.rsplit(':', 1)
                    d = parse_dur(d)
                    mids = [] if pitch.lower() == 'r' else [note_midi(p, keysig, bar_acc) for p in pitch.split('+')]
                except ValueError as e:
                    sys.exit(f"[{v['name']}] 第 {bi} 小节 (行 {lineno}) 记号 '{tok}': {e}")
                new_tie = {}
                for mid in mids:
                    if mid in pending_tie:   # 连音线：延长上一个音
                        events[pending_tie[mid]][2] += d
                        idx = pending_tie[mid]
                    else:
                        idx = len(events)
                        if lo <= bi <= hi:
                            events.append([vi, pos, d, mid])
                        else:
                            idx = None
                    if tie and idx is not None:
                        new_tie[mid] = idx
                pending_tie = new_tie
                pos += d
                total += d
            if total != bar_len:
                problems.append(f"[{v['name']}] 第 {bi} 小节 (行 {lineno}): 共 {float(total):g} 拍，应为 {float(bar_len):g} 拍")
        if len(v['bars']) != nbars:
            problems.append(f"[{v['name']}] 只有 {len(v['bars'])} 小节，其他声部最多 {nbars} 小节")
    offset = (lo - 1) * bar_len
    events = [[vi, s - offset, d, mid] for vi, s, d, mid in events]
    return events, problems


def tone(midi, dur_s, vel):
    f = 440 * 2 ** ((midi - 69) / 12)
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for k, a in enumerate([1, .5, .3, .18, .1, .06], 1):
        if f * k > SR / 2.2:
            break
        decay = np.exp(-t * (1.2 + 0.9 * k) * (f / 440) ** 0.35)
        sig += a * decay * np.sin(2 * np.pi * f * k * t * (1 + 0.0004 * k))
    attack = np.minimum(1, t / 0.004)
    release = np.minimum(1, (n - np.arange(n)) / (0.08 * SR))
    return vel * sig * attack * release


def render_audio(events, voices, bpm):
    beat = 60 / bpm
    end = max((float(s + d) * beat + voices[vi]['pedal'] for vi, s, d, _ in events), default=0) + 0.1
    buf = np.zeros(int(SR * end))
    for vi, s, d, mid in events:
        v = voices[vi]
        w = tone(mid, float(d) * beat + v['pedal'], v['vel'])
        a = int(SR * float(s) * beat)
        buf[a:a + len(w)] += w[:len(buf) - a]
    peak = np.max(np.abs(buf)) or 1
    # 去掉结尾几乎听不见的余音（低于峰值 -50dB），再加 0.15 秒淡出，避免结尾空白
    loud = np.nonzero(np.abs(buf) > peak * 10 ** (-50 / 20))[0]
    if len(loud):
        buf = buf[:loud[-1] + 1]
        fade = min(len(buf), int(0.15 * SR))
        buf[-fade:] *= np.linspace(1, 0, fade)
    return (buf / peak * 0.9 * 32767).astype(np.int16)


def write_midi(path, events, voices, bpm):
    tpq = 480
    def vlq(x):
        out = [x & 0x7F]
        while x > 0x7F:
            x >>= 7
            out.append((x & 0x7F) | 0x80)
        return bytes(reversed(out))
    tracks = []
    tempo = int(60_000_000 / bpm)
    tracks.append(b'\x00\xff\x51\x03' + tempo.to_bytes(3, 'big') + b'\x00\xff\x2f\x00')
    for vi, v in enumerate(voices):
        ch = vi % 16
        evs = []
        for evi, s, d, mid in events:
            if evi == vi:
                evs.append((int(s * tpq), 1, 0x90 | ch, mid, 80))
                evs.append((int((s + d) * tpq), 0, 0x80 | ch, mid, 0))
        evs.sort()
        data, last = bytearray(), 0
        name = v['name'].encode()
        data += b'\x00\xff\x03' + vlq(len(name)) + name
        data += bytes([0, 0xC0 | ch, 0])                   # 钢琴音色
        for t, _, st, mid, vel in evs:
            data += vlq(t - last) + bytes([st, mid, vel])
            last = t
        data += b'\x00\xff\x2f\x00'
        tracks.append(bytes(data))
    with open(path, 'wb') as f:
        f.write(b'MThd' + struct.pack('>IHHH', 6, 1, len(tracks), tpq))
        for tr in tracks:
            f.write(b'MTrk' + struct.pack('>I', len(tr)) + tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('score')
    ap.add_argument('-o', '--out', help='输出路径（不含扩展名），默认与乐谱同名')
    ap.add_argument('--check', action='store_true', help='只检查小节时值')
    ap.add_argument('--bars', help='只渲染某段小节，如 5-12')
    ap.add_argument('--midi', action='store_true', help='额外输出 MIDI 文件')
    ap.add_argument('--tempo', type=float, help='覆盖速度 (BPM)')
    args = ap.parse_args()

    meta, voices = parse(args.score)
    if not voices:
        sys.exit("乐谱里没有任何 [声部]")
    rng = tuple(map(int, args.bars.split('-'))) if args.bars else None
    events, problems = build_events(meta, voices, rng)
    for p in problems:
        print("⚠", p)
    print(f"{meta['title']}: {len(voices)} 个声部, {max(len(v['bars']) for v in voices)} 小节, "
          f"{len(events)} 个音, 调号 {meta['key']}, 拍号 {meta['time']}")
    if args.check:
        sys.exit(1 if problems else 0)

    bpm = args.tempo or float(meta['tempo'])
    out = Path(args.out or Path(args.score).with_suffix(''))
    out.parent.mkdir(parents=True, exist_ok=True)
    pcm = render_audio(events, voices, bpm)
    wav = out.with_suffix('.wav')
    with wave.open(str(wav), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    if shutil.which('ffmpeg'):
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(wav), '-b:a', '192k',
                        str(out.with_suffix('.mp3'))], check=True)
        wav.unlink()
        print("输出:", out.with_suffix('.mp3'))
    else:
        print("输出:", wav)
    if args.midi:
        write_midi(out.with_suffix('.mid'), events, voices, bpm)
        print("输出:", out.with_suffix('.mid'))
    print(f"时长 {len(pcm) / SR:.1f} 秒, ♩={bpm:g}")


if __name__ == '__main__':
    main()
