"""Generate a synthetic drum loop with known ground truth, then score the pipeline against it.

Self-contained: no dataset needed. Any extra arguments are forwarded to drum2midi.py.

    python evaluate.py
    python evaluate.py --separator larsnet
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

SR = 44100
ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(7)


def _env(n, tau):
    return np.exp(-np.arange(n) / (tau * SR))


def kick(vel):
    n = int(0.35 * SR); t = np.arange(n) / SR
    f = 48 + 62 * np.exp(-t / 0.022)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.085)
    x[:220] += rng.normal(0, 0.5, 220) * _env(220, 0.0015)
    return x * vel


def snare(vel):
    n = int(0.25 * SR); t = np.arange(n) / SR
    noise = rng.normal(0, 1, n) * _env(n, 0.055)
    tone = (np.sin(2 * np.pi * 190 * t) + 0.6 * np.sin(2 * np.pi * 331 * t)) * _env(n, 0.07)
    return (0.8 * noise + 0.5 * tone) * vel


def _hp_noise(n, tau, cutoff=6000):
    x = rng.normal(0, 1, n)
    spec = np.fft.rfft(x)
    spec[np.fft.rfftfreq(n, 1 / SR) < cutoff] *= 0.04
    return np.fft.irfft(spec, n) * _env(n, tau)


def hat_closed(vel):
    return _hp_noise(int(0.12 * SR), 0.018) * vel


def hat_open(vel):
    return _hp_noise(int(0.6 * SR), 0.18) * vel


def crash(vel):
    return _hp_noise(int(2.0 * SR), 0.9, cutoff=3000) * vel * 0.9


def tom(freq, vel):
    n = int(0.6 * SR); t = np.arange(n) / SR
    f = freq * (1 + 0.28 * np.exp(-t / 0.05))
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.20)
    body += 0.35 * np.sin(2 * np.pi * np.cumsum(f * 1.58) / SR) * _env(n, 0.08)
    attack = rng.normal(0, 1, n) * _env(n, 0.006) * 0.55
    return (body + attack) * vel


LEAD_IN = 0.25


BPM = 120.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT
BARS = 8

truth = []   # (time, family, velocity 0..1)
layers = []


def place(sig, t, family, vel):
    t = t + LEAD_IN
    layers.append((sig, t))
    truth.append((t, family, vel))


for b in range(BARS):
    t0 = b * BAR
    for off, v in [(0.0, 1.00), (0.5 * BEAT, 0.55), (2 * BEAT, 0.92), (2.75 * BEAT, 0.70)]:
        place(kick(v), t0 + off, "kick", v)
    for off, v in [(BEAT, 0.95), (3 * BEAT, 1.00)]:
        place(snare(v), t0 + off, "snare", v)
    for off, v in [(1.5 * BEAT, 0.28), (2.5 * BEAT, 0.22)]:   # ghost notes
        place(snare(v), t0 + off, "snare", v)
    if b != BARS - 1:
        for i in range(8):
            v = 0.85 if i % 2 == 0 else 0.45
            if i == 6:
                place(hat_open(0.8), t0 + i * 0.5 * BEAT, "hat", 0.8)
            else:
                place(hat_closed(v), t0 + i * 0.5 * BEAT, "hat", v)
    else:                                                     # tom fill
        for i, (f, v) in enumerate([(210, 0.95), (210, 0.6), (160, 0.9),
                                    (160, 0.62), (110, 1.0), (110, 0.7)]):
            place(tom(f, v), t0 + 2 * BEAT + i * 0.33 * BEAT, "tom", v)
    if b % 4 == 0:
        place(crash(1.0), t0, "crash", 1.0)

total = int((BARS * BAR + LEAD_IN + 3.0) * SR)
mix = np.zeros(total)
for sig, t in layers:
    i = int(t * SR)
    mix[i:i + len(sig)] += sig[:max(0, total - i)]
mix /= np.max(np.abs(mix)) * 1.08

ref = ROOT / "input" / "synth.wav"
ref.parent.mkdir(parents=True, exist_ok=True)
sf.write(ref, np.stack([mix, mix], axis=1), SR)
print(f"Wrote {ref}  ({total / SR:.1f}s, {len(truth)} ground-truth hits)")

out_mid = ROOT / "out" / "synth.mid"
extra = sys.argv[1:]
cmd = [sys.executable, str(ROOT / "drum2midi.py"), str(ref), "-o", str(out_mid),
       "--device", "auto"] + extra
print("Running pipeline " + (" ".join(extra) if extra else "(defaults)") + " ...")
res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
print(res.stdout[-1800:])
if res.returncode != 0:
    print("PIPELINE FAILED", res.stderr[-2000:])
    raise SystemExit(1)

FAMILY = {35: "kick", 36: "kick", 38: "snare", 40: "snare",
          41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
          42: "hat", 44: "hat", 46: "hat", 49: "crash", 51: "crash"}

det = [(n.start, FAMILY.get(n.pitch, "?"), n.velocity / 127.0)
       for n in pretty_midi.PrettyMIDI(str(out_mid)).instruments[0].notes]

TOL = 0.05
print(f"\n{'family':<8}{'truth':>6}{'det':>6}{'hit':>6}{'miss':>6}{'extra':>7}"
      f"{'prec':>7}{'rec':>7}{'F1':>7}{'vel r':>8}")
print("-" * 68)
overall = [0, 0, 0]
for fam in ["kick", "snare", "hat", "tom", "crash"]:
    gt = sorted(x for x in truth if x[1] == fam)
    dt = sorted(x for x in det if x[1] == fam)
    used, pairs = set(), []
    for gtime, _, gvel in gt:
        best, bd = None, TOL
        for i, (dtime, _, dvel) in enumerate(dt):
            if i in used:
                continue
            if abs(dtime - gtime) <= bd:
                best, bd = i, abs(dtime - gtime)
        if best is not None:
            used.add(best)
            pairs.append((gvel, dt[best][2]))
    tp, fn, fp = len(pairs), len(gt) - len(pairs), len(dt) - len(pairs)
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    if len(pairs) > 2:
        gv = np.array([p[0] for p in pairs]); dv = np.array([p[1] for p in pairs])
        r = float(np.corrcoef(gv, dv)[0, 1]) if gv.std() > 0 and dv.std() > 0 else float("nan")
    else:
        r = float("nan")
    overall[0] += tp; overall[1] += fn; overall[2] += fp
    print(f"{fam:<8}{len(gt):>6}{len(dt):>6}{tp:>6}{fn:>6}{fp:>7}"
          f"{prec:>7.2f}{rec:>7.2f}{f1:>7.2f}{r:>8.2f}")

tp, fn, fp = overall
prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
print("-" * 68)
print(f"{'TOTAL':<8}{tp + fn:>6}{tp + fp:>6}{tp:>6}{fn:>6}{fp:>7}"
      f"{prec:>7.2f}{rec:>7.2f}{2 * prec * rec / max(prec + rec, 1e-9):>7.2f}")

# hi-hat articulation: how many true-open / true-closed got the right pitch
notes = pretty_midi.PrettyMIDI(str(out_mid)).instruments[0].notes
open_true = {round(LEAD_IN + b * BAR + 6 * 0.5 * BEAT, 3) for b in range(BARS - 1)}
ok_open = ok_closed = wrong_open = wrong_closed = 0
for n in notes:
    if n.pitch not in (42, 44, 46):
        continue
    is_true_open = any(abs(n.start - o) <= TOL for o in open_true)
    if is_true_open:
        ok_open += n.pitch == 46
        wrong_closed += n.pitch != 46
    else:
        ok_closed += n.pitch != 46
        wrong_open += n.pitch == 46
print(f"\nhi-hat articulation: open {ok_open}/{ok_open + wrong_closed} correct, "
      f"closed {ok_closed}/{ok_closed + wrong_open} correct")
