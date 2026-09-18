"""Compares two MIDI files note by note, with a per-instrument breakdown.

A speed-up or a shortcut is only worth having if you know what it costs. Onset
detection is a threshold on a continuous curve, so a small change upstream can flip
notes near the threshold -- and it rarely does so evenly across the kit.

The first file is the reference, the second is the one under test.

`--by-instrument` matches kick against kick regardless of whether it was written as note
35 or 36. Use it whenever the two files come from different tools: General MIDI offers
several numbers for the same drum, and comparing on the exact number reports total
disagreement where there is none.

    python compare_midi.py reference.mid test.mid [tolerance_ms] [--by-instrument]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pretty_midi

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if len(sys.argv) < 3 or {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0 if len(sys.argv) > 1 else 1)

a_path, b_path = Path(sys.argv[1]), Path(sys.argv[2])
rest = [x for x in sys.argv[3:] if not x.startswith("-")]
tol = float(rest[0]) / 1000 if rest else 0.002
coarse = "--by-instrument" in sys.argv[3:]

NAMES = {35: "kick", 36: "kick", 38: "snare", 40: "snare", 37: "side stick",
         42: "hi-hat closed", 44: "hi-hat pedal", 46: "hi-hat open",
         41: "low tom", 43: "low tom", 45: "mid tom", 47: "mid tom",
         48: "high tom", 50: "high tom", 49: "crash", 57: "crash",
         51: "ride", 59: "ride", 53: "ride bell", 55: "splash"}

# Coarser grouping for cross-product comparison. General MIDI gives several numbers for
# the same drum (35/36 kick, 38/40 snare) and transcribers choose differently, so
# matching on the exact number reports total disagreement where there is none.
GROUP = {35: "kick", 36: "kick",
         37: "snare", 38: "snare", 40: "snare",
         41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
         42: "hi-hat", 44: "hi-hat", 46: "hi-hat",
         49: "crash", 57: "crash", 55: "crash",
         51: "ride", 59: "ride", 53: "ride"}


def label(pitch: int, coarse: bool) -> str:
    table = GROUP if coarse else NAMES
    return table.get(pitch, f"note {pitch}")


def notes(path: Path):
    m = pretty_midi.PrettyMIDI(str(path))
    return sorted(((n.start, n.pitch, n.velocity)
                   for inst in m.instruments for n in inst.notes))


a, b = notes(a_path), notes(b_path)
print(f"reference : {a_path.name}  {len(a)} notes")
print(f"test      : {b_path.name}  {len(b)} notes")
print(f"tolerance : {tol*1000:.0f} ms"
      + ("   matching by instrument group\n" if coarse else "   matching by exact pitch\n"))

used = [False] * len(b)
per = defaultdict(lambda: {"ref": 0, "test": 0, "hit": 0})
vel_diff, time_diff = [], []

for start, pitch, vel in a:
    name = label(pitch, coarse)
    per[name]["ref"] += 1
    best, best_dt = None, tol
    for j, (s2, p2, v2) in enumerate(b):
        if used[j] or label(p2, coarse) != name:
            continue
        dt = abs(s2 - start)
        if dt <= best_dt:
            best, best_dt = j, dt
    if best is not None:
        used[best] = True
        per[name]["hit"] += 1
        vel_diff.append(abs(b[best][2] - vel))
        time_diff.append(best_dt)

for _, p2, _ in b:
    per[label(p2, coarse)]["test"] += 1

matched = sum(v["hit"] for v in per.values())
print(f"{'instrument':<16}{'reference':>10}{'test':>7}{'matched':>9}{'recall':>9}{'precision':>11}")
print("-" * 62)
for name in sorted(per, key=lambda n: -per[n]["ref"]):
    c = per[name]
    recall = c["hit"] / c["ref"] if c["ref"] else float("nan")
    prec = c["hit"] / c["test"] if c["test"] else float("nan")
    print(f"{name:<16}{c['ref']:>10}{c['test']:>7}{c['hit']:>9}"
          f"{recall:>9.3f}{prec:>11.3f}")

recall = matched / len(a) if a else 0.0
prec = matched / len(b) if b else 0.0
f1 = 2 * recall * prec / (recall + prec) if recall + prec else 0.0
print("-" * 62)
print(f"{'TOTAL':<16}{len(a):>10}{len(b):>7}{matched:>9}{recall:>9.3f}{prec:>11.3f}")
print(f"\nagreement F1: {f1:.3f}")

if vel_diff:
    same = sum(1 for d in vel_diff if d == 0)
    mean_vel = sum(vel_diff) / len(vel_diff)
    mean_t = sum(time_diff) / len(time_diff) * 1000
    print(f"velocity: identical for {same}/{len(vel_diff)}, "
          f"mean difference {mean_vel:.1f}, max {max(vel_diff)}")
    print(f"timing:   mean {mean_t:.2f} ms, max {max(time_diff)*1000:.2f} ms")

identical = len(a) == len(b) == matched and vel_diff and max(vel_diff) == 0
print("\nIDENTICAL" if identical else "\nDIFFERENT")
raise SystemExit(0 if identical else 2)
