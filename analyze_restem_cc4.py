"""How ReStem's cc4 maps onto real hi-hat articulation.

In trigger_events.json a hi-hat event carries only the continuous cc4 value, with no
class label. That means the closed/pedal/open split is made by ReStem's MIDI writer using
its own thresholds. To keep the comparison fair, those thresholds are recovered from the
ground truth rather than guessed.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
MDB = ROOT / "mdbdrums" / "MDB Drums"
SUB = MDB / "annotations" / "subclass"
EVENTS = ROOT / "restem_events"
WINDOW = 0.05
LABELS = {"CHH": "closed", "OHH": "open", "PHH": "pedal"}

pairs = []
for f in sorted(EVENTS.glob("*.json")):
    base = f.stem.replace("_Drum", "")
    ann_path = SUB / f"{base}_subclass.txt"
    if not ann_path.exists():
        continue
    ann = []
    for line in ann_path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] in LABELS:
            ann.append((float(p[0]), LABELS[p[1]]))
    d = json.loads(f.read_text(encoding="utf-8"))
    sr = float(d.get("sr", 44100))
    hh = [(float(e["sample"]) / sr, float(e.get("cc4", 0.0)))
          for e in d["stems"].get("hh", [])]
    if not hh or not ann:
        continue
    times = np.array([t for t, _ in hh])
    used = set()
    for t, kind in ann:
        d_ = np.abs(times - t)
        order = np.argsort(d_)
        for j in order:
            if d_[j] > WINDOW:
                break
            if j in used:
                continue
            used.add(int(j))
            pairs.append((kind, hh[j][1]))
            break

print(f"matched hits: {len(pairs)}")
by = {k: sorted(v for kk, v in pairs if kk == k) for k in ("closed", "open", "pedal")}
print(f"\n{'articulation':<12}{'n':>6}{'cc4 p10':>10}{'median':>10}{'p90':>10}")
print("-" * 48)
for k, v in by.items():
    if not v:
        print(f"{k:<12}{0:>6}")
        continue
    a = np.array(v)
    print(f"{k:<12}{len(a):>6}{np.percentile(a,10):>10.3f}"
          f"{np.median(a):>10.3f}{np.percentile(a,90):>10.3f}")

# two thresholds: closed < lo <= pedal < hi <= open, found by brute force
best = (0.0, None)
grid = np.linspace(-0.05, 0.95, 51)
for lo in grid:
    for hi in grid:
        if hi <= lo:
            continue
        ok = 0
        for kind, v in pairs:
            pred = "closed" if v < lo else ("pedal" if v < hi else "open")
            ok += pred == kind
        acc = ok / max(len(pairs), 1)
        if acc > best[0]:
            best = (acc, (lo, hi))

print(f"\nbest three-class split on cc4: accuracy {best[0]:.3f} "
      f"at thresholds {best[1][0]:.2f} / {best[1][1]:.2f}")

# for comparison: closed/open only
best2 = (0.0, None)
for t in grid:
    ok = sum(1 for kind, v in pairs
             if (("open" if v >= t else "closed") == kind))
    n = sum(1 for kind, _ in pairs if kind in ("closed", "open"))
    acc = ok / max(len(pairs), 1)
    if acc > best2[0]:
        best2 = (acc, t)
print(f"closed/open only (pedal counted as closed): accuracy {best2[0]:.3f} "
      f"at threshold {best2[1]:.2f}")

cm = Counter()
lo, hi = best[1]
for kind, v in pairs:
    pred = "closed" if v < lo else ("pedal" if v < hi else "open")
    cm[(kind, pred)] += 1
print(f"\nconfusion matrix at the best thresholds:")
print(f"{'truth \\ cc4':<14}" + "".join(f"{k:>9}" for k in ("closed", "pedal", "open")))
for k in ("closed", "pedal", "open"):
    row = sum(cm[(k, e)] for e in ("closed", "pedal", "open"))
    acc = f"  ({cm[(k,k)]/row:.2f})" if row else ""
    print(f"{k:<14}" + "".join(f"{cm[(k,e)]:>9}" for e in ("closed", "pedal", "open")) + acc)
