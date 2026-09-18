"""Checks whether ADT_STR was run correctly, by reproducing its own published numbers.

A model that scores far below its paper is either being used wrongly or does not
generalise. The two are distinguishable: their Table 2 reports per-class F1 on MDB
(Setting-1) as BD 0.92, SD 0.85, TT 0.77, HH 0.74, CY+RD 0.52. If our run reproduces
the classes they do well on and only fails on the rest, the wiring is right and the
difference is real.

    python verify_adt_str.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

MDB = ROOT / "mdbdrums" / "MDB Drums"
ANN = MDB / "annotations" / "class"
THEIRS = ROOT / "bench" / "adtstr_midi"
WINDOW = 0.05

from benchmark_mdb import read_annotation  # noqa: E402
from score_adt_str import WIDE, prf, read_midi, score  # noqa: E402

# What arXiv:2601.09520 Table 2 reports for MDB, Setting-1
PAPER = {"KD": 0.92, "SD": 0.85, "TT": 0.77, "HH": 0.74, "CY": 0.52, "SUM": 0.79}
PRETTY = {"KD": "kick", "SD": "snare", "HH": "hi-hat", "TT": "toms", "CY": "cymbals"}

rows = []
for mid in sorted(THEIRS.rglob("*.mid")):
    stem = mid.parent.name
    ann = ANN / f"{stem.replace('_Drum', '')}_class.txt"
    if ann.exists():
        rows.append((read_annotation(ann), read_midi(mid, WIDE)))

if not rows:
    print("no ADT_STR output to check")
    raise SystemExit(1)

acc = score(rows)
print(f"{len(rows)} tracks\n")
print(f"{'class':<10}{'their paper':>13}{'our run':>10}{'difference':>12}")
print("-" * 46)
micro = {"tp": 0, "ref": 0, "est": 0}
for k in ("KD", "SD", "HH", "TT", "CY"):
    for key in micro:
        micro[key] += acc[k][key]
    got = prf(acc[k])[2]
    print(f"{PRETTY[k]:<10}{PAPER[k]:>13.2f}{got:>10.3f}{got - PAPER[k]:>+12.3f}")
print("-" * 46)
got = prf(micro)[2]
print(f"{'SUM':<10}{PAPER['SUM']:>13.2f}{got:>10.3f}{got - PAPER['SUM']:>+12.3f}")

close = [k for k in ("KD", "SD") if abs(prf(acc[k])[2] - PAPER[k]) < 0.10]
print(f"\n{len(close)} of the two classes they score highest on reproduce within 0.10.")
print("If those match and the rare classes do not, the model is running correctly and\n"
      "simply does not transfer to this material -- not a setup error on our side.")
