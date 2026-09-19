"""Splits the IDMT-SMT-Drums score by how the audio was made.

The 0.935 headline on this corpus is an average over three very different things, and
the proportions are lopsided: 60 WaveDrum02 + 10 WaveDrum01 files are built from samples,
11 TechnoDrum files are a drum machine, and only 14 RealDrum files are an actual kit in
a room. Quoting one number for all 95 hides that three quarters of the evidence is
synthetic.

Scores the MIDI benchmark_idmt.py already produced, so this costs seconds rather than a
re-run.

    python idmt_by_kind.py
    python idmt_by_kind.py --dir bench/idmt_larsnet
"""

from __future__ import annotations

import argparse
import random
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mir_eval  # noqa: E402
import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402

IDMT = ROOT / "idmt"
ANN = IDMT / "annotation_xml"
WINDOW = 0.05

# IDMT labels: KD kick, SD snare, HH hi-hat. Toms and cymbals are not annotated here.
CLASSES = {"KD": {35, 36}, "SD": {38, 40}, "HH": {42, 44, 46}}
ORDER = ["KD", "SD", "HH"]
PRETTY = {"KD": "kick", "SD": "snare", "HH": "hi-hat"}

KINDS = {
    "RealDrum": "real kit in a room",
    "WaveDrum": "sample-based",
    "TechnoDrum": "drum machine",
}


def kind_of(stem: str) -> str:
    for k in KINDS:
        if stem.startswith(k):
            return k
    return "other"


def read_xml(path: Path) -> dict:
    """IDMT annotations are XML: <event><onsetSec/><instrument/></event>."""
    out = {k: [] for k in CLASSES}
    root = ET.parse(path).getroot()
    for ev in root.iter("event"):
        sec = ev.findtext("onsetSec")
        inst = (ev.findtext("instrument") or "").strip().upper()
        if sec is None or inst not in out:
            continue
        out[inst].append(float(sec))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def read_midi(path: Path) -> dict:
    notes = [n for inst in pretty_midi.PrettyMIDI(str(path)).instruments
             for n in inst.notes]
    return {k: np.array(sorted(n.start for n in notes if n.pitch in p))
            for k, p in CLASSES.items()}


def counts(ref: dict, est: dict) -> dict:
    acc = {}
    for k in CLASSES:
        r, e = ref[k], est[k]
        tp = 0
        if len(r) and len(e):
            _, p, _ = mir_eval.onset.f_measure(r, e, window=WINDOW)
            tp = int(round(p * len(e)))
        acc[k] = {"tp": tp, "ref": len(r), "est": len(e)}
    return acc


def prf(c):
    p = c["tp"] / max(c["est"], 1)
    r = c["tp"] / max(c["ref"], 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def micro(rows, picks, family=None) -> float:
    tot = {"tp": 0, "ref": 0, "est": 0}
    for name in picks:
        for k in ([family] if family else ORDER):
            for key in tot:
                tot[key] += rows[name][k][key]
    return prf(tot)[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "bench" / "idmt_uvr")
    ap.add_argument("--rounds", type=int, default=4000)
    args = ap.parse_args()

    if not args.dir.exists():
        print(f"no MIDI in {args.dir}; run benchmark_idmt.py first")
        return 1

    rows, by_kind = {}, {}
    for mid in sorted(args.dir.glob("*.mid")):
        ann = ANN / f"{mid.stem}.xml"
        if not ann.exists():
            continue
        rows[mid.stem] = counts(read_xml(ann), read_midi(mid))
        by_kind.setdefault(kind_of(mid.stem), []).append(mid.stem)

    if not rows:
        print("no annotation/MIDI pairs found")
        return 1

    print(f"{len(rows)} files scored from {args.dir.name}\n")
    print(f"{'subset':<12}{'files':>6}{'onsets':>8}"
          f"{'kick':>8}{'snare':>8}{'hi-hat':>8}{'MICRO':>8}   what it is")
    print("-" * 82)

    for kind in ["RealDrum", "WaveDrum", "TechnoDrum", "other"]:
        names = by_kind.get(kind)
        if not names:
            continue
        onsets = sum(rows[n][k]["ref"] for n in names for k in ORDER)
        cells = "".join(f"{micro(rows, names, k):>8.3f}" for k in ORDER)
        print(f"{kind:<12}{len(names):>6}{onsets:>8}{cells}"
              f"{micro(rows, names):>8.3f}   {KINDS.get(kind, '')}")

    everything = sorted(rows)
    total = sum(rows[n][k]["ref"] for n in everything for k in ORDER)
    cells = "".join(f"{micro(rows, everything, k):>8.3f}" for k in ORDER)
    print("-" * 82)
    print(f"{'ALL':<12}{len(everything):>6}{total:>8}{cells}"
          f"{micro(rows, everything):>8.3f}   the number the README quotes")

    real = by_kind.get("RealDrum", [])
    synth = by_kind.get("WaveDrum", []) + by_kind.get("TechnoDrum", [])
    if real and synth:
        rng = random.Random(0)
        diffs = []
        for _ in range(args.rounds):
            a = [real[rng.randrange(len(real))] for _ in real]
            b = [synth[rng.randrange(len(synth))] for _ in synth]
            diffs.append(micro(rows, a) - micro(rows, b))
        diffs.sort()
        lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
        gap = micro(rows, real) - micro(rows, synth)
        sig = "significant" if lo > 0 or hi < 0 else "NOT significant"
        print(f"\nreal minus synthetic: {gap:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   {sig}")
        print(f"(real rests on {len(real)} files, synthetic on {len(synth)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
