"""Second, independent test set: IDMT-SMT-Drums.

Everything else is tuned and measured on the 23 MDB Drums tracks, which is not enough to
call the settings general: thresholds, separator choice and articulation rules could have
quietly adapted to a single corpus.

IDMT-SMT-Drums is a classic automatic drum transcription benchmark: real (RealDrum),
electronic (TechnoDrum) and hybrid (WaveDrum) recordings, 95 mixes with hand annotations.
It has only three classes - KD, SD, HH - so toms and cymbals are not measured here.

Licensed CC BY-NC-ND 4.0: fine for evaluation, not for training a commercial model.

    python benchmark_idmt.py --limit 30
    python benchmark_idmt.py --separator larsnet
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
IDMT = ROOT / "idmt"
WINDOW = 0.05

CLASSES = {
    "KD": (35, 36),
    "SD": (38, 40),
    "HH": (42, 44, 46),
}
PRETTY = {"KD": "kick", "SD": "snare", "HH": "hi-hat"}


def read_xml(path: Path) -> dict:
    out = {k: [] for k in CLASSES}
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return {k: np.array([]) for k in CLASSES}
    for ev in root.iter("event"):
        inst = ev.findtext("instrument", "").strip()
        onset = ev.findtext("onsetSec")
        if inst in out and onset:
            out[inst].append(float(onset))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def read_midi(path: Path) -> dict:
    notes = [n for inst in pretty_midi.PrettyMIDI(str(path)).instruments for n in inst.notes]
    return {k: np.array(sorted(n.start for n in notes if n.pitch in pitches))
            for k, pitches in CLASSES.items()}


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--separator", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    args, passthrough = ap.parse_known_args()
    if args.help:
        print(__doc__)
        return 0

    audio_dir = next((p for p in IDMT.rglob("audio") if p.is_dir()), None)
    xml_dir = next((p for p in IDMT.rglob("annotation_xml") if p.is_dir()), None)
    if not audio_dir or not xml_dir:
        print(f"IDMT-SMT-Drums not found in {IDMT}")
        return 1

    pairs = []
    for x in sorted(xml_dir.glob("*MIX.xml")):
        wav = audio_dir / f"{x.stem}.wav"
        if wav.exists():
            pairs.append((wav, x))
    pairs = pairs[: args.limit]
    print(f"files: {len(pairs)}")

    tag = args.tag or (args.separator or "default")
    outdir = ROOT / "bench" / f"idmt_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.rescore:
        extra = list(passthrough)
        if args.separator:
            extra += ["--separator", args.separator]
        t0 = time.time()
        for i, (wav, _) in enumerate(pairs, 1):
            out = outdir / f"{wav.stem}.mid"
            if out.exists():
                continue
            subprocess.run(
                [sys.executable, str(ROOT / "drum2midi.py"), str(wav),
                 "-o", str(out), "--device", "auto"] + extra,
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            if i % 5 == 0 or i == len(pairs):
                el = time.time() - t0
                print(f"  [{i:3d}/{len(pairs)}] {el/60:.1f} min, "
                      f"eta ~{el/i*(len(pairs)-i)/60:.0f} min", flush=True)

    totals = {k: {"tp": 0, "ref": 0, "est": 0} for k in CLASSES}
    by_group = {}
    for wav, xml in pairs:
        mid = outdir / f"{wav.stem}.mid"
        if not mid.exists():
            continue
        ref, est = read_xml(xml), read_midi(mid)
        group = wav.stem.split("0")[0].replace("#MIX", "")
        g = by_group.setdefault(group, {k: {"tp": 0, "ref": 0, "est": 0} for k in CLASSES})
        for k in CLASSES:
            r, e = ref[k], est[k]
            tp = 0
            if len(r) and len(e):
                _, p, _ = mir_eval.onset.f_measure(r, e, window=WINDOW)
                tp = int(round(p * len(e)))
            for acc in (totals[k], g[k]):
                acc["tp"] += tp
                acc["ref"] += len(r)
                acc["est"] += len(e)

    def prf(c):
        p = c["tp"] / max(c["est"], 1)
        r = c["tp"] / max(c["ref"], 1)
        return p, r, 2 * p * r / max(p + r, 1e-9)

    print(f"\n{'instrument':<11}{'reference':>8}{'found':>9}{'P':>8}{'R':>8}{'F1':>8}")
    print("-" * 52)
    micro = {"tp": 0, "ref": 0, "est": 0}
    for k in CLASSES:
        c = totals[k]
        for key in micro:
            micro[key] += c[key]
        p, r, f = prf(c)
        print(f"{PRETTY[k]:<11}{c['ref']:>8}{c['est']:>9}{p:>8.3f}{r:>8.3f}{f:>8.3f}")
    p, r, f = prf(micro)
    print("-" * 52)
    print(f"{'TOTAL':<11}{micro['ref']:>8}{micro['est']:>9}{p:>8.3f}{r:>8.3f}{f:>8.3f}")

    print(f"\nby recording type:")
    for group, acc in sorted(by_group.items()):
        m = {"tp": 0, "ref": 0, "est": 0}
        for k in CLASSES:
            for key in m:
                m[key] += acc[k][key]
        print(f"  {group:<16}{prf(m)[2]:>8.3f}  ({m['ref']} hits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
