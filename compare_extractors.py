"""Which model should pull the drums out of a full mix?

htdemucs was chosen by default rather than by measurement -- it is what audio-separator
offers first -- and it sits on the pipeline's weakest path: feeding a whole song costs
40% of the ride hits. audio-separator carries four models that emit a drums stem, and
their published SDR on drums differs by 1.5 dB:

    htdemucs_ft   10.0      hdemucs_mmi   9.6
    htdemucs       9.4      htdemucs_6s   8.5

SDR is not our metric though. What matters is how many notes survive to the MIDI, so
each extractor is run over MDB's full mixes, the pipeline transcribes the result, and
the output is scored against the same hand annotations with the same code as everywhere
else. The drum-only recordings give the ceiling.

    python compare_extractors.py --limit 5
    python compare_extractors.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MDB = ROOT / "mdbdrums" / "MDB Drums"
FULL = MDB / "audio" / "full_mix"
ANN = MDB / "annotations" / "class"
CACHE = ROOT / "bench" / "extractors"
WINDOW = 0.05

from benchmark_mdb import CLASSES, ORDER, PRETTY, read_annotation  # noqa: E402

EXTRACTORS = {
    "htdemucs": "htdemucs.yaml",
    "htdemucs_ft": "htdemucs_ft.yaml",
    "hdemucs_mmi": "hdemucs_mmi.yaml",
}


def read_midi(path: Path) -> dict:
    notes = [n for inst in pretty_midi.PrettyMIDI(str(path)).instruments
             for n in inst.notes]
    return {k: np.array(sorted(n.start for n in notes if n.pitch in pitches))
            for k, pitches in CLASSES.items()}


def transcribe(src: Path, out: Path, extractor: str | None) -> bool:
    """Runs the pipeline; extractor=None means the audio is already a drum stem."""
    if out.exists():
        return True
    cmd = [sys.executable, str(ROOT / "drum2midi.py"), str(src), "-o", str(out),
           "--device", "auto"]
    if extractor:
        cmd += ["--from-song", "--extractor", EXTRACTORS[extractor]]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    if not out.exists():
        tail = (res.stderr or res.stdout or "").strip().splitlines()
        print(f"    ! {src.stem}: {tail[-1][:100] if tail else 'no output'}")
        return False
    return True


def score(pairs) -> dict:
    acc = {k: {"tp": 0, "ref": 0, "est": 0} for k in CLASSES}
    for ref, est in pairs:
        for k in CLASSES:
            r, e = ref[k], est[k]
            tp = 0
            if len(r) and len(e):
                _, p, _ = mir_eval.onset.f_measure(r, e, window=WINDOW)
                tp = int(round(p * len(e)))
            acc[k]["tp"] += tp
            acc[k]["ref"] += len(r)
            acc[k]["est"] += len(e)
    return acc


def prf(c):
    p = c["tp"] / max(c["est"], 1)
    r = c["tp"] / max(c["ref"], 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", nargs="*", choices=list(EXTRACTORS),
                    help="test a subset")
    args = ap.parse_args()

    if not FULL.exists():
        print(f"MDB full mixes not found at {FULL}")
        return 1

    tracks = sorted(FULL.glob("*.wav"))[: args.limit]
    wanted = args.only or list(EXTRACTORS)
    print(f"{len(tracks)} full mixes, extractors: {', '.join(wanted)}\n")

    results = {}
    for name in wanted:
        outdir = CACHE / name
        outdir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        pairs = []
        for i, wav in enumerate(tracks, 1):
            ann = ANN / f"{wav.stem.replace('_MIX', '')}_class.txt"
            if not ann.exists():
                continue
            mid = outdir / f"{wav.stem}.mid"
            if transcribe(wav, mid, name):
                pairs.append((read_annotation(ann), read_midi(mid)))
            if i % 5 == 0:
                print(f"  {name}: {i}/{len(tracks)}, {(time.time()-t0)/60:.1f} min",
                      flush=True)
        results[name] = (score(pairs), time.time() - t0, len(pairs))
        print(f"  {name}: done, {(time.time()-t0)/60:.1f} min")

    print(f"\n{'class':<10}" + "".join(f"{n:>14}" for n in wanted))
    print("-" * (10 + 14 * len(wanted)))
    micro = {n: {"tp": 0, "ref": 0, "est": 0} for n in wanted}
    for k in ORDER:
        row = f"{PRETTY[k]:<10}"
        for n in wanted:
            acc = results[n][0]
            for key in micro[n]:
                micro[n][key] += acc[k][key]
            row += f"{prf(acc[k])[2]:>14.3f}"
        print(row)
    print("-" * (10 + 14 * len(wanted)))
    print(f"{'MICRO':<10}" + "".join(f"{prf(micro[n])[2]:>14.3f}" for n in wanted))
    print(f"{'minutes':<10}" + "".join(f"{results[n][1]/60:>14.1f}" for n in wanted))

    best = max(wanted, key=lambda n: prf(micro[n])[2])
    print(f"\nbest: {best} ({prf(micro[best])[2]:.3f})")
    print("Compare against the drum-only ceiling reported by benchmark_mdb.py (0.882).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
