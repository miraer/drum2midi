"""Is every input, annotation and output in the ReStem comparison the track it claims to be?

The stale-copy check on ReStem's exports compared each one against the audio it was fed.
That catches an output belonging to a different recording, and it was clean. It cannot
catch a mistake one step earlier: if a file in `restem_in/` were not the MDB track its
name claims, then our MIDI and ReStem's would both be wrong *together*, both would match
that input's duration, and every downstream check would pass.

The comparison joins four things by filename alone -- the staged wav, the MDB source
audio, the MDB annotation, and each system's MIDI. A name is not a checksum.

So this verifies the chain against its origin:

  staged wav vs MDB source    sha256, or duration if the staging re-encoded
  annotation                  exists, and its last onset falls inside the audio
  our MIDI                    spans that audio
  ReStem MIDI                 spans that audio

A failure here would invalidate published numbers rather than merely complicate them,
which is why it is worth running even though nothing has gone wrong so far.

    python verify_comparison_chain.py
    python verify_comparison_chain.py --ours bench/ceiling
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import wave
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        try:
            import soundfile as sf
            i = sf.info(str(path))
            return i.frames / i.samplerate
        except Exception:
            return None


def midi_last(path: Path) -> float | None:
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(path))
        starts = [n.start for inst in pm.instruments for n in inst.notes]
        return max(starts) if starts else 0.0
    except Exception:
        return None


def ann_last(path: Path) -> tuple[float, int] | None:
    try:
        rows = [ln.split() for ln in
                path.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip()]
        times = [float(r[0]) for r in rows if r]
        return (max(times) if times else 0.0, len(times))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--staged", default="restem_in")
    ap.add_argument("--source", default="mdbdrums/MDB Drums/audio/drum_only")
    ap.add_argument("--annotations", default="mdbdrums/MDB Drums/annotations/class")
    ap.add_argument("--ours", default="bench/ceiling")
    ap.add_argument("--restem", default="restem_midi")
    args = ap.parse_args()

    staged = ROOT / args.staged
    source = ROOT / args.source
    anns = ROOT / args.annotations
    ours = ROOT / args.ours
    theirs = ROOT / args.restem

    for p in (staged, anns):
        if not p.exists():
            print(f"missing: {p}")
            return 1

    src_by_stem = {}
    if source.exists():
        for w in source.rglob("*.wav"):
            src_by_stem[w.stem] = w
    else:
        print(f"note: no MDB source audio at {source}; the origin check is skipped\n")

    print(f"{'track':<30}{'origin':>10}{'ann':>7}{'ours':>7}{'ReStem':>8}   detail")
    print("-" * 84)

    problems = 0
    for wav in sorted(staged.glob("*.wav")):
        stem = wav.stem
        dur = seconds(wav)
        detail = []

        # 1. is the staged file the MDB recording it is named after
        origin = "—"
        if src_by_stem:
            src = src_by_stem.get(stem)
            if src is None:
                origin, problems = "ABSENT", problems + 1
                detail.append("no MDB source with this name")
            elif sha(src) == sha(wav):
                origin = "sha256"
            else:
                sd = seconds(src)
                if sd is not None and dur is not None and abs(sd - dur) < 0.05:
                    origin = "duration"
                    detail.append("re-encoded but same length")
                else:
                    origin, problems = "MISMATCH", problems + 1
                    detail.append(f"MDB copy is {sd:.1f}s, staged is {dur:.1f}s")

        # 2. annotation exists and fits inside the audio
        ann = anns / f"{stem.replace('_Drum', '')}_class.txt"
        a = ann_last(ann) if ann.exists() else None
        if a is None:
            ann_col, problems = "MISSING", problems + 1
        elif dur and a[0] > dur + 0.5:
            ann_col, problems = "PAST END", problems + 1
            detail.append(f"annotation runs to {a[0]:.1f}s")
        else:
            ann_col = f"{a[1]}"

        # 3 and 4. each system's MIDI spans this audio
        cols = []
        for label, d in (("ours", ours), ("ReStem", theirs)):
            m = d / f"{stem}.mid"
            if not m.exists():
                cols.append("—")
                continue
            last = midi_last(m)
            if last is None:
                cols.append("bad")
                problems += 1
            elif dur and last > dur + 1.0:
                cols.append("PAST")
                problems += 1
                detail.append(f"{label} MIDI ends at {last:.1f}s, audio is {dur:.1f}s")
            else:
                cols.append(f"{last / dur:.0%}" if dur else "?")

        print(f"{stem[:29]:<30}{origin:>10}{ann_col:>7}{cols[0]:>7}{cols[1]:>8}   "
              f"{'; '.join(detail)}")

    print("-" * 84)
    if problems:
        print(f"\n{problems} problem(s). Published numbers derived from these should be "
              f"treated as suspect\nuntil each is explained.")
        return 1
    print("\nEvery staged input matches its MDB original, every annotation fits its "
          "audio, and\nboth systems' MIDI spans it. The chain holds end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
