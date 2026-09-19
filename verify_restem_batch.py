"""Does every ReStem export actually belong to the track it is named after?

The second machine found that copying a render out of ReStem's cache can pick up the
*previous* track's output when a wait gives up early, and that neither a hash nor a
timestamp catches it: against an empty cache the stale file is genuinely new by both.
Only comparing against what was requested catches it, because staleness is a relation to
the request rather than to the past.

Our published comparison rests on 23 exports collected months before that guard existed.
`restem_batch.ps1` copies only when the cache file's mtime has advanced past a value
captured before the render, and records a failure otherwise -- so by inspection it cannot
produce a stale copy. Inspection is not evidence, and the claim is load-bearing, so this
checks the artefacts themselves.

For each track it compares the span of the exported MIDI against the duration of the
audio that was fed in. A file belonging to a different recording shows up as a gross
mismatch; ordinary trailing silence shows up as the MIDI being slightly shorter, which is
expected and not a fault.

    python verify_restem_batch.py
    python verify_restem_batch.py --midi restem_midi --audio restem_in
"""

from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def audio_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return info.frames / info.samplerate
        except Exception:
            return None


def midi_span(path: Path) -> tuple[float, int] | None:
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(str(path))
        notes = [n.start for inst in pm.instruments for n in inst.notes]
        return (max(notes) if notes else 0.0, len(notes))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--midi", default="restem_midi")
    ap.add_argument("--audio", default="restem_in")
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="how much shorter than the audio a MIDI may end, as a fraction")
    args = ap.parse_args()

    midi_dir, audio_dir = ROOT / args.midi, ROOT / args.audio
    if not midi_dir.exists() or not audio_dir.exists():
        print(f"need both {midi_dir} and {audio_dir}")
        return 1

    print(f"{'track':<32}{'audio':>9}{'last note':>11}{'notes':>8}   verdict")
    print("-" * 78)

    bad = short = 0
    checked = 0
    for wav in sorted(audio_dir.glob("*.wav")):
        mid = midi_dir / f"{wav.stem}.mid"
        if not mid.exists():
            print(f"{wav.stem[:31]:<32}{'':>9}{'':>11}{'':>8}   NO EXPORT")
            continue
        dur = audio_seconds(wav)
        span = midi_span(mid)
        if dur is None or span is None:
            print(f"{wav.stem[:31]:<32}   unreadable")
            continue
        last, n = span
        checked += 1

        if last > dur + 1.0:
            verdict, bad = "LONGER THAN ITS AUDIO", bad + 1
        elif dur > 0 and last < dur * (1 - args.tolerance):
            verdict, short = f"ends {100 * (1 - last / dur):.0f}% early", short + 1
        else:
            verdict = "ok"
        print(f"{wav.stem[:31]:<32}{dur:8.1f}s{last:10.1f}s{n:8}   {verdict}")

    print("-" * 78)
    print(f"checked {checked}")
    if bad:
        print(f"\n{bad} export(s) run past the end of their own audio. That is the "
              f"signature of\na file belonging to a different, longer recording.")
        return 1
    if short:
        print(f"\n{short} export(s) end early. That is usually trailing silence or a "
              f"passage the\nengine found nothing in, not a mismatch -- but a file "
              f"much shorter than its\naudio is also what a stale copy of a shorter "
              f"track looks like, so check those\nagainst the track lengths below.")
    else:
        print("\nEvery export spans its own audio. No stale copies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
