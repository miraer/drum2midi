"""Does our conversion of ReStem's trigger JSON match the MIDI ReStem itself exports?

The published comparison scores `restem_midi/`, which `restem_to_midi.py` builds from
ReStem's `trigger_events.json`. The note and CC4 mapping was validated once, on a single
song, against one real export -- 374 events against 374 notes. Everything since has
assumed that mapping generalises.

It need not. ReStem's exporter could round times differently, shape velocity, drop events
below a threshold, or decide articulation from something the JSON does not carry. If it
does, our figures describe TrigNet's internal output rather than what the product actually
writes, and the distinction matters in a letter to the people who wrote it.

ReStem leaves its own export in a temporary directory, one per render, so the originals
from the comparison batch are still on disk. This compares them note for note.

Read-only, and it copies nothing: the same directory holds renders of private material,
which has no business being under a repository.

    python compare_restem_export.py
    python compare_restem_export.py --theirs "C:/path/to/midi-drag" --ours restem_midi
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DEFAULT_THEIRS = Path(os.environ.get("LOCALAPPDATA", "")) / "Temp" / "ReStem 2" / "midi-drag"


def notes(path: Path) -> list[tuple[float, int, int]]:
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    out = [(round(n.start, 4), n.pitch, n.velocity)
           for inst in pm.instruments for n in inst.notes]
    return sorted(out)


def compare(a: list, b: list, tol: float = 0.002) -> dict:
    """Match by time and pitch; report what does not line up."""
    used = [False] * len(b)
    matched = vel_same = 0
    for t, p, v in a:
        for j, (t2, p2, v2) in enumerate(b):
            if used[j] or p2 != p:
                continue
            if abs(t2 - t) <= tol:
                used[j] = True
                matched += 1
                if v2 == v:
                    vel_same += 1
                break
    return {"ours": len(a), "theirs": len(b), "matched": matched,
            "vel_same": vel_same,
            "only_ours": len(a) - matched,
            "only_theirs": len(b) - matched}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theirs", type=Path, default=DEFAULT_THEIRS)
    ap.add_argument("--ours", type=Path, default=ROOT / "restem_midi")
    ap.add_argument("--on", default="17.09",
                    help="only use their exports written on this day, dd.MM")
    args = ap.parse_args()

    if not args.theirs.exists():
        print(f"no such directory: {args.theirs}")
        return 1

    # one export per track, the one written on the batch day
    theirs: dict[str, Path] = {}
    for f in args.theirs.rglob("MusicDelta_*.mid"):
        day = __import__("datetime").datetime.fromtimestamp(
            f.stat().st_mtime).strftime("%d.%m")
        if args.on and day != args.on:
            continue
        stem = f.name.replace("_midi.mid", "")
        prev = theirs.get(stem)
        if prev is None or f.stat().st_mtime > prev.stat().st_mtime:
            theirs[stem] = f

    print(f"their exports from {args.on}: {len(theirs)}")
    print(f"\n{'track':<28}{'ours':>6}{'theirs':>8}{'matched':>9}{'vel=':>7}   note")
    print("-" * 74)

    totals = Counter()
    mismatched = 0
    for stem in sorted(theirs):
        ours_path = args.ours / f"{stem}.mid"
        if not ours_path.exists():
            print(f"{stem[:27]:<28}{'':>6}{'':>8}{'':>9}{'':>7}   we have no MIDI")
            continue
        a, b = notes(ours_path), notes(theirs[stem])
        r = compare(a, b)
        for k, v in r.items():
            totals[k] += v
        note = ""
        if r["only_ours"] or r["only_theirs"]:
            note = f"+{r['only_ours']} ours / +{r['only_theirs']} theirs"
            mismatched += 1
        elif r["vel_same"] != r["matched"]:
            note = f"{r['matched'] - r['vel_same']} velocity differences"
        else:
            note = "identical"
        print(f"{stem[:27]:<28}{r['ours']:>6}{r['theirs']:>8}{r['matched']:>9}"
              f"{r['vel_same']:>7}   {note}")

    print("-" * 74)
    print(f"{'TOTAL':<28}{totals['ours']:>6}{totals['theirs']:>8}"
          f"{totals['matched']:>9}{totals['vel_same']:>7}")

    print()
    if not totals["ours"] and not totals["theirs"]:
        print("Nothing was compared. Every export was skipped, so this says nothing at")
        print("all -- check the day filter and that the names line up with --ours.")
        return 1
    if totals["ours"] == totals["theirs"] == totals["matched"]:
        if totals["vel_same"] == totals["matched"]:
            print("Every note matches in time, pitch and velocity. Our conversion is the")
            print("product's own output, and the published figures describe what ReStem")
            print("writes rather than an interpretation of its internals.")
        else:
            d = totals["matched"] - totals["vel_same"]
            print(f"Every note matches in time and pitch; {d} differ in velocity only.")
            print("Onset scoring is unaffected, since it uses time and class. Any")
            print("velocity claim would need their export rather than our conversion.")
    else:
        print(f"{mismatched} track(s) disagree on which notes exist. The published")
        print("comparison scores our conversion, not their export, and the README")
        print("should say which.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
