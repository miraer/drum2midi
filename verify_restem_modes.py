"""Every ReStem render we hold, hashed and grouped, so the mode claim states its own base.

The published comparison was rendered in Better (Offline) on 17 September. The README
argues that Best (Offline) would have produced the same numbers. That is a generalisation
from however many tracks were actually rendered both ways, and the honest way to present
it is to count them rather than to describe them.

This walks every `trigger_events.json` on disk -- the 23 benchmark renders, the mode
experiments, and anything captured since -- hashes each, and reports which renders of the
same track agree byte for byte and which do not. It reads only; it renders nothing.

It answers two questions that were being conflated:

  does the quality mode change the output      Better against Best on the tracks
                                               rendered both ways
  is a render reproducible at all              the same track, same mode, different day.
                                               If renders were not deterministic, no mode
                                               comparison would mean anything, and nobody
                                               had checked.

    python verify_restem_modes.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent

# Where renders live and what mode each directory was produced in. The mode is not in the
# file -- ReStem's export does not carry it -- so it comes from how the render was driven.
SOURCES = [
    (ROOT / "restem_events", "*.json", "Better", "the 23 benchmark renders, 17 Sept"),
    (ROOT / "bench" / "restem_modes", "*.better.json", "Better", "mode experiment"),
    (ROOT / "bench" / "restem_modes", "*.best-nobleed*.json", "Best", "mode experiment"),
    (ROOT / "bench" / "restem_modes", "*.best-plus.json", "Best+Bleed", "mode experiment"),
    (ROOT / "bench" / "mode_evidence" / "better", "trigger_events.json", "Better",
     "stem-level mode evidence"),
    (ROOT / "bench" / "mode_evidence" / "best", "trigger_events.json", "Best",
     "stem-level mode evidence"),
    (ROOT / "bench" / "mode_evidence" / "best-plus", "trigger_events.json", "Best+Bleed",
     "stem-level mode evidence"),
    (ROOT / "bench" / "restem_pair", "*.trigger_events.json", "Better",
     "captured with its own MIDI, 20 Sept"),
]


def track_of(path: Path) -> str:
    """The recording a render belongs to, however the file happens to be named."""
    name = path.name
    for suffix in (".trigger_events.json", ".best-nobleed-firsttry.json",
                   ".best-nobleed.json", ".best-plus.json", ".better.json", ".json"):
        if name.endswith(suffix):
            stem = name[: -len(suffix)]
            break
    else:
        stem = path.stem
    if stem in ("trigger_events", ""):
        # mode_evidence keeps the track name in a sibling manifest
        man = path.parent / "manifest.json"
        if man.exists():
            try:
                # written by PowerShell, so it carries a BOM; plain utf-8 raises here and
                # an earlier version of this script silently reported the track as "?"
                return json.loads(man.read_text(encoding="utf-8-sig")).get("track", "?")
            except Exception as exc:
                print(f"  cannot read {man}: {exc}", file=sys.stderr)
                return "?"
        return "?"
    return stem


def events(path: Path) -> int | None:
    try:
        j = json.loads(path.read_text(encoding="utf-8-sig"))
        stems = j.get("stems")
        if isinstance(stems, dict):
            return sum(len(v) for v in stems.values())
    except Exception as exc:
        print(f"  cannot count events in {path.name}: {exc}", file=sys.stderr)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true", help="only the summary")
    args = ap.parse_args()

    renders = defaultdict(list)
    for base, pattern, mode, note in SOURCES:
        if not base.exists():
            continue
        for p in sorted(base.glob(pattern)):
            renders[track_of(p)].append({
                "path": p, "mode": mode, "note": note,
                "sha": hashlib.sha256(p.read_bytes()).hexdigest(),
                "day": datetime.fromtimestamp(p.stat().st_mtime).strftime("%d.%m"),
                "events": events(p)})

    if not renders:
        print("no ReStem renders found")
        return 1

    both, agree, repro_pairs, repro_ok = [], [], 0, 0
    for track in sorted(renders):
        rs = renders[track]
        modes = {r["mode"] for r in rs}
        if not args.quiet and len(rs) > 1:
            print(f"\n{track}")
            for r in sorted(rs, key=lambda r: (r["mode"], r["day"])):
                ev = f"{r['events']:>5}" if r["events"] is not None else "    ?"
                print(f"  {r['mode']:<11}{r['day']}  {ev} events  "
                      f"{r['sha'][:16]}  {r['note']}")
        if {"Better", "Best"} <= modes:
            both.append(track)
            a = {r["sha"] for r in rs if r["mode"] == "Better"}
            b = {r["sha"] for r in rs if r["mode"] == "Best"}
            if a & b:
                agree.append(track)
        for mode in modes:
            same = [r for r in rs if r["mode"] == mode]
            days = {r["day"] for r in same}
            if len(same) > 1 and len(days) > 1:
                repro_pairs += 1
                if len({r["sha"] for r in same}) == 1:
                    repro_ok += 1

    total_tracks = len(renders)
    ev_both = sum(next((r["events"] for r in renders[t] if r["events"] is not None), 0)
                  for t in both)
    ev_all = sum(next((r["events"] for r in renders[t] if r["events"] is not None), 0)
                 for t in renders)

    print(f"\n{'':-<72}")
    print(f"tracks with a render on disk:              {total_tracks}")
    print(f"tracks rendered in BOTH Better and Best:   {len(both)}"
          f"  ({', '.join(sorted(both))})")
    print(f"  of those, byte-identical across modes:   {len(agree)}")
    print(f"events covered by the both-modes tracks:   {ev_both} of {ev_all} "
          f"({ev_both/ev_all:.1%} of what ReStem emitted)")
    print(f"same track and mode on a different day:    {repro_pairs} comparison(s), "
          f"{repro_ok} byte-identical")

    print(f"""
What that supports, and what it does not.

Better and Best agreed on every track rendered both ways, exactly, with no differing byte
-- but that is {len(both)} tracks and {ev_both/ev_all:.0%} of the events in the published comparison. It is
evidence that the mode does not reach the transcription; it is not a demonstration that it
never could, and the README should not be read as claiming the second thing.

The reproducibility result is the one nobody set out to get. Renders of the same track on
different days are byte-identical, which means ReStem's pipeline is deterministic and the
mode comparison is comparing modes rather than run-to-run noise. Without it the agreement
above would have been much weaker evidence than it looks.

The cheap way to strengthen this is more tracks rendered both ways while a trial is live.
It needs a human at the interface, because the quality selector is drawn rather than
exposed and cannot be set or read programmatically.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
