"""Do Better and Best actually render differently, and does their MIDI agree anyway?

The README claimed Better and Best produce byte-identical MIDI. Two different separation
models doing that is an extraordinary claim, and the load-bearing part was never proven:
that the selector was applied at all. A cached result re-emitted would advance the cache
file's timestamp, satisfy the old guard, and look exactly like agreement.

So the test was moved off the MIDI and onto the audio. Two different models must produce
different stems even where the trigger engine lands on the same frames, so:

  stems differ, MIDI identical -> the modes are real and TrigNet genuinely agrees
  stems identical              -> the selector did nothing and everything is one mode
  both differ                  -> the original measurement was simply wrong

This compares two captures made by restem_mode_evidence.ps1, which read the mode from the
window before rendering and refused if it was not the one asked for.

    python compare_mode_evidence.py
    python compare_mode_evidence.py --a bench/mode_evidence/better --b bench/mode_evidence/best
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def load(d: Path) -> dict:
    m = d / "manifest.json"
    if not m.exists():
        raise SystemExit(f"no manifest in {d}")
    # PowerShell's Out-File writes a BOM even for UTF8, which json.loads rejects
    return json.loads(m.read_text(encoding="utf-8-sig", errors="replace"))


def events(d: Path) -> tuple[int, dict[str, int]]:
    p = d / "trigger_events.json"
    data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    per = {k: len(v) for k, v in (data.get("stems") or {}).items()}
    return sum(per.values()), per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="bench/mode_evidence/better")
    ap.add_argument("--b", default="bench/mode_evidence/best")
    args = ap.parse_args()

    da, db = ROOT / args.a, ROOT / args.b
    ma, mb = load(da), load(db)

    print(f"{'':16}{ma['label']:>22}{mb['label']:>22}")
    print("-" * 60)
    print(f"{'mode in window':16}{ma['mode']:>22}{mb['mode']:>22}")
    print(f"{'track':16}{ma['track'][:21]:>22}{mb['track'][:21]:>22}")
    print(f"{'render seconds':16}{ma['render_seconds']:>22}{mb['render_seconds']:>22}")
    same_input = ma["input_sha256"] == mb["input_sha256"]
    print(f"{'same input':16}{str(same_input):>22}")
    if not same_input:
        print("\nDifferent inputs. Nothing below means anything.")
        return 1

    ha = {f["file"]: f["sha256"] for f in ma["files"]}
    hb = {f["file"]: f["sha256"] for f in mb["files"]}

    print("\nstems")
    same = diff = 0
    for name in sorted(set(ha) | set(hb)):
        if name.endswith(".json"):
            continue
        a, b = ha.get(name), hb.get(name)
        if a is None or b is None:
            print(f"  {name:<14} missing from one capture")
            continue
        if a == b:
            print(f"  {name:<14} IDENTICAL")
            same += 1
        else:
            print(f"  {name:<14} differ")
            diff += 1

    ea, pa = events(da)
    eb, pb = events(db)
    json_same = ha.get("trigger_events.json") == hb.get("trigger_events.json")
    print(f"\ntrigger events   {ea} vs {eb}   "
          f"json {'IDENTICAL' if json_same else 'differs'}")
    for k in sorted(set(pa) | set(pb)):
        if pa.get(k, 0) != pb.get(k, 0):
            print(f"  {k:<10} {pa.get(k, 0)} vs {pb.get(k, 0)}")

    print("\nverdict")
    a_name, b_name = ma["label"], mb["label"]
    if diff == 0 and same:
        print(f"  Every stem is byte-identical between {a_name} and {b_name}. Whatever")
        print("  distinguishes those two settings did not reach the renderer, so both")
        print("  captures describe one configuration rather than two.")
    elif diff and json_same:
        print(f"  {diff} of {diff + same} stems differ between {a_name} and {b_name}, and")
        print("  the trigger events are still byte-identical. The settings are real and")
        print("  the separation genuinely changes; the trigger engine lands on exactly")
        print("  the same events regardless.")
    elif diff:
        print(f"  {diff} of {diff + same} stems differ between {a_name} and {b_name}, and")
        print(f"  so do the trigger events: {ea} against {eb}. That setting reaches the")
        print("  transcription, not only the audio.")
        print("  Used as a positive control for a pair that agreed, this is the result")
        print("  that matters: the method detects event-level change where it exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
