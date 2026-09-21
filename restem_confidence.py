"""What ReStem's own confidence figures can and cannot tell us.

Their `trigger_events.json` carries a `prob` on every event, which is the only view either
machine has into their decision. It is tempting to read it as evidence about their model --
their tom recall is 0.882 against our 0.463 and the obvious question is whether that comes
from a better model or a more liberal cut.

It cannot answer that, and the reason is in the data rather than in an argument: **they
export only the events they accepted.** Nothing below 0.657 appears anywhere in 9 168
events across six stems, which is what a threshold looks like from outside, not what a
model's output distribution looks like. Their raw activations would settle it and we do not
have them.

What the file does support, because it is internal to their own output and needs no
cross-calibration:

  * which of their classes they are least sure of, on the same scale as each other
  * whether their false events are marginal accepts or confident mistakes

Both are reported below. Absolute comparison with our activation values is not, and should
not be attempted from here: two models, two calibrations, and a number near 0.9 in one
says nothing about a number near 0.9 in the other.

    python restem_confidence.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark_mdb import ANN, WINDOW, read_annotation  # noqa: E402

EVENTS = ROOT / "restem_events"
STEM_TO_CLASS = {"kick": "KD", "snare": "SD", "toms": "TT", "hh": "HH",
                 "crash": "CY", "ride": "CY"}


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter
                            ).parse_args()

    if not EVENTS.is_dir():
        print(f"no event JSON in {EVENTS.name}", file=sys.stderr)
        return 1

    per_stem: dict[str, list[float]] = {}
    hit: dict[str, list[float]] = {}
    miss: dict[str, list[float]] = {}
    files = 0
    for path in sorted(EVENTS.glob("*.json")):
        # PowerShell writes these with a BOM, which json.loads will not accept
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        sr = data["sr"]
        ann = ANN / f"{path.stem.replace('_Drum', '')}_class.txt"
        ref = read_annotation(ann) if ann.exists() else None
        files += 1
        for stem, events in data["stems"].items():
            cls = STEM_TO_CLASS.get(stem)
            for e in events:
                p = float(e["prob"])
                per_stem.setdefault(stem, []).append(p)
                if ref is None or cls is None:
                    continue
                r = ref[cls]
                t = e["sample"] / sr
                ok = bool(len(r)) and bool(np.abs(r - t).min() <= WINDOW)
                (hit if ok else miss).setdefault(stem, []).append(p)

    if not per_stem:
        print("no events found", file=sys.stderr)
        return 1

    every = np.concatenate([np.array(v) for v in per_stem.values()])
    print(f"{files} recordings, {len(every)} exported events\n")
    print(f"{'stem':<8}{'events':>8}{'min':>8}{'p10':>8}{'median':>9}{'max':>8}"
          f"{'below 0.5':>11}")
    print("-" * 60)
    for stem in ("kick", "snare", "hh", "toms", "crash", "ride"):
        v = per_stem.get(stem)
        if not v:
            continue
        v = np.array(v)
        print(f"{stem:<8}{len(v):>8}{v.min():>8.3f}{np.percentile(v, 10):>8.3f}"
              f"{np.median(v):>9.3f}{v.max():>8.3f}{np.mean(v < 0.5):>10.0%}")

    print(f"\nlowest probability anywhere: {every.min():.3f}. "
          f"{np.mean(every < 0.6):.0%} of events fall below 0.6.")
    print("That is the shape of an export filter, not of a model's output. It says where")
    print("their cut is in their own units and nothing about what sits underneath it.")

    print(f"\n{'stem':<8}{'correct':>9}{'median':>9}{'wrong':>8}{'median':>9}"
          f"{'  are the wrong ones marginal?':>32}")
    print("-" * 76)
    for stem in ("kick", "snare", "hh", "toms", "crash", "ride"):
        h, m = hit.get(stem), miss.get(stem)
        if not h or not m:
            continue
        h, m = np.array(h), np.array(m)
        floor = float(every.min())
        # a marginal accept would sit near the export floor; a confident mistake does not
        margin = (np.median(m) - floor) / max(np.median(h) - floor, 1e-9)
        verdict = "no, confident" if margin > 0.5 else "yes, near the floor"
        print(f"{stem:<8}{len(h):>9}{np.median(h):>9.3f}{len(m):>8}{np.median(m):>9.3f}"
              f"{verdict:>32}")

    toms = per_stem.get("toms")
    if toms:
        meds = {s: float(np.median(v)) for s, v in per_stem.items() if v}
        weakest = min(meds, key=meds.get)
        print()
        print(f"Their least confident class is **{weakest}** at {meds[weakest]:.3f}, "
              f"against {max(meds.values()):.3f} for their most.")
        if weakest == "toms":
            print("Toms are the class they are least sure of, as they are ours. That is a")
            print("statement about their ranking, not about whose model is better: their")
            print("scale and ours are not the same scale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
