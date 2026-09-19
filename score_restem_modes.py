"""Scores ReStem's own exports against the MDB annotations, one mode at a time.

We discovered that the published comparison was run in Better (Offline) while a
stronger mode was available, which means a commercial product was benchmarked below its
best. Before re-running all 23 tracks, the question is which mode is actually its best
-- and that is a measurement, not a reading of the marketing.

Takes the trigger_events.json files produced per mode and scores each against the hand
annotations with the same matching code and 50 ms window used everywhere else.

    python score_restem_modes.py MusicDelta_Rock_Drum
    python score_restem_modes.py MusicDelta_Rock_Drum --dir bench/restem_modes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from compare_with_restem import CLS, FAMILIES, PRETTY, match, prf, read_ann  # noqa: E402

# ReStem names its stems; map them onto the annotation's families.
STEM_TO_FAMILY = {"kick": "KD", "snare": "SD", "toms": "TT",
                  "hh": "HH", "ride": "CY", "crash": "CY"}


def load_events(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    sr = data.get("sr", 44100)
    out = {f: [] for f in FAMILIES}
    for stem, events in (data.get("stems") or {}).items():
        fam = STEM_TO_FAMILY.get(stem)
        if not fam:
            continue
        for e in events:
            if "sample" in e:
                out[fam].append(e["sample"] / sr)
    return {f: np.array(sorted(v)) for f, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("track")
    ap.add_argument("--dir", type=Path, default=ROOT / "bench" / "restem_modes")
    ap.add_argument("--stored", type=Path, default=ROOT / "restem_events")
    args = ap.parse_args()

    ann_path = CLS / f"{args.track.replace('_Drum', '')}_class.txt"
    if not ann_path.exists():
        print(f"no annotation at {ann_path}")
        return 1
    ann = read_ann(ann_path)
    ref = {f: np.array(sorted(t for t, l in ann if l == f)) for f in FAMILIES}

    runs = []
    stored = args.stored / f"{args.track}.json"
    if stored.exists():
        runs.append(("stored (17 Sep)", stored))
    for path in sorted(args.dir.glob(f"{args.track}.*.json")):
        runs.append((path.stem.split(".", 1)[1], path))
    if not runs:
        print(f"no exports found for {args.track}")
        return 1

    print(f"{args.track}   reference: "
          + ", ".join(f"{PRETTY[f]} {len(ref[f])}" for f in FAMILIES if len(ref[f])))
    print(f"\n{'mode':<18}" + "".join(f"{PRETTY[f]:>10}" for f in FAMILIES) + f"{'MICRO':>10}")
    print("-" * (18 + 10 * (len(FAMILIES) + 1)))

    for label, path in runs:
        est = load_events(path)
        cells, tp_t = [], {"tp": 0, "ref": 0, "est": 0}
        for f in FAMILIES:
            r, e = ref[f], est.get(f, np.array([]))
            found, _ = match(list(r), list(e))
            tp = int(np.sum(found)) if len(r) and len(e) else 0
            tp_t["tp"] += tp
            tp_t["ref"] += len(r)
            tp_t["est"] += len(e)
            if len(r) == 0 and len(e) == 0:
                cells.append(f"{'-':>10}")
            else:
                cells.append(f"{prf(tp, len(r), len(e))[2]:>10.3f}")
        micro = prf(tp_t["tp"], tp_t["ref"], tp_t["est"])[2]
        print(f"{label:<18}" + "".join(cells) + f"{micro:>10.3f}")

    print("\nA dash means the annotation has none of that class and the run emitted none.")
    print("One track is not evidence about the modes in general -- it decides which mode\n"
          "to re-run all 23 in, and that re-run is what the comparison will rest on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
