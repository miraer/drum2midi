"""Benchmark drum2midi against the Groove MIDI Dataset.

GMD pairs real human performances with the exact MIDI the drum module produced, so
unlike MDB Drums it carries ground-truth VELOCITY and full articulation detail:
open / closed / pedal hi-hat, and ride separate from crash. Those are precisely the
things the rest of our test suite cannot check.

Caveat: the audio is a Roland TD-11 module output, not a miked acoustic kit. There is
no mic bleed, so separator quality here is optimistic relative to real recordings.

    python benchmark_gmd.py --limit 12
    python benchmark_gmd.py --limit 12 --separator uvr
    python benchmark_gmd.py --rescore
"""

from __future__ import annotations

import argparse
import csv
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
GMD = ROOT / "gmd" / "groove"
WINDOW = 0.05

# Roland TD-11 pitches used by GMD -> the family our pipeline works in
GT_FAMILY = {
    **{p: "kick" for p in (35, 36)},
    **{p: "snare" for p in (38, 40, 37)},
    **{p: "tom" for p in (48, 50, 45, 47, 43, 58)},
    **{p: "hat" for p in (42, 22, 46, 26, 44)},
    **{p: "cym" for p in (49, 55, 57, 52, 51, 59, 53)},
}
# finer articulation labels, for the checks only GMD can support
GT_ARTIC = {
    **{p: "closed" for p in (42, 22)},
    **{p: "open" for p in (46, 26)},
    44: "pedal",
    **{p: "crash" for p in (49, 55, 57, 52)},
    **{p: "ride" for p in (51, 59, 53)},
}

EST_FAMILY = {
    **{p: "kick" for p in (35, 36)},
    **{p: "snare" for p in (38, 40)},
    **{p: "tom" for p in (41, 43, 45, 47, 48, 50)},
    **{p: "hat" for p in (42, 44, 46)},
    **{p: "cym" for p in (49, 51)},
}
EST_ARTIC = {42: "closed", 46: "open", 44: "pedal", 49: "crash", 51: "ride"}
FAMILIES = ["kick", "snare", "hat", "tom", "cym"]


def load_gt(midi_path: Path):
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    out = []
    for inst in pm.instruments:
        for n in inst.notes:
            fam = GT_FAMILY.get(n.pitch)
            if fam:
                out.append((float(n.start), fam, GT_ARTIC.get(n.pitch), int(n.velocity)))
    return sorted(out)


def load_est(midi_path: Path):
    notes = pretty_midi.PrettyMIDI(str(midi_path)).instruments[0].notes
    out = []
    for n in notes:
        fam = EST_FAMILY.get(n.pitch)
        if fam:
            out.append((float(n.start), fam, EST_ARTIC.get(n.pitch), int(n.velocity)))
    return sorted(out)


def match(ref, est, window=WINDOW):
    """Greedy nearest-neighbour pairing within the tolerance window."""
    used, pairs = set(), []
    for r in ref:
        best, dist = None, window
        for i, e in enumerate(est):
            if i in used:
                continue
            d = abs(e[0] - r[0])
            if d <= dist:
                best, dist = i, d
        if best is not None:
            used.add(best)
            pairs.append((r, est[best]))
    return pairs


def pick_tracks(limit, lo=8.0, hi=45.0):
    info = GMD / "info.csv"
    rows = []
    with info.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if not r.get("audio_filename"):
                continue
            try:
                dur = float(r["duration"])
            except (TypeError, ValueError):
                continue
            if lo <= dur <= hi and (GMD / r["audio_filename"]).exists():
                rows.append(r)
    rows.sort(key=lambda r: (r["split"] != "test", r["id"]))
    return rows[:limit]


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--tag", default="default")
    ap.add_argument("-h", "--help", action="store_true")
    args, passthrough = ap.parse_known_args()
    if args.help:
        print(__doc__)
        return 0
    if not (GMD / "info.csv").exists():
        print(f"GMD not found at {GMD}")
        return 1

    tracks = pick_tracks(args.limit)
    outdir = ROOT / "bench" / f"gmd_{args.tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"{len(tracks)} tracks from GMD "
          f"({sum(float(r['duration']) for r in tracks) / 60:.1f} min)\n")

    if not args.rescore:
        t0 = time.time()
        for i, r in enumerate(tracks, 1):
            wav = GMD / r["audio_filename"]
            res = subprocess.run(
                [sys.executable, str(ROOT / "drum2midi.py"), str(wav),
                 "-o", str(outdir / f"{r['id'].replace('/', '_')}.mid"),
                 "--device", "auto"] + passthrough,
                capture_output=True, text=True, encoding="utf-8", errors="replace")
            print(f"  [{i:2d}/{len(tracks)}] {'ok ' if res.returncode == 0 else 'FAIL'} "
                  f"{r['id']}  ({r['style']}, {r['bpm']} bpm)")
        print(f"  inference took {time.time() - t0:.0f}s\n")

    onset = {f: {"tp": 0, "ref": 0, "est": 0} for f in FAMILIES}
    # Velocity is normalised per track (relative to that track's loudest hit of each
    # drum), so pooling hits across tracks would measure something meaningless.
    # Correlate within each track, then average.
    vel = {f: [] for f in FAMILIES}
    hat_cm = {g: {e: 0 for e in ("closed", "open", "pedal")}
              for g in ("closed", "open", "pedal")}
    cym_cm = {g: {e: 0 for e in ("crash", "ride")} for g in ("crash", "ride")}

    for r in tracks:
        est_path = outdir / f"{r['id'].replace('/', '_')}.mid"
        if not est_path.exists():
            continue
        gt = load_gt(GMD / r["midi_filename"])
        est = load_est(est_path)
        for fam in FAMILIES:
            g = [x for x in gt if x[1] == fam]
            e = [x for x in est if x[1] == fam]
            onset[fam]["ref"] += len(g)
            onset[fam]["est"] += len(e)
            pairs = match(g, e)
            onset[fam]["tp"] += len(pairs)
            gv, ev = [], []
            for gr, es in pairs:
                gv.append(gr[3])
                ev.append(es[3])
                if fam == "hat" and gr[2] and es[2]:
                    hat_cm[gr[2]][es[2]] += 1
                if fam == "cym" and gr[2] and es[2]:
                    cym_cm[gr[2]][es[2]] += 1
            gv, ev = np.array(gv, float), np.array(ev, float)
            if gv.size > 4 and gv.std() > 0 and ev.std() > 0:
                vel[fam].append(float(np.corrcoef(gv, ev)[0, 1]))

    print(f"{'drum':<8}{'ref':>7}{'est':>7}{'match':>7}{'prec':>7}{'rec':>7}{'F1':>7}"
          f"{'vel r':>8}{'tracks':>8}")
    print("-" * 65)
    tot = {"tp": 0, "ref": 0, "est": 0}
    for fam in FAMILIES:
        c = onset[fam]
        if c["ref"] == 0:
            continue
        p = c["tp"] / max(c["est"], 1); rc = c["tp"] / max(c["ref"], 1)
        for k in tot:
            tot[k] += c[k]
        rs = vel[fam]
        r_ = float(np.mean(rs)) if rs else float("nan")
        print(f"{fam:<8}{c['ref']:>7}{c['est']:>7}{c['tp']:>7}{p:>7.3f}{rc:>7.3f}"
              f"{2 * p * rc / max(p + rc, 1e-9):>7.3f}{r_:>8.2f}{len(rs):>8}")
    p = tot["tp"] / max(tot["est"], 1); rc = tot["tp"] / max(tot["ref"], 1)
    print("-" * 65)
    print(f"{'MICRO':<8}{tot['ref']:>7}{tot['est']:>7}{tot['tp']:>7}{p:>7.3f}{rc:>7.3f}"
          f"{2 * p * rc / max(p + rc, 1e-9):>7.3f}")
    print("  vel r = Pearson correlation with the module's real velocities, "
          "averaged over tracks")

    def confusion(title, cm, labels):
        total = sum(sum(row.values()) for row in cm.values())
        if not total:
            print(f"\n{title}: no matched hits")
            return
        hits = sum(cm[l][l] for l in labels)
        print(f"\n{title}  (accuracy {hits / total:.3f} over {total} matched hits)")
        print("  truth \\ est   " + "".join(f"{l:>9}" for l in labels))
        for g in labels:
            row = sum(cm[g].values())
            print(f"  {g:<13}" + "".join(f"{cm[g][e]:>9}" for e in labels)
                  + (f"   ({cm[g][g] / row:.2f} correct)" if row else ""))

    confusion("HI-HAT articulation", hat_cm, ["closed", "open", "pedal"])
    confusion("CYMBAL articulation", cym_cm, ["crash", "ride"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
