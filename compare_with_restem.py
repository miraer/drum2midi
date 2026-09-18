"""Symmetric comparison: this pipeline against ReStem 2 Pro on MDB Drums.

Both sides are scored by the same code, against the same hand annotations, with the same
50 ms tolerance. Up to this point we compared our own measurements against their
marketing - here measurements are compared with measurements.

Getting MIDI out of ReStem: open ReStem 2, enable MIDI Output on every stem tab, run the
files from restem_in\\ and drop the .mid files into restem_midi\\. Names must match the
wav files (MusicDelta_*_Drum.mid).

    python compare_with_restem.py
    python compare_with_restem.py --dir restem_midi
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pretty_midi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
MDB = ROOT / "mdbdrums" / "MDB Drums"
AUDIO = MDB / "audio" / "drum_only"
CLS = MDB / "annotations" / "class"
SUB = MDB / "annotations" / "subclass"
WINDOW = 0.05

# MDB class -> the General MIDI notes that stand for it
FAMILY_PITCHES = {
    "KD": (35, 36),
    "SD": (38, 40, 37),
    "TT": (41, 43, 45, 47, 48, 50),
    "HH": (42, 44, 46, 26, 22),
    "CY": (49, 51, 52, 55, 57, 53, 59),
}
FAMILIES = ["KD", "SD", "TT", "HH", "CY"]
PRETTY = {"KD": "kick", "SD": "snare", "TT": "toms", "HH": "hi-hat", "CY": "cymbals"}

# detailed labels for articulations
ARTIC_REF = {
    "ghost": {"SDG"}, "normal": {"SD"}, "buzz": {"SDB"},
    "ride": {"RDC", "RDB"}, "crash": {"CRC", "CHC", "SPC"},
    "closed": {"CHH"}, "open": {"OHH"}, "pedal": {"PHH"},
}
ARTIC_EST = {
    "ride": (51, 59, 53), "crash": (49, 52, 55, 57),
    "closed": (42, 22), "open": (46, 26), "pedal": (44,),
}


def read_ann(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2:
            out.append((float(p[0]), p[1]))
    return sorted(out)


def load_midi(path: Path):
    pm = pretty_midi.PrettyMIDI(str(path))
    return sorted((float(n.start), n.pitch, n.velocity)
                  for inst in pm.instruments for n in inst.notes)


def match(ref_times, est_times):
    found = np.zeros(len(ref_times), dtype=bool)
    hit = np.full(len(est_times), -1, dtype=int)
    used = set()
    for i, t in enumerate(ref_times):
        best, dist = None, WINDOW
        for j, e in enumerate(est_times):
            if j in used:
                continue
            d = abs(e - t)
            if d <= dist:
                best, dist = j, d
        if best is not None:
            used.add(best)
            found[i] = True
            hit[best] = i
    return found, hit


def prf(tp, ref, est):
    p = tp / max(est, 1)
    r = tp / max(ref, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def score_system(name: str, midi_dir: Path, tracks) -> dict:
    fam = {f: {"tp": 0, "ref": 0, "est": 0} for f in FAMILIES}
    artic = Counter()
    artic_tot = Counter()
    cym = Counter()
    hat = Counter()
    missing = []

    for wav, cls_ann, sub_ann in tracks:
        mid = midi_dir / f"{wav.stem}.mid"
        if not mid.exists():
            missing.append(wav.stem)
            continue
        notes = load_midi(mid)

        for f in FAMILIES:
            ref_t = [t for t, l in cls_ann if l == f]
            est_t = [t for t, p, _ in notes if p in FAMILY_PITCHES[f]]
            found, _ = match(ref_t, est_t)
            fam[f]["tp"] += int(found.sum())
            fam[f]["ref"] += len(ref_t)
            fam[f]["est"] += len(est_t)

        # ghost notes: recall by snare subclass
        snare_est = [t for t, p, _ in notes if p in FAMILY_PITCHES["SD"]]
        for kind, labels in (("normal", ARTIC_REF["normal"]),
                             ("ghost", ARTIC_REF["ghost"]),
                             ("buzz", ARTIC_REF["buzz"])):
            ref_t = [t for t, l in sub_ann if l in labels]
            if not ref_t:
                continue
            found, _ = match(ref_t, snare_est)
            artic[kind] += int(found.sum())
            artic_tot[kind] += len(ref_t)

        # ride vs crash
        cy_est = [(t, p) for t, p, _ in notes if p in FAMILY_PITCHES["CY"]]
        for truth, labels in (("ride", ARTIC_REF["ride"]), ("crash", ARTIC_REF["crash"])):
            ref_t = [t for t, l in sub_ann if l in labels]
            if not ref_t:
                continue
            found, hit = match(ref_t, [t for t, _ in cy_est])
            for j, ri in enumerate(hit):
                if ri < 0:
                    continue
                p = cy_est[j][1]
                lab = "ride" if p in ARTIC_EST["ride"] else "crash"
                cym[(truth, lab)] += 1

        # hi-hat articulation
        hh_est = [(t, p) for t, p, _ in notes if p in FAMILY_PITCHES["HH"]]
        for truth, labels in (("closed", ARTIC_REF["closed"]),
                              ("open", ARTIC_REF["open"]),
                              ("pedal", ARTIC_REF["pedal"])):
            ref_t = [t for t, l in sub_ann if l in labels]
            if not ref_t:
                continue
            found, hit = match(ref_t, [t for t, _ in hh_est])
            for j, ri in enumerate(hit):
                if ri < 0:
                    continue
                p = hh_est[j][1]
                lab = ("open" if p in ARTIC_EST["open"] else
                       "pedal" if p in ARTIC_EST["pedal"] else "closed")
                hat[(truth, lab)] += 1

    return {"name": name, "fam": fam, "artic": artic, "artic_tot": artic_tot,
            "cym": cym, "hat": hat, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="restem_midi",
                    help="directory holding ReStem MIDI")
    ap.add_argument("--ours", default="bench/ours_midi",
                    help="directory holding this pipeline's MIDI")
    args = ap.parse_args()

    tracks = []
    for wav in sorted(AUDIO.glob("*.wav")):
        base = wav.stem.replace("_Drum", "")
        c, s = CLS / f"{base}_class.txt", SUB / f"{base}_subclass.txt"
        if c.exists() and s.exists():
            tracks.append((wav, read_ann(c), read_ann(s)))

    ours = ROOT / args.ours
    restem = ROOT / args.dir
    dirs = [d for d in (ours, restem) if d.exists()]
    if not dirs:
        print(f"no MIDI in either {ours} or {restem}")
        return 1

    # a fair comparison needs the exact same track set on both sides
    before = len(tracks)
    tracks = [t for t in tracks
              if all((d / f"{t[0].stem}.mid").exists() for d in dirs)]
    print(f"tracks with annotations: {before}, shared by all systems: {len(tracks)}")
    if not tracks:
        print("no tracks processed by both systems")
        return 1

    systems = []
    if ours.exists():
        systems.append(score_system("ours", ours, tracks))
    if restem.exists():
        systems.append(score_system("ReStem", restem, tracks))
    if not systems:
        print("nothing to compare")
        return 1
    for s in systems:
        if s["missing"]:
            print(f"  {s['name']}: no MIDI for {len(s['missing'])} tracks")

    print(f"\n{'='*70}\nONSETS PER INSTRUMENT (mir_eval, 50 ms tolerance)\n{'='*70}")
    head = f"{'instrument':<11}{'reference':>8}"
    for s in systems:
        head += f"{s['name'] + ' F1':>12}{'P':>7}{'R':>7}"
    print(head)
    print("-" * len(head))
    micro = [{"tp": 0, "ref": 0, "est": 0} for _ in systems]
    for f in FAMILIES:
        row = f"{PRETTY[f]:<11}{systems[0]['fam'][f]['ref']:>8}"
        for k, s in enumerate(systems):
            c = s["fam"][f]
            for key in micro[k]:
                micro[k][key] += c[key]
            p, r, f1 = prf(c["tp"], c["ref"], c["est"])
            row += f"{f1:>12.3f}{p:>7.3f}{r:>7.3f}"
        print(row)
    print("-" * len(head))
    row = f"{'TOTAL':<11}{micro[0]['ref']:>8}"
    for k, s in enumerate(systems):
        p, r, f1 = prf(micro[k]["tp"], micro[k]["ref"], micro[k]["est"])
        row += f"{f1:>12.3f}{p:>7.3f}{r:>7.3f}"
    print(row)

    print(f"\n{'='*70}\nGHOST NOTES (recall by snare subclass)\n{'='*70}")
    print(f"{'hit type':<12}{'reference':>8}" + "".join(f"{s['name']:>12}" for s in systems))
    for kind, label in (("normal", "normal"), ("ghost", "ghost"), ("buzz", "buzz")):
        n = systems[0]["artic_tot"][kind]
        if n < 10:
            continue
        print(f"{label:<12}{n:>8}" +
              "".join(f"{s['artic'][kind]/max(s['artic_tot'][kind],1):>12.3f}"
                      for s in systems))

    print(f"\n{'='*70}\nRIDE VS CRASH\n{'='*70}")
    for s in systems:
        tot = sum(s["cym"].values())
        if not tot:
            continue
        ok = s["cym"][("ride", "ride")] + s["cym"][("crash", "crash")]
        print(f"  {s['name']:<8} accuracy {ok/tot:.3f} over {tot} hits "
              f"(ride {s['cym'][('ride','ride')]}/"
              f"{s['cym'][('ride','ride')]+s['cym'][('ride','crash')]}, "
              f"crash {s['cym'][('crash','crash')]}/"
              f"{s['cym'][('crash','crash')]+s['cym'][('crash','ride')]})")

    print(f"\n{'='*70}\nHI-HAT ARTICULATION\n{'='*70}")
    for s in systems:
        tot = sum(s["hat"].values())
        if not tot:
            continue
        ok = sum(s["hat"][(k, k)] for k in ("closed", "open", "pedal"))
        detail = ", ".join(
            f"{k} {s['hat'][(k,k)]}/{sum(s['hat'][(k,e)] for e in ('closed','open','pedal'))}"
            for k in ("closed", "open", "pedal"))
        print(f"  {s['name']:<8} accuracy {ok/tot:.3f} over {tot} hits  ({detail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
