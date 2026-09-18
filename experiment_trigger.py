"""Does a stem trigger catch the ghost notes ADTOF misses.

Compared on MDB Drums using the subclass annotations:
  SD   plain snare hit   ADTOF gets 0.979
  SDG  ghost note        ADTOF gets 0.476  <- the target
  SDB  buzz roll         ADTOF gets 0.684

Uses the MDX23C stem cache and the ADTOF activation cache.

    python experiment_trigger.py
    python experiment_trigger.py --sweep
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
MDB = ROOT / "mdbdrums" / "MDB Drums"
AUDIO = MDB / "audio" / "drum_only"
SUB = MDB / "annotations" / "subclass"
STEMS = ROOT / "bench" / "uvr_stems"
ACTS = ROOT / "bench" / "activations"
WINDOW = 0.05
STOCK = [0.22, 0.14, 0.32, 0.22, 0.30]

sys.path.insert(0, str(ROOT))
from trigger import trigger  # noqa: E402

# family -> (ADTOF class index, stem name, subclass labels)
TARGETS = {
    "snare": (1, "snare", {"SD": "normal", "SDG": "ghost", "SDB": "buzz",
                           "SDF": "flam", "SDD": "double", "SDNS": "no snares",
                           "SST": "sidestick"}),
    "kick": (0, "kick", {"KD": "normal"}),
    "hihat": (3, "hihat", {"CHH": "closed", "OHH": "open", "PHH": "pedal"}),
}


def read_sub(path: Path):
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2:
            out.append((float(p[0]), p[1]))
    return sorted(out)


def load_stem(track: str, name: str):
    f = STEMS / track / f"{name}.flac"
    if not f.exists():
        return None
    data, _ = sf.read(str(f), always_2d=True)
    return data.T.astype(np.float32).mean(axis=0)


def pick_adtof(col, thr):
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    p = NotePeakPickingProcessor(threshold=float(thr), pre_avg=0.1, post_avg=0.01,
                                 pre_max=0.02, post_max=0.01, combine=0.02, fps=100)
    return np.array([t for t, _ in p.process(col)])


def matched(ref_times, est):
    """For each reference note: was it found."""
    found = np.zeros(len(ref_times), dtype=bool)
    used = set()
    for i, t in enumerate(ref_times):
        best, dist = None, WINDOW
        for j, e in enumerate(est):
            if j in used:
                continue
            d = abs(e - t)
            if d <= dist:
                best, dist = j, d
        if best is not None:
            used.add(best)
            found[i] = True
    return found


def load_tracks():
    out = []
    for wav in sorted(AUDIO.glob("*.wav")):
        ann = SUB / f"{wav.stem.replace('_Drum', '')}_subclass.txt"
        act = ACTS / f"{wav.stem}.npy"
        if ann.exists() and act.exists() and (STEMS / wav.stem).is_dir():
            out.append((wav, read_sub(ann), np.load(act)))
    return out


def evaluate(tracks, family, sens, holdoff=0.020, min_jump=1.5, floor=-55.0):
    ci, stem_name, labels = TARGETS[family]
    got_t = Counter(); got_a = Counter(); got_f = Counter(); tot = Counter()
    est_n = trig_n = fuse_n = 0
    tp = {"adtof": 0, "trigger": 0, "fuse": 0}
    for wav, ann, act in tracks:
        stem = load_stem(wav.stem, stem_name)
        if stem is None:
            continue
        adt = pick_adtof(act[:, ci], STOCK[ci])
        trg = trigger(stem, sensitivity=sens, holdoff=holdoff,
                      min_jump_db=min_jump, noise_floor_db=floor)
        extra = [t for t in trg
                 if not len(adt) or np.min(np.abs(adt - t)) > WINDOW]
        fuse = np.array(sorted(list(adt) + extra))
        est_n += len(adt); trig_n += len(trg); fuse_n += len(fuse)

        ref = [(t, labels[l]) for t, l in ann if l in labels]
        if not ref:
            continue
        times = np.array([t for t, _ in ref])
        kinds = [k for _, k in ref]
        for name, est, sink in (("adtof", adt, got_a), ("trigger", trg, got_t),
                                ("fuse", fuse, got_f)):
            f = matched(times, est)
            tp[name] += int(f.sum())
            for k, ok in zip(kinds, f):
                sink[k] += int(ok)
        for k in kinds:
            tot[k] += 1
    counts = {"adtof": est_n, "trigger": trig_n, "fuse": fuse_n}
    return tot, got_a, got_t, got_f, counts, tp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--fusion", action="store_true")
    ap.add_argument("--family", default="snare", choices=list(TARGETS))
    ap.add_argument("--sens", type=float, default=2.5)
    ap.add_argument("--floor", type=float, default=-55.0)
    args = ap.parse_args()

    tracks = load_tracks()
    print(f"tracks: {len(tracks)}\n")
    if not tracks:
        return 1

    if args.fusion:
        print("FUSION: ADTOF + precise trigger (only notes ADTOF missed")
        print("are added)\n")
        print(f"{'sens':>6}{'floor':>7}{'ghost R':>9}{'overall R':>9}{'P':>8}"
              f"{'F1':>8}{'notes':>8}{'added':>9}")
        print("-" * 64)
        tot, ga, gt, gf, counts, tp = evaluate(tracks, args.family, 2.5)
        ref_n = sum(tot.values())
        p = tp["adtof"] / max(counts["adtof"], 1)
        r = tp["adtof"] / max(ref_n, 1)
        print(f"{'ADTOF':>13}{ga.get('ghost',0)/max(tot.get('ghost',1),1):>9.3f}"
              f"{r:>9.3f}{p:>8.3f}{2*p*r/max(p+r,1e-9):>8.3f}"
              f"{counts['adtof']:>8}{'—':>9}")
        for floor in (-45.0, -38.0, -32.0):
            for sens in (3.5, 5.0, 6.5):
                tot, ga, gt, gf, counts, tp = evaluate(tracks, args.family, sens,
                                                       floor=floor)
                ref_n = sum(tot.values())
                p = tp["fuse"] / max(counts["fuse"], 1)
                r = tp["fuse"] / max(ref_n, 1)
                print(f"{sens:>6.1f}{floor:>7.0f}"
                      f"{gf.get('ghost',0)/max(tot.get('ghost',1),1):>9.3f}"
                      f"{r:>9.3f}{p:>8.3f}{2*p*r/max(p+r,1e-9):>8.3f}"
                      f"{counts['fuse']:>8}{counts['fuse']-counts['adtof']:>9}")
        return 0

    if args.sweep:
        print(f"{'sens':>6}{'floor':>7}{'ghost R':>9}{'normal R':>9}"
              f"{'overall R':>9}{'P':>8}{'F1':>8}{'notes':>8}")
        print("-" * 64)
        for floor in (-55.0, -45.0, -38.0):
            for sens in (1.5, 2.5, 3.5, 5.0):
                tot, ga, gt, gf, counts, tp = evaluate(tracks, args.family, sens,
                                                       floor=floor)
                ref_n = sum(tot.values())
                g = max(tot.get("ghost", 0), 1)
                o = max(tot.get("normal", 0), 1)
                p = tp["trigger"] / max(counts["trigger"], 1)
                r = tp["trigger"] / max(ref_n, 1)
                f1 = 2 * p * r / max(p + r, 1e-9)
                print(f"{sens:>6.1f}{floor:>7.0f}{gt.get('ghost',0)/g:>9.3f}"
                      f"{gt.get('normal',0)/o:>9.3f}{r:>9.3f}{p:>8.3f}"
                      f"{f1:>8.3f}{counts['trigger']:>8}")
        # for comparison ADTOF
        tot, ga, gt, gf, counts, tp = evaluate(tracks, args.family, 2.5)
        ref_n = sum(tot.values())
        p = tp["adtof"] / max(counts["adtof"], 1)
        r = tp["adtof"] / max(ref_n, 1)
        print("-" * 64)
        print(f"{'ADTOF':>13}{gt and ga.get('ghost',0)/max(tot.get('ghost',1),1):>9.3f}"
              f"{ga.get('normal',0)/max(tot.get('normal',1),1):>9.3f}{r:>9.3f}"
              f"{p:>8.3f}{2*p*r/max(p+r,1e-9):>8.3f}{counts['adtof']:>8}")
        return 0

    for family in ("snare", "kick", "hihat"):
        tot, ga, gt, gf, counts, tp = evaluate(tracks, family, 2.5)
        ref_n = sum(tot.values())
        print("=" * 72)
        print(f"{family.upper()}   reference hits {ref_n}")
        for name, key in (("ADTOF", "adtof"), ("trigger", "trigger"),
                          ("fusion", "fuse")):
            p = tp[key] / max(counts[key], 1)
            r = tp[key] / max(ref_n, 1)
            print(f"   {name:<12} notes {counts[key]:>6}  P={p:.3f} R={r:.3f} "
                  f"F1={2*p*r/max(p+r,1e-9):.3f}")
        print("-" * 72)
        print(f"{'hit type':<12}{'total':>7}{'ADTOF':>9}{'trigger':>10}{'fusion':>10}")
        for k in sorted(tot, key=lambda x: -tot[x]):
            n = tot[k]
            if n < 10:
                continue
            print(f"{k:<12}{n:>7}{ga[k]/n:>9.3f}{gt[k]/n:>10.3f}{gf[k]/n:>10.3f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
