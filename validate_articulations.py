"""Articulation checks on real miked recordings from MDB Drums.

Until now ride/crash, pedal hi-hat and tom splitting were only verified on synthetic
audio or on GMD's electronic kits. MDB ships detailed subclass annotations taken from
actual recordings:

  SDG  snare ghost notes    LFT/MHT/HFT  individual toms
  PHH  pedal hi-hat         RDC/CRC      ride vs crash

Uses the MDX23C stem cache and the activation cache, so it runs quickly.

    python validate_articulations.py
"""

from __future__ import annotations

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
SR = 44100
WINDOW = 0.05
STOCK = [0.22, 0.14, 0.32, 0.22, 0.30]
TOM_PCT, TOM_FLOOR = 98.5, 0.25

sys.path.insert(0, str(ROOT))
from drum2midi import (_mono, peak_at, split_hats, split_toms, split_ride,  # noqa: E402
                       ADTOF_HAT, ADTOF_TOM)


def read_sub(path: Path) -> list:
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
    return data.T.astype(np.float32)


def pick(col, thr):
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    p = NotePeakPickingProcessor(threshold=float(thr), pre_avg=0.1, post_avg=0.01,
                                 pre_max=0.02, post_max=0.01, combine=0.02, fps=100)
    return np.array([t for t, _ in p.process(col)])


def nearest(times, t, tol=WINDOW):
    if not len(times):
        return None
    i = int(np.argmin(np.abs(np.asarray(times) - t)))
    return i if abs(times[i] - t) <= tol else None


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    tracks = []
    for wav in sorted(AUDIO.glob("*.wav")):
        ann = SUB / f"{wav.stem.replace('_Drum', '')}_subclass.txt"
        act = ACTS / f"{wav.stem}.npy"
        if ann.exists() and act.exists() and (STEMS / wav.stem).is_dir():
            tracks.append((wav, read_sub(ann), np.load(act)))
    print(f"tracks with subclass labels, stems and activations: {len(tracks)}\n")
    if not tracks:
        return 1

    # --------------------------------------------------------------- ghost notes
    print("=" * 70)
    print("SNARE GHOST NOTES ON REAL RECORDINGS (SDG vs plain SD)")
    print("=" * 70)
    got = Counter(); tot = Counter()
    for wav, ann, act in tracks:
        est = pick(act[:, 1], STOCK[1])
        for t, label in ann:
            if label in ("SD", "SDG", "SDB"):
                key = {"SD": "normal", "SDG": "ghost", "SDB": "buzz"}[label]
                tot[key] += 1
                if nearest(est, t) is not None:
                    got[key] += 1
    print(f"{'hit type':<12}{'total':>8}{'found':>10}{'recall':>10}")
    print("-" * 40)
    for k in ("normal", "ghost", "buzz"):
        if tot[k]:
            print(f"{k:<12}{tot[k]:>8}{got[k]:>10}{got[k]/tot[k]:>10.3f}")

    # ----------------------------------------------------------- ride and crash
    print("\n" + "=" * 70)
    print("RIDE VS CRASH (RDC/RDB vs CRC/CHC) - MDX23C stems")
    print("=" * 70)
    cm = Counter()
    for wav, ann, act in tracks:
        crash_stem = load_stem(wav.stem, "cymbals")
        ride_stem = load_stem(wav.stem, "ride")
        if crash_stem is None or ride_stem is None:
            continue
        est = pick(act[:, 4], STOCK[4])
        events = [(float(t), 49, 100) for t in est]
        split_ride(events, {"cymbals": crash_stem, "ride": ride_stem})
        times = [e[0] for e in events]
        for t, label in ann:
            truth = ("ride" if label in ("RDC", "RDB") else
                     "crash" if label in ("CRC", "CHC", "SPC") else None)
            if truth is None:
                continue
            i = nearest(times, t)
            if i is None:
                continue
            cm[(truth, "ride" if events[i][1] == 51 else "crash")] += 1
    total = sum(cm.values())
    if total:
        ok = cm[("ride", "ride")] + cm[("crash", "crash")]
        print(f"{'truth \\ model':<18}{'crash':>9}{'ride':>9}")
        for truth in ("crash", "ride"):
            row = cm[(truth, "crash")] + cm[(truth, "ride")]
            acc = f"  ({cm[(truth, truth)]/row:.2f} correct)" if row else ""
            print(f"{truth:<18}{cm[(truth,'crash')]:>9}{cm[(truth,'ride')]:>9}{acc}")
        print(f"accuracy: {ok/total:.3f} over {total} matched hits")

    # ------------------------------------------------------------ pedal hi-hat
    print("\n" + "=" * 70)
    print("HI-HAT: CLOSED / OPEN / PEDAL (CHH / OHH / PHH)")
    print("=" * 70)
    for mode in ("heuristic", "learned model"):
        cm = Counter()
        pedal_model = None
        if mode == "learned model":
            import pickle
            p = ROOT / "models" / "pedal.pkl"
            if not p.exists():
                continue
            with p.open("rb") as fh:
                pedal_model = pickle.load(fh)
        for wav, ann, act in tracks:
            hat = load_stem(wav.stem, "hihat")
            if hat is None:
                continue
            est = pick(act[:, 3], STOCK[3])
            if not len(est):
                continue
            times = list(map(float, est))
            pitches = split_hats(times, hat, 0.30,
                                 detect_pedal=(mode == "heuristic"))
            if pedal_model is not None:
                from features import features_for
                mix, _ = sf.read(str(wav), always_2d=True)
                feats = features_for(_mono(hat), times, mix.T.mean(axis=0))
                prob = pedal_model["model"].predict_proba(feats)[:, 1]
                for i, pr in enumerate(prob):
                    if pitches[i] == ADTOF_HAT and pr >= pedal_model["threshold"]:
                        pitches[i] = 44
            for t, label in ann:
                truth = {"CHH": "closed", "OHH": "open", "PHH": "pedal"}.get(label)
                if truth is None:
                    continue
                i = nearest(times, t)
                if i is None:
                    continue
                est_lab = {42: "closed", 46: "open", 44: "pedal"}.get(pitches[i], "closed")
                cm[(truth, est_lab)] += 1
        total = sum(cm.values())
        if not total:
            continue
        ok = sum(cm[(k, k)] for k in ("closed", "open", "pedal"))
        print(f"\n-- {mode}: accuracy {ok/total:.3f} over {total} hits")
        print(f"{'truth \\ model':<18}" + "".join(f"{k:>9}" for k in ("closed", "open", "pedal")))
        for truth in ("closed", "open", "pedal"):
            row = sum(cm[(truth, e)] for e in ("closed", "open", "pedal"))
            acc = f"  ({cm[(truth,truth)]/row:.2f})" if row else ""
            print(f"{truth:<18}" + "".join(f"{cm[(truth,e)]:>9}"
                                           for e in ("closed", "open", "pedal")) + acc)

    # ------------------------------------------------------------------- toms
    print("\n" + "=" * 70)
    print("TOM SPLITTING BY PITCH (MHT above HFT above LFT)")
    print("=" * 70)
    pairs_ok = pairs_total = 0
    per_track = []
    split_counts = Counter()
    for wav, ann, act in tracks:
        toms = load_stem(wav.stem, "toms")
        if toms is None:
            continue
        thr = max(float(np.percentile(act[:, 2], TOM_PCT)), TOM_FLOOR)
        est = list(map(float, pick(act[:, 2], thr)))
        if len(est) < 2:
            continue
        pitches = split_toms(est, toms, 3, min_ratio=float(sys.argv[1])
                             if len(sys.argv) > 1 else 1.06)
        # MHT = mid-high tom, above both floor toms; HFT = high floor, LFT = low floor
        truth_rank = {"MHT": 2, "HFT": 1, "LFT": 0}
        matched = []
        for t, label in ann:
            if label not in truth_rank:
                continue
            i = nearest(est, t)
            if i is not None:
                matched.append((truth_rank[label], pitches[i]))
        if matched:
            split_counts[len(set(p for _, p in matched))] += 1
        ok = tot_p = same = 0
        for a in range(len(matched)):
            for b in range(a + 1, len(matched)):
                ra, pa = matched[a]
                rb, pb = matched[b]
                if ra == rb:
                    continue
                tot_p += 1
                if pa == pb:
                    same += 1
                elif (ra > rb) == (pa > pb):
                    ok += 1
        if tot_p:
            per_track.append((wav.stem.replace("MusicDelta_", "").replace("_Drum", ""),
                              ok, same, tot_p))
            pairs_ok += ok
            pairs_total += tot_p
    if pairs_total:
        print(f"{'track':<16}{'right order':>16}{'same tom':>20}{'pairs':>6}")
        for name, ok, same, tot_p in per_track:
            print(f"   {name:<13}{ok:>16}{same:>20}{tot_p:>6}")
        decided = pairs_total - sum(s for _, _, s, _ in per_track)
        print(f"\npairs from different toms: {pairs_total}")
        print(f"   model gave both the same tom: "
              f"{pairs_total - decided} ({(pairs_total-decided)/pairs_total:.0%})")
        print(f"   split: {decided}", end="")
        if decided:
            print(f", of which the order is right for {pairs_ok} ({pairs_ok/decided:.2f}, "
                  f"chance would be 0.50)")
        else:
            print(" — the split never fired")
        print(f"\ndistinct toms assigned per track: "
              f"{dict(sorted(split_counts.items()))}")
    else:
        print("not enough annotated toms to check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
