"""Does the transcriber's blind spot explain where it scores badly?

blind_spots.py found that a real song has 18.2% of its audible onsets in a window where
every ADTOF class stays under half its threshold, while four of six MDB tracks have
none at all. That makes the failure material-specific rather than general, and raises
the question this script answers: on the benchmark, do the tracks with more blind
onsets also score worse?

It matters because of an experiment already rejected here. A stem-based onset detector
scored 0.623 against 0.882, because it fired everywhere and buried the good tracks in
false positives. A fallback that only fires where the model is silent is a different
proposition -- on four of six tracks measured it would never fire at all -- but that
argument is only worth making if the blind share actually predicts the damage.

Correlation is not proof of a fix. If the two are unrelated, the fallback idea dies
here and the finding is recorded as another negative result.

    python blind_spots_mdb.py
    python blind_spots_mdb.py --limit 5 --csv bench/blind_mdb.csv
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import blind_spots  # noqa: E402
import drum2midi  # noqa: E402
from benchmark_mdb import ANN, AUDIO, CLASSES, ORDER, read_annotation  # noqa: E402

WINDOW = 0.05
FPS = 100


def blind_share(wav: Path, thr: np.ndarray, window: float = 0.06) -> tuple:
    """Share of audible onsets the model does not react to, and how many there were."""
    times, strengths, duration = blind_spots.onset_peaks(wav)
    if not len(times):
        return 0.0, 0, duration
    act = blind_spots.activations(wav)
    half = int(round(window * FPS))
    ratios = []
    for t in times:
        centre = int(round(t * FPS))
        lo, hi = max(0, centre - half), min(act.shape[0], centre + half + 1)
        if lo >= hi:
            continue
        ratios.append(float((act[lo:hi].max(axis=0) / thr).max()))
    ratios = np.array(ratios)
    return float((ratios < 0.5).mean()), len(ratios), duration


def track_f1(wav: Path, mid: Path) -> float:
    """Mean per-class F1 for one track, matching benchmark_mdb's own definition."""
    import mir_eval
    import pretty_midi

    ann = ANN / f"{wav.stem.replace('_Drum', '')}_class.txt"
    if not ann.exists() or not mid.exists():
        return float("nan")
    ref = read_annotation(ann)
    notes = [n for inst in pretty_midi.PrettyMIDI(str(mid)).instruments for n in inst.notes]
    est = {k: np.array(sorted(n.start for n in notes if n.pitch in pitches))
           for k, pitches in CLASSES.items()}
    scores = []
    for k in ORDER:
        r, e = ref.get(k, np.array([])), est.get(k, np.array([]))
        if not len(r):
            continue
        if not len(e):
            scores.append(0.0)
            continue
        f, _, _ = mir_eval.onset.f_measure(r, e, window=WINDOW)
        scores.append(float(f))
    return float(np.mean(scores)) if scores else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--midi-dir", type=Path, default=ROOT / "bench" / "note36",
                    help="folder of transcriptions to score against")
    ap.add_argument("--csv", type=Path, default=ROOT / "bench" / "blind_mdb.csv")
    args = ap.parse_args()

    if not AUDIO.exists():
        print(f"MDB Drums not found at {AUDIO}")
        return 1

    thr = np.array(drum2midi.DEFAULT_THRESHOLDS, dtype=float)
    wavs = sorted(AUDIO.glob("*.wav"))[:args.limit]
    print(f"{len(wavs)} tracks, thresholds {list(thr)}\n")
    print(f"{'track':<34}{'onsets':>8}{'blind':>8}{'F1':>8}")
    print("-" * 58)

    rows = []
    for wav in wavs:
        share, n, _ = blind_share(wav, thr)
        f1 = track_f1(wav, args.midi_dir / f"{wav.stem}.mid")
        rows.append((wav.stem, n, share, f1))
        name = wav.stem.replace("MusicDelta_", "").replace("_Drum", "")
        print(f"{name:<34}{n:>8}{share:>8.1%}{f1:>8.3f}", flush=True)

    usable = [(s, f) for _, _, s, f in rows if not np.isnan(f)]
    if len(usable) >= 3:
        shares = [s for s, _ in usable]
        f1s = [f for _, f in usable]
        r = float(np.corrcoef(shares, f1s)[0, 1])
        print(f"\ncorrelation between blind share and F1: {r:+.3f}")
        print(f"  blind share: mean {statistics.mean(shares):.1%}, "
              f"max {max(shares):.1%}, {sum(1 for s in shares if s == 0)} tracks at zero")
        clean = [f for s, f in usable if s < 0.02]
        dirty = [f for s, f in usable if s >= 0.02]
        if clean and dirty:
            print(f"  F1 where nothing is blind ({len(clean)} tracks): "
                  f"{statistics.mean(clean):.3f}")
            print(f"  F1 where something is  ({len(dirty)} tracks): "
                  f"{statistics.mean(dirty):.3f}")
            print(f"  difference: {statistics.mean(clean) - statistics.mean(dirty):+.3f}")
        print("\nA negative correlation means blind onsets travel with poor scores,\n"
              "which is what a fallback would have to exploit. Near zero means they do\n"
              "not, and the idea should be dropped.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8") as fh:
            fh.write("track,onsets,blind_share,f1\n")
            for name, n, s, f in rows:
                fh.write(f"{name},{n},{s:.4f},{f:.4f}\n")
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
