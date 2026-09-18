"""Learns ride vs crash from the separated stems, instead of comparing two peaks.

ADTOF emits a single cymbal class. The pipeline currently decides ride or crash with one
rule -- "the ride stem is 1.2x louder than the crash stem at this instant" -- which
reaches 0.917 accuracy on MDB's subclass labels while ReStem 2 Pro reaches 0.953. That
gap is the largest one left that is clearly our own doing rather than a data shortage.

A ride and a crash differ in more than loudness. A ride is a defined stick attack on a
heavy cymbal: bright, short, concentrated. A crash is a broadband wash that takes far
longer to decay. So this extracts a handful of features describing exactly that, and
lets a small model weigh them.

Evaluation is grouped by track: every hit from one recording goes entirely into train
or entirely into test. Splitting hits at random would let the model memorise a specific
cymbal and report a score it cannot reproduce on new material.

    python train_ride.py                 # build, train, compare against the heuristic
    python train_ride.py --folds 5
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MDB = ROOT / "mdbdrums" / "MDB Drums"
AUDIO = MDB / "audio" / "drum_only"
SUB = MDB / "annotations" / "subclass"
STEMS = ROOT / "bench" / "uvr_stems"
ACTS = ROOT / "bench" / "activations"
SR = 44100
WINDOW = 0.05

from drum2midi import DEFAULT_THRESHOLDS, _mono, peak_at  # noqa: E402

RIDE_LABELS = {"RDC", "RDB"}
CRASH_LABELS = {"CRC", "CHC", "SPC"}

FEATURES = (
    "ride_peak", "crash_peak", "log_ratio",
    "ride_decay", "crash_decay",
    "ride_centroid", "crash_centroid",
    "ride_flatness", "crash_flatness",
    "ride_rolloff", "crash_rolloff",
    "decay_ratio", "sustain_ratio",
)


def read_sub(path: Path) -> list:
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2:
            out.append((float(p[0]), p[1]))
    return sorted(out)


def load_stem(track: str, name: str):
    import soundfile as sf
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


def _decay(sig: np.ndarray, t: float, drop_db: float = 15.0,
           limit: float = 0.5) -> float:
    """Seconds for the hit to fall drop_db below its own peak. Crashes ring longer."""
    a = int(t * SR)
    b = min(a + int(limit * SR), len(sig))
    if b - a < int(0.02 * SR):
        return limit
    seg = np.abs(sig[a:b])
    win = int(0.004 * SR)
    env = np.convolve(seg, np.ones(win) / win, mode="same")
    peak = env.max()
    if peak < 1e-6:
        return limit
    target = peak * (10 ** (-drop_db / 20))
    after = np.where(env[np.argmax(env):] < target)[0]
    return float(after[0] / SR) if len(after) else limit


def _spectral(sig: np.ndarray, t: float, length: float = 0.12) -> tuple:
    """Centroid, flatness and 85% rolloff of the hit, in normalised units."""
    a = int(t * SR)
    b = min(a + int(length * SR), len(sig))
    if b - a < 256:
        return 0.0, 0.0, 0.0
    seg = sig[a:b] * np.hanning(b - a)
    spec = np.abs(np.fft.rfft(seg)) + 1e-10
    freqs = np.fft.rfftfreq(len(seg), 1 / SR)
    total = spec.sum()
    centroid = float((spec * freqs).sum() / total / (SR / 2))
    flatness = float(np.exp(np.mean(np.log(spec))) / (total / len(spec)))
    cumulative = np.cumsum(spec)
    idx = int(np.searchsorted(cumulative, 0.85 * total))
    rolloff = float(freqs[min(idx, len(freqs) - 1)] / (SR / 2))
    return centroid, flatness, rolloff


def features_at(ride: np.ndarray, crash: np.ndarray, t: float) -> np.ndarray:
    rp, cp = peak_at(ride, t), peak_at(crash, t)
    r_dec, c_dec = _decay(ride, t), _decay(crash, t)
    r_cen, r_flat, r_roll = _spectral(ride, t)
    c_cen, c_flat, c_roll = _spectral(crash, t)
    return np.array([
        rp, cp, float(np.log((rp + 1e-8) / (cp + 1e-8))),
        r_dec, c_dec, r_cen, c_cen, r_flat, c_flat, r_roll, c_roll,
        float(r_dec / (c_dec + 1e-6)),
        float((rp + 1e-8) / (rp + cp + 1e-8)),
    ], dtype=np.float32)


def nearest(times: np.ndarray, t: float):
    if not len(times):
        return None
    i = int(np.argmin(np.abs(times - t)))
    return i if abs(times[i] - t) <= WINDOW else None


def build() -> tuple:
    """Returns features, labels (1 = ride), the track, and the current rule's answer."""
    X, y, groups, heuristic = [], [], [], []
    tracks = sorted(AUDIO.glob("*.wav"))
    print(f"scanning {len(tracks)} tracks")
    for wav in tracks:
        ann_path = SUB / f"{wav.stem.replace('_Drum', '')}_subclass.txt"
        act_path = ACTS / f"{wav.stem}.npy"
        ride = load_stem(wav.stem, "ride")
        crash = load_stem(wav.stem, "cymbals")
        if not ann_path.exists() or not act_path.exists() or ride is None or crash is None:
            continue
        ride, crash = _mono(ride), _mono(crash)
        act = np.load(act_path)
        onsets = pick(act[:, 4], DEFAULT_THRESHOLDS[4])
        if not len(onsets):
            continue
        for t, label in read_sub(ann_path):
            truth = 1 if label in RIDE_LABELS else 0 if label in CRASH_LABELS else None
            if truth is None:
                continue
            i = nearest(onsets, t)
            if i is None:
                continue
            at = float(onsets[i])
            X.append(features_at(ride, crash, at))
            y.append(truth)
            groups.append(wav.stem)
            # the current rule, evaluated on exactly the same hits
            heuristic.append(1 if peak_at(ride, at) > peak_at(crash, at) * 1.2 else 0)
    return (np.array(X), np.array(y), np.array(groups), np.array(heuristic))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--model", default="gb", choices=["gb", "logistic", "tree"],
                    help="gb: gradient boosting; logistic: linear, far fewer parameters; "
                         "tree: a single shallow tree, close to a readable rule")
    ap.add_argument("--save", action="store_true",
                    help="write models/ride.pkl when the model wins")
    args = ap.parse_args()

    if not AUDIO.exists():
        print(f"MDB Drums not found at {AUDIO}\nSee the Install section of README.md.")
        return 1

    X, y, groups, heuristic = build()
    if not len(X):
        print("no labelled cymbal hits found; is the stem cache built? "
              "(python cache_uvr_stems.py)")
        return 1

    n_ride = int(y.sum())
    print(f"\n{len(X)} labelled cymbal hits from {len(set(groups))} tracks "
          f"({n_ride} ride, {len(y) - n_ride} crash)")

    def scores_on(p, truth) -> tuple:
        acc = float((p == truth).mean())
        per = [float((p[truth == k] == k).mean()) if (truth == k).any() else float("nan")
               for k in (0, 1)]
        return acc, float(np.nanmean(per)), per

    def scores(p) -> tuple:
        """Plain accuracy plus the mean of the two per-class rates.

        The classes are imbalanced roughly 6:1 towards ride, so accuracy alone can
        improve while crash detection gets worse. Balanced accuracy cannot.
        """
        return scores_on(p, y)

    base_acc, base_bal, base_per = scores(heuristic)
    print(f"current heuristic: {base_acc:.3f} accuracy, {base_bal:.3f} balanced "
          f"(crash {base_per[0]:.2f}, ride {base_per[1]:.2f})")

    # Before reaching for a model, check whether the one number the rule already has
    # is simply set wrong. ride_peak and crash_peak are columns 0 and 1.
    print("\nmargin sweep (ride_peak > crash_peak * margin):")
    ratio = X[:, 0] / (X[:, 1] + 1e-8)
    best_margin, best_margin_bal = None, -1.0
    for margin in (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0):
        p = (ratio > margin).astype(int)
        a, b, per = scores(p)
        mark = "  <- current" if abs(margin - 1.2) < 1e-6 else ""
        print(f"  {margin:>4.1f}: {a:.3f} accuracy, {b:.3f} balanced "
              f"(crash {per[0]:.2f}, ride {per[1]:.2f}){mark}")
        if b > best_margin_bal:
            best_margin, best_margin_bal = margin, b

    # The sweep above picks and reports on the same hits, which is exactly how I
    # previously fooled myself with the peak-picking thresholds. Choose the margin
    # inside each training fold and score it on the held-out one.
    from sklearn.model_selection import GroupKFold as _GKF

    grid = (0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0)
    honest = np.zeros(len(y), dtype=int)
    chosen = []
    for tr, te in _GKF(n_splits=min(args.folds, len(set(groups)))).split(X, y, groups):
        best_m, best_b = 1.2, -1.0
        for m in grid:
            _, b, _ = scores_on((ratio[tr] > m).astype(int), y[tr])
            if b > best_b:
                best_m, best_b = m, b
        chosen.append(best_m)
        honest[te] = (ratio[te] > best_m).astype(int)
    h_acc, h_bal, h_per = scores(honest)
    print(f"\nmargin chosen per fold {chosen} -> held out: {h_acc:.3f} accuracy, "
          f"{h_bal:.3f} balanced (crash {h_per[0]:.2f}, ride {h_per[1]:.2f})")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    def make_model():
        # 130 crashes across 17 tracks is very little data; the simpler options exist
        # because a boosted ensemble can memorise individual cymbals at that size.
        if args.model == "logistic":
            return make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000,
                                                    class_weight="balanced"))
        if args.model == "tree":
            return DecisionTreeClassifier(max_depth=3, class_weight="balanced",
                                          random_state=0)
        return HistGradientBoostingClassifier(max_iter=220, learning_rate=0.06,
                                              max_depth=4, random_state=0)

    folds = min(args.folds, len(set(groups)))
    gkf = GroupKFold(n_splits=folds)
    preds = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in gkf.split(X, y, groups):
        # crashes are outnumbered 6:1, so weight them up or the model learns to
        # answer "ride" and still look accurate
        tr_y = y[train_idx]
        model = make_model()
        if args.model == "gb":
            weights = np.where(tr_y == 0,
                               (tr_y == 1).sum() / max((tr_y == 0).sum(), 1), 1.0)
            model.fit(X[train_idx], tr_y, sample_weight=weights)
        else:
            model.fit(X[train_idx], tr_y)
        preds[test_idx] = model.predict(X[test_idx])

    acc, bal, per = scores(preds)
    print(f"learned ({args.model}): {acc:.3f} accuracy, {bal:.3f} balanced "
          f"(crash {per[0]:.2f}, ride {per[1]:.2f}), {folds}-fold grouped by track")

    def report(name, p):
        print(f"\n{name}")
        print(f"{'truth \\ model':<16}{'crash':>8}{'ride':>8}")
        for truth, label in ((0, "crash"), (1, "ride")):
            row = [(int(((y == truth) & (p == k)).sum())) for k in (0, 1)]
            correct = row[truth] / max(sum(row), 1)
            print(f"{label:<16}{row[0]:>8}{row[1]:>8}   ({correct:.2f} correct)")

    report("heuristic", heuristic)
    report("learned", preds)

    if bal > base_bal:
        print(f"\nThe model wins on balanced accuracy by {bal - base_bal:+.3f}.")
    else:
        print(f"\nThe heuristic holds: balanced {base_bal:.3f} vs {bal:.3f}. "
              f"Keeping the simple rule.")

    if args.save and bal > base_bal:
        final = make_model()
        if args.model == "gb":
            w = np.where(y == 0, (y == 1).sum() / max((y == 0).sum(), 1), 1.0)
            final.fit(X, y, sample_weight=w)
        else:
            final.fit(X, y)
        out = ROOT / "models" / "ride.pkl"
        out.parent.mkdir(exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump({"model": final, "names": list(FEATURES),
                         "cv_accuracy": acc, "cv_balanced": bal,
                         "heuristic_accuracy": base_acc,
                         "heuristic_balanced": base_bal}, fh)
        print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
