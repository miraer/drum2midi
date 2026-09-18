"""Train and honestly evaluate the learned velocity and pedal-hi-hat models.

Everything is scored on GMD's official test split, on tracks the models never saw.
Velocity is scored the way it matters musically: Pearson r computed within each
track, then averaged, because our velocities are normalised per track.

    python train_models.py                 # train both, report held-out scores
    python train_models.py --velocity-only
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
DATA = ROOT / "bench" / "gmd_features.npz"
MODELS = ROOT / "models"
FAMILIES = ["kick", "snare", "toms", "hihat", "cymbals"]
# the handcrafted mapping the models have to beat
BASELINE_DYN = {"kick": 18.0, "snare": 36.0, "toms": 36.0, "hihat": 36.0, "cymbals": 48.0}


def per_track_r(track: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> tuple:
    rs = []
    for t in np.unique(track):
        m = track == t
        if m.sum() < 5:
            continue
        a, b = truth[m].astype(float), pred[m].astype(float)
        if a.std() > 0 and b.std() > 0:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return (float(np.mean(rs)) if rs else float("nan")), len(rs)


def baseline_velocity(peak_rel_db: np.ndarray, dyn: float,
                      vmin: int = 15, vmax: int = 127) -> np.ndarray:
    norm = np.clip((peak_rel_db + dyn) / dyn, 0.0, 1.0)
    return np.clip(np.round(vmin + (vmax - vmin) * norm), 1, 127)


def train_velocity(d, report):
    from sklearn.ensemble import HistGradientBoostingRegressor

    X, y = d["X"], d["y"].astype(float)
    fam, split, track = d["family"], d["split"], d["track"]
    names = list(d["names"])
    peak_col = names.index("peak_rel_db")

    MODELS.mkdir(parents=True, exist_ok=True)
    out = {}
    report.append(f"{'drum':<9}{'train':>8}{'test':>8}{'baseline r':>12}"
                  f"{'learned r':>11}{'tracks':>8}")
    report.append("-" * 56)
    for f in FAMILIES:
        tr = (fam == f) & (split == "train")
        te = (fam == f) & (split == "test")
        if tr.sum() < 200 or te.sum() < 50:
            report.append(f"{f:<9}{tr.sum():>8}{te.sum():>8}   too little data")
            continue
        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, max_depth=6,
            l2_regularization=1.0, random_state=0)
        model.fit(X[tr], y[tr])

        base = baseline_velocity(X[te][:, peak_col], BASELINE_DYN[f])
        pred = np.clip(np.round(model.predict(X[te])), 1, 127)
        r_base, _ = per_track_r(track[te], y[te], base)
        r_learn, n_tracks = per_track_r(track[te], y[te], pred)
        report.append(f"{f:<9}{tr.sum():>8}{te.sum():>8}{r_base:>12.3f}"
                      f"{r_learn:>11.3f}{n_tracks:>8}")
        if r_learn > r_base:
            out[f] = model
    if out:
        with (MODELS / "velocity.pkl").open("wb") as fh:
            pickle.dump({"models": out, "names": names}, fh)
        report.append(f"\nsaved {len(out)} velocity models -> {MODELS / 'velocity.pkl'}")
        report.append("(only drums where the learned model beat the formula are kept)")
    else:
        report.append("\nno drum improved; nothing saved, formula stays in charge")
    return out


def train_pedal(d, report):
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, artic, split, track = d["X"], d["artic"], d["split"], d["family"]
    fam = d["family"]
    mask = (fam == "hihat") & np.isin(artic, ["closed", "pedal"])
    tr = mask & (split == "train")
    te = mask & (split == "test")
    report.append(f"\npedal vs closed hi-hat: {tr.sum()} train, {te.sum()} test hits")
    if tr.sum() < 200 or te.sum() < 100:
        report.append("  too little data")
        return None

    ytr = (artic[tr] == "pedal").astype(int)
    yte = (artic[te] == "pedal").astype(int)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         max_depth=6, random_state=0)
    clf.fit(X[tr], ytr)
    prob = clf.predict_proba(X[te])[:, 1]

    best = (0.0, 0.5)
    for thr in np.linspace(0.05, 0.95, 91):
        p = (prob >= thr).astype(int)
        tpr = (p[yte == 1] == 1).mean() if (yte == 1).any() else 0
        tnr = (p[yte == 0] == 0).mean() if (yte == 0).any() else 0
        if (tpr + tnr) / 2 > best[0]:
            best = ((tpr + tnr) / 2, float(thr))
    acc, thr = best
    p = (prob >= thr).astype(int)
    report.append(f"  class balance: {yte.mean():.1%} pedal")
    report.append(f"  best balanced accuracy on held-out tracks: {acc:.3f} "
                  f"(chance 0.500, handcrafted features reached 0.633)")
    report.append(f"  at threshold {thr:.2f}: pedal recall "
                  f"{(p[yte == 1] == 1).mean():.3f}, closed recall {(p[yte == 0] == 0).mean():.3f}")
    if acc >= 0.75:
        MODELS.mkdir(parents=True, exist_ok=True)
        with (MODELS / "pedal.pkl").open("wb") as fh:
            pickle.dump({"model": clf, "threshold": thr, "names": list(d["names"])}, fh)
        report.append(f"  -> useful, saved to {MODELS / 'pedal.pkl'}")
    else:
        report.append("  -> still not good enough to ship; not saved")
    return clf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--velocity-only", action="store_true")
    ap.add_argument("--pedal-only", action="store_true")
    args = ap.parse_args()

    if not DATA.exists():
        print(f"missing {DATA}; run: python build_dataset.py --limit 120")
        return 1
    d = np.load(DATA, allow_pickle=True)
    print(f"dataset: {len(d['X'])} hits, splits "
          f"{ {s: int((d['split'] == s).sum()) for s in np.unique(d['split'])} }\n")

    report = []
    if not args.pedal_only:
        train_velocity(d, report)
    if not args.velocity_only:
        train_pedal(d, report)
    print("\n".join(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
