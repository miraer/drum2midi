"""Predicting hi-hat pedal position (CC4) - the feature ReStem 2 Pro advertises.

GMD stores continuous pedal position in CC4 (0..90). This collects hi-hat stem features
at every hit and trains a regressor. Evaluation runs on tracks the model never saw,
using GMD's official splits.

    python train_cc4.py --build --limit 60    # collect features and train
    python train_cc4.py                       # train on existing features
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import pickle
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
GMD = ROOT / "gmd" / "groove"
LARSNET_DIR = ROOT / "larsnet"
DATA = ROOT / "bench" / "cc4_features.npz"
MODELS = ROOT / "models"
SR = 44100
HAT_PITCHES = {42, 22, 46, 26, 44}


@contextlib.contextmanager
def _in_dir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def cc4_at(mid_path: Path):
    """Time series of (time, value) for CC4."""
    import mido
    mf = mido.MidiFile(str(mid_path))
    tempo, ticks, out = 500000, 0, []
    ppq = mf.ticks_per_beat
    for msg in mido.merge_tracks(mf.tracks):
        ticks += msg.time
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "control_change" and msg.control == 4:
            out.append((ticks / ppq * tempo / 1e6, msg.value))
    return out


def build(limit: int) -> None:
    import pretty_midi
    import soundfile as sf
    import torch
    sys.path.insert(0, str(ROOT))
    from features import FEATURE_NAMES, features_for

    rows = [r for r in csv.DictReader(open(GMD / "info.csv", encoding="utf-8"))
            if r.get("audio_filename") and (GMD / r["audio_filename"]).exists()
            and 5 <= float(r["duration"]) <= 60]
    by_split = {s: [x for x in rows if x["split"] == s] for s in ("train", "test", "validation")}
    chosen = (by_split["train"][: int(limit * 0.6)]
              + by_split["test"][: int(limit * 0.3)]
              + by_split["validation"][: int(limit * 0.1)])
    print(f"tracks: {len(chosen)}")

    X, y, spl, trk = [], [], [], []
    sys.path.insert(0, str(LARSNET_DIR))
    with _in_dir(LARSNET_DIR):
        from larsnet import LarsNet
        net = LarsNet(wiener_filter=True, wiener_exponent=1.0, device="cpu",
                      config="config.yaml")
        for i, r in enumerate(chosen, 1):
            cc = cc4_at(GMD / r["midi_filename"])
            if not cc:
                continue
            cc_t = np.array([t for t, _ in cc])
            cc_v = np.array([v for _, v in cc], dtype=float)

            pm = pretty_midi.PrettyMIDI(str(GMD / r["midi_filename"]))
            hits = sorted(float(n.start) for inst in pm.instruments
                          for n in inst.notes if n.pitch in HAT_PITCHES)
            if len(hits) < 8:
                continue

            audio, sr = sf.read(str(GMD / r["audio_filename"]), always_2d=True)
            audio = audio.T.astype(np.float32)
            if audio.shape[0] == 1:
                audio = np.vstack([audio, audio])
            with torch.no_grad():
                stems = net.separate_wiener(torch.from_numpy(audio))
            hat = stems["hihat"].cpu().numpy()
            hat = hat.mean(axis=0) if hat.ndim > 1 else hat

            feats = features_for(hat, hits, audio.mean(axis=0))
            # the CC4 value in effect just before the hit
            idx = np.searchsorted(cc_t, hits, side="right") - 1
            idx = np.clip(idx, 0, len(cc_v) - 1)
            X.append(feats)
            y.append(cc_v[idx])
            spl += [r["split"]] * len(hits)
            trk += [r["id"]] * len(hits)
            print(f"  [{i:2d}/{len(chosen)}] {r['id']:<34} {len(hits):>4} hi-hat hits")
    sys.path.remove(str(LARSNET_DIR))

    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA, X=np.vstack(X), y=np.concatenate(y),
                        split=np.array(spl), track=np.array(trk),
                        names=np.array(FEATURE_NAMES))
    print(f"\nsaved {sum(len(a) for a in X)} hits -> {DATA}")


def train() -> int:
    from sklearn.ensemble import HistGradientBoostingRegressor

    d = np.load(DATA, allow_pickle=True)
    X, y, split, track = d["X"], d["y"], d["split"], d["track"]
    tr, te = split == "train", split == "test"
    print(f"train {tr.sum()} hits, test {te.sum()}")
    print(f"CC4 in the data: {y.min():.0f}..{y.max():.0f}, median {np.median(y):.0f}, "
          f"unique {len(np.unique(y))}")
    if tr.sum() < 200 or te.sum() < 100:
        print("not enough data")
        return 1

    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                          max_depth=6, l2_regularization=1.0,
                                          random_state=0)
    model.fit(X[tr], y[tr])
    pred = np.clip(model.predict(X[te]), 0, 127)

    base = np.full(te.sum(), float(np.median(y[tr])))
    def rmse(p):
        return float(np.sqrt(np.mean((p - y[te]) ** 2)))

    rs = []
    for t in np.unique(track[te]):
        m = track[te] == t
        if m.sum() >= 5 and y[te][m].std() > 0 and pred[m].std() > 0:
            rs.append(float(np.corrcoef(y[te][m], pred[m])[0, 1]))

    print(f"\n{'':<22}{'RMSE':>8}{'r per track':>14}")
    print("-" * 44)
    print(f"{'constant (median)':<22}{rmse(base):>8.2f}{'—':>14}")
    print(f"{'learned model':<22}{rmse(pred):>8.2f}"
          f"{(np.mean(rs) if rs else float('nan')):>14.3f}")
    print(f"\ntracks evaluated: {len(rs)}")

    if rs and np.mean(rs) >= 0.5 and rmse(pred) < rmse(base) * 0.85:
        MODELS.mkdir(parents=True, exist_ok=True)
        with (MODELS / "cc4.pkl").open("wb") as fh:
            pickle.dump({"model": model, "names": list(d["names"])}, fh)
        print(f"-> good enough, saved to {MODELS / 'cc4.pkl'}")
    else:
        print("-> not good enough, not saved")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()
    if args.build or not DATA.exists():
        build(args.limit)
    return train()


if __name__ == "__main__":
    raise SystemExit(main())
