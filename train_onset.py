"""Trains an onset detector that works on separated stems.

The pipeline finds notes with ADTOF, which looks at the mixture. On an isolated stem the
instrument is already known, so the only question left is when it was struck -- a much
easier problem in principle. Running ADTOF on stems was measured and failed badly (tom
F1 0.000), and a handwritten amplitude trigger did not beat it either. This is the third
attempt: a small convolutional network trained on exactly what the separator produces,
artefacts included.

Data comes from build_onset_dataset.py: log-mel spectrograms of five stems from 302
Groove MIDI performances, with onset targets taken from the MIDI the drum module
recorded. The train/test split ships with the data and is by performance, so no take
appears on both sides.

Convolution is what an integrated GPU is good at, so this trains on the GPU even though
the rest of the transcription stage stays on the CPU.

Nothing in the pipeline loads what this writes. The five classes come from ADTOF, and
models/onset_<stem>_<dataset>.pt is read only by whatever experiment is built to score it.
The file name carries the dataset so that two datasets cannot overwrite each other's model.

    python train_onset.py --epochs 6                        # LarsNet data, as built by default
    python train_onset.py --separator uvr --stem kick --epochs 10
    python train_onset.py --data bench/egmd_onset_data --stem toms
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

DATA: Path = ROOT / "bench" / "onset_data"      # set from --separator / --data in main()
STEMS = ["kick", "snare", "toms", "hihat", "cymbals"]
CONTEXT = 7          # frames either side, so the window is 15 frames = 150 ms
FPS = 100


def load(split: str, stem_idx: int, limit: int | None = None):
    """Windows and labels for one stem, drawn from one side of the split."""
    files = sorted(DATA.glob("*.npz"))
    X_all, y_all = [], []
    used = 0
    for path in files:
        with np.load(path, allow_pickle=True) as d:
            if str(d["split"]) != split:
                continue
            X = d["X"][stem_idx].astype(np.float32)     # (T, mels)
            y = d["y"][stem_idx].astype(np.float32)     # (T,)
        if len(X) < 2 * CONTEXT + 1:
            continue
        X_all.append(X)
        y_all.append(y)
        used += 1
        if limit and used >= limit:
            break
    return X_all, y_all


def make_windows(X_list, y_list, negatives_per_positive: float = 3.0, seed: int = 0):
    """Balances the set: every onset frame, plus a sample of the silence between them.

    Onsets occupy a few percent of frames. Training on all of them would let the model
    score 97% by always answering "no onset".
    """
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for X, y in zip(X_list, y_list):
        n = len(X)
        pos = np.where(y >= 0.99)[0]
        pos = pos[(pos >= CONTEXT) & (pos < n - CONTEXT)]
        # anything above 0 is a neighbour frame of a real onset; excluded from negatives
        neg_pool = np.where(y <= 0.0)[0]
        neg_pool = neg_pool[(neg_pool >= CONTEXT) & (neg_pool < n - CONTEXT)]
        take = min(len(neg_pool), int(len(pos) * negatives_per_positive))
        neg = rng.choice(neg_pool, size=take, replace=False) if take else np.empty(0, int)
        for idx, label in ((pos, 1.0), (neg, 0.0)):
            for f in idx:
                xs.append(X[f - CONTEXT: f + CONTEXT + 1])
                ys.append(label)
    if not xs:
        return np.empty((0, 2 * CONTEXT + 1, 1), np.float32), np.empty(0, np.float32)
    return np.stack(xs).astype(np.float32), np.array(ys, np.float32)


def build_net(n_mels: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv2d(1, 16, (3, 3), padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.MaxPool2d((1, 2)),
        nn.Conv2d(16, 32, (3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(32 * ((2 * CONTEXT + 1) // 2) * (n_mels // 4), 64), nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )


def evaluate(model, X, y, device, threshold: float = 0.5) -> dict:
    import torch

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            batch = torch.from_numpy(X[i:i + 4096]).unsqueeze(1).to(device)
            preds.append(torch.sigmoid(model(batch)).squeeze(1).cpu().numpy())
    p = np.concatenate(preds) if preds else np.empty(0)
    hit = (p >= threshold).astype(np.float32)
    tp = float(((hit == 1) & (y == 1)).sum())
    fp = float(((hit == 1) & (y == 0)).sum())
    fn = float(((hit == 0) & (y == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(prec + rec, 1e-9)}


def evaluate_full(model, X_list, y_list, device, thresholds=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """Scores every frame of the test tracks, not a balanced sample of them.

    The balanced set used for training has roughly one onset per four windows. A real
    track has one per thirty or more, so precision measured on the balanced set is
    meaningless -- false positives are counted against a far smaller pool of negatives.
    This is the number that predicts pipeline behaviour.
    """
    import torch

    model.eval()
    probs, truth = [], []
    with torch.no_grad():
        for X, y in zip(X_list, y_list):
            n = len(X)
            if n < 2 * CONTEXT + 1:
                continue
            windows = np.lib.stride_tricks.sliding_window_view(
                X, 2 * CONTEXT + 1, axis=0).transpose(0, 2, 1)
            out = []
            for i in range(0, len(windows), 4096):
                batch = torch.from_numpy(
                    np.ascontiguousarray(windows[i:i + 4096])).unsqueeze(1).to(device)
                out.append(torch.sigmoid(model(batch)).squeeze(1).cpu().numpy())
            probs.append(np.concatenate(out))
            truth.append((y[CONTEXT:n - CONTEXT] >= 0.99).astype(np.float32))
    if not probs:
        return {}
    p = np.concatenate(probs)
    t = np.concatenate(truth)
    rows = {}
    for thr in thresholds:
        hit = p >= thr
        tp = float((hit & (t == 1)).sum())
        fp = float((hit & (t == 0)).sum())
        fn = float((~hit & (t == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        rows[thr] = {"precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / max(prec + rec, 1e-9)}
    rows["_frames"] = len(t)
    rows["_onsets"] = int(t.sum())
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stem", default="all",
                    choices=["all"] + STEMS, help="which stem to train on")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only N performances per split, for a quick trial")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--separator", default="larsnet", choices=["larsnet", "uvr"],
                    help="read the data build_onset_dataset.py wrote for this separator; "
                         "the same default as that script, so the pair runs as documented. "
                         "The onset experiments in the README used --separator uvr")
    ap.add_argument("--data", type=Path, default=None,
                    help="a dataset directory in the same format, e.g. one built from E-GMD")
    args = ap.parse_args()

    global DATA
    from build_onset_dataset import dataset_dir
    DATA = args.data if args.data else dataset_dir(args.separator)
    if not DATA.exists():
        how = ("" if args.data else
               f"\nBuild it with: python build_onset_dataset.py --separator {args.separator}")
        print(f"no dataset at {DATA}{how}")
        return 1

    import torch

    if args.device == "auto":
        device = ("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device
    print(f"device: {device}")

    targets = STEMS if args.stem == "all" else [args.stem]
    results = {}
    for stem in targets:
        idx = STEMS.index(stem)
        t0 = time.time()
        tr_X, tr_y = load("train", idx, args.limit)
        te_X, te_y = load("test", idx, args.limit)
        if not tr_X or not te_X:
            print(f"{stem}: no data")
            continue
        Xtr, ytr = make_windows(tr_X, tr_y)
        Xte, yte = make_windows(te_X, te_y, seed=1)
        if not len(Xtr) or not len(Xte):
            print(f"{stem}: no onsets in the data")
            continue
        n_mels = Xtr.shape[2]
        print(f"\n{stem}: {len(Xtr)} train windows, {len(Xte)} test, "
              f"{int(ytr.sum())} onsets, loaded in {time.time()-t0:.0f}s")

        model = build_net(n_mels).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        order = np.arange(len(Xtr))
        batch = 256

        for epoch in range(args.epochs):
            model.train()
            np.random.default_rng(epoch).shuffle(order)
            total = 0.0
            for i in range(0, len(order), batch):
                sel = order[i:i + batch]
                xb = torch.from_numpy(Xtr[sel]).unsqueeze(1).to(device)
                yb = torch.from_numpy(ytr[sel]).unsqueeze(1).to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
                total += float(loss) * len(sel)
            m = evaluate(model, Xte, yte, device)
            print(f"  epoch {epoch+1}/{args.epochs}  loss {total/len(order):.4f}  "
                  f"test F1 {m['f1']:.3f} (P {m['precision']:.3f} R {m['recall']:.3f})")

        results[stem] = evaluate(model, Xte, yte, device)
        torch.save({"state": model.state_dict(), "n_mels": n_mels,
                    "context": CONTEXT, "stem": stem, "data": DATA.name},
                   ROOT / "models" / f"onset_{stem}_{DATA.name}.pt")

        full = evaluate_full(model, te_X, te_y, device)
        if full:
            print(f"  on complete tracks ({full['_onsets']} onsets in "
                  f"{full['_frames']} frames, {full['_onsets']/full['_frames']:.1%}):")
            for thr in (0.1, 0.3, 0.5, 0.7, 0.9):
                m = full[thr]
                print(f"    threshold {thr:.1f}: F1 {m['f1']:.3f} "
                      f"(P {m['precision']:.3f} R {m['recall']:.3f})")
            best = max((t for t in (0.1, 0.3, 0.5, 0.7, 0.9)),
                       key=lambda t: full[t]["f1"])
            results[stem]["full_f1"] = full[best]["f1"]
            results[stem]["full_threshold"] = best

    if results:
        print(f"\n{'stem':<10}{'balanced F1':>13}{'whole-track F1':>16}{'at threshold':>14}")
        print("-" * 54)
        for stem, m in results.items():
            full = m.get("full_f1")
            thr = m.get("full_threshold")
            print(f"{stem:<10}{m['f1']:>13.3f}"
                  f"{(f'{full:.3f}' if full is not None else '-'):>16}"
                  f"{(f'{thr:.1f}' if thr is not None else '-'):>14}")
        print("\nThe whole-track column is the honest one. Whether it helps the "
              "pipeline\nstill has to be measured end to end by benchmark_mdb.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
