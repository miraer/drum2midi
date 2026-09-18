"""Item 2: does running ADTOF on an isolated stem beat running it on the mix?

Toms are our worst class (F1 0.543 on MDB Drums). The transcriber only ever sees the
mix, while a dedicated tom stem is sitting right there. This measures whether feeding
the stem to ADTOF raises the tom class, and whether combining both is better still.

Note ADTOF was trained on full mixes, so a lone stem is out of distribution - that is
exactly what needs testing rather than assuming.

    python experiment_stem_adt.py --limit 10
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import tempfile
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
MDB = ROOT / "mdbdrums" / "MDB Drums"
AUDIO = MDB / "audio" / "drum_only"
ANN = MDB / "annotations" / "class"
LARSNET_DIR = ROOT / "larsnet"
SR = 44100
WINDOW = 0.05

# ADTOF class index -> (MDB label, LarsNet stem)
CLASSES = [(0, "KD", "kick"), (1, "SD", "snare"), (2, "TT", "toms"),
           (3, "HH", "hihat"), (4, "CY", "cymbals")]
DEFAULT_THR = [0.22, 0.14, 0.45, 0.18, 0.34]


@contextlib.contextmanager
def _in_dir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def activations(path: Path, model, device="cpu") -> np.ndarray:
    import torch
    from adtof_pytorch import load_audio_for_model
    with torch.no_grad():
        return model(load_audio_for_model(str(path)).to(device)).cpu().numpy()[0]


def pick(col: np.ndarray, thr: float) -> np.ndarray:
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    p = NotePeakPickingProcessor(threshold=thr, pre_avg=0.1, post_avg=0.01,
                                 pre_max=0.02, post_max=0.01, combine=0.02, fps=100)
    return np.array([t for t, _ in p.process(col)])


def read_ann(path: Path) -> dict:
    out: dict = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2:
            out.setdefault(p[1], []).append(float(p[0]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def score(ref, est) -> tuple:
    if len(ref) == 0:
        return 0, 0, len(est)
    if len(est) == 0:
        return 0, len(ref), 0
    _, p, _ = mir_eval.onset.f_measure(ref, est, window=WINDOW)
    tp = int(round(p * len(est)))
    return tp, len(ref), len(est)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    import torch
    from adtof_pytorch import (calculate_n_bins, create_frame_rnn_model,
                               get_default_weights_path, load_pytorch_weights)

    model = create_frame_rnn_model(calculate_n_bins())
    model.eval()
    w = get_default_weights_path()
    if w and Path(w).exists():
        model = load_pytorch_weights(model, w, strict=False)

    tracks = sorted(AUDIO.glob("*.wav"))[: args.limit]
    variants = ["mix", "stem", "max", "mean"]
    totals = {v: {lbl: {"tp": 0, "ref": 0, "est": 0} for _, lbl, _ in CLASSES}
              for v in variants}

    sys.path.insert(0, str(LARSNET_DIR))
    with _in_dir(LARSNET_DIR):
        from larsnet import LarsNet
        net = LarsNet(wiener_filter=True, wiener_exponent=1.0, device="cpu",
                      config="config.yaml")

        for i, wav in enumerate(tracks, 1):
            ann_path = ANN / f"{wav.stem.replace('_Drum', '')}_class.txt"
            if not ann_path.exists():
                continue
            ref = read_ann(ann_path)

            audio, sr = sf.read(str(wav), always_2d=True)
            audio = audio.T.astype(np.float32)
            if audio.shape[0] == 1:
                audio = np.vstack([audio, audio])
            with torch.no_grad():
                stems = net.separate_wiener(torch.from_numpy(audio))
            stems = {k: v.cpu().numpy() for k, v in stems.items()}

            act_mix = activations(wav, model)
            tmp = Path(tempfile.mkdtemp(prefix="stemadt_"))
            try:
                act_stem = {}
                for _, _, name in CLASSES:
                    s = stems.get(name)
                    if s is None:
                        continue
                    p = tmp / f"{name}.wav"
                    sf.write(str(p), s.T, SR)
                    act_stem[name] = activations(p, model)

                for ci, lbl, name in CLASSES:
                    a_mix = act_mix[:, ci]
                    a_st = act_stem.get(name)
                    if a_st is None:
                        continue
                    a_st = a_st[:, ci]
                    n = min(len(a_mix), len(a_st))
                    cand = {"mix": a_mix[:n], "stem": a_st[:n],
                            "max": np.maximum(a_mix[:n], a_st[:n]),
                            "mean": 0.5 * (a_mix[:n] + a_st[:n])}
                    r = ref.get(lbl, np.array([]))
                    for v in variants:
                        tp, nr, ne = score(r, pick(cand[v], DEFAULT_THR[ci]))
                        totals[v][lbl]["tp"] += tp
                        totals[v][lbl]["ref"] += nr
                        totals[v][lbl]["est"] += ne
            finally:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            print(f"  [{i:2d}/{len(tracks)}] {wav.stem}")

    sys.path.remove(str(LARSNET_DIR))

    def f1(c):
        p = c["tp"] / max(c["est"], 1); r = c["tp"] / max(c["ref"], 1)
        return 2 * p * r / max(p + r, 1e-9)

    print(f"\nADTOF input -> F1 per class ({len(tracks)} MDB tracks)")
    print(f"{'variant':<8}" + "".join(f"{lbl:>10}" for _, lbl, _ in CLASSES) + f"{'MICRO':>10}")
    print("-" * (8 + 10 * (len(CLASSES) + 1)))
    for v in variants:
        micro = {"tp": 0, "ref": 0, "est": 0}
        cells = []
        for _, lbl, _ in CLASSES:
            c = totals[v][lbl]
            for k in micro:
                micro[k] += c[k]
            cells.append(f"{f1(c):>10.3f}")
        print(f"{v:<8}" + "".join(cells) + f"{f1(micro):>10.3f}")
    print("\nmix = current behaviour; stem = ADTOF on the isolated stem;")
    print("max/mean = the two activations combined per frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
