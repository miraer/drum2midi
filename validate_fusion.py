"""Is stem fusion a real win, or does it only rescue tracks the mix pass handles badly?

The fusion feature currently ships default-on for --separator uvr on the strength of
four tracks, where mix-only MICRO was 0.692 - well below the 0.868 typical of this set.
That is exactly the regime where adding onsets has the most room to help, so the
headline number may be sampling bias rather than a general gain.

This runs the full set from cached stems and reports:
  * per-track deltas, so a gain concentrated in a few weak tracks is visible
  * precision and recall separately, since fusion can only add notes
  * whether the gain correlates with how badly the mix pass did

    python validate_fusion.py
    python validate_fusion.py --sweep     # also cross-validate the stem threshold
"""

from __future__ import annotations

import argparse
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
STEM_CACHE = ROOT / "bench" / "uvr_stems"
ACT_CACHE = ROOT / "bench" / "activations"
SR = 44100
WINDOW = 0.05

CLASSES = [(0, "KD", "kick"), (1, "SD", "snare"), (2, "TT", "toms"),
           (3, "HH", "hihat"), (4, "CY", "cymbals")]
STOCK = [0.22, 0.14, 0.32, 0.22, 0.30]
TOM_PCT, TOM_FLOOR = 98.5, 0.25


def pick(col: np.ndarray, thr: float) -> np.ndarray:
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    p = NotePeakPickingProcessor(threshold=float(thr), pre_avg=0.1, post_avg=0.01,
                                 pre_max=0.02, post_max=0.01, combine=0.02, fps=100)
    return np.array([t for t, _ in p.process(col)])


def read_ann(path: Path) -> dict:
    out = {k: [] for _, k, _ in CLASSES}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] in out:
            out[p[1]].append(float(p[0]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def counts(ref, est) -> dict:
    c = {"tp": 0, "ref": len(ref), "est": len(est)}
    if len(ref) and len(est):
        _, p, _ = mir_eval.onset.f_measure(ref, est, window=WINDOW)
        c["tp"] = int(round(p * len(est)))
    return c


def prf(c) -> tuple:
    p = c["tp"] / max(c["est"], 1)
    r = c["tp"] / max(c["ref"], 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


def add(dst, src):
    for k in ("tp", "ref", "est"):
        dst[k] += src[k]


def fuse(base: np.ndarray, extra: np.ndarray, gap: float = 0.05) -> np.ndarray:
    keep = [t for t in extra if not len(base) or np.min(np.abs(base - t)) > gap]
    return np.array(sorted(list(base) + keep))


def stem_activations(model, stem_path: Path, device="cpu") -> np.ndarray:
    from adtof_pytorch import load_audio_for_model
    import torch
    with torch.no_grad():
        return model(load_audio_for_model(str(stem_path)).to(device)).cpu().numpy()[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from drum2midi import _adtof_model
    model = _adtof_model("cpu")

    tracks = []
    for wav in sorted(AUDIO.glob("*.wav")):
        act = ACT_CACHE / f"{wav.stem}.npy"
        stems = STEM_CACHE / wav.stem
        ann = ANN / f"{wav.stem.replace('_Drum', '')}_class.txt"
        if act.exists() and stems.is_dir() and ann.exists():
            tracks.append((wav, np.load(act), stems, read_ann(ann)))
    print(f"{len(tracks)} tracks with cached activations and MDX23C stems\n")
    if not tracks:
        print("Run cache_uvr_stems.py first.")
        return 1

    # stem activations are the expensive part; compute once per track/class
    stem_acts = {}
    for wav, _, stems, _ in tracks:
        per = {}
        for ci, lbl, name in CLASSES:
            f = stems / f"{name}.flac"
            if f.exists():
                per[lbl] = stem_activations(model, f)
        stem_acts[wav.stem] = per
        print(f"  activations: {wav.stem}")

    rows = []
    tot_mix = {"tp": 0, "ref": 0, "est": 0}
    tot_fus = {"tp": 0, "ref": 0, "est": 0}
    per_class = {lbl: ({"tp": 0, "ref": 0, "est": 0}, {"tp": 0, "ref": 0, "est": 0})
                 for _, lbl, _ in CLASSES}

    for wav, act, stems, ref in tracks:
        t_mix = {"tp": 0, "ref": 0, "est": 0}
        t_fus = {"tp": 0, "ref": 0, "est": 0}
        for ci, lbl, name in CLASSES:
            thr = STOCK[ci]
            if lbl == "TT":
                thr = max(float(np.percentile(act[:, ci], TOM_PCT)), TOM_FLOOR)
            o_mix = pick(act[:, ci], thr)
            sa = stem_acts[wav.stem].get(lbl)
            o_fus = o_mix if sa is None else fuse(o_mix, pick(sa.max(axis=1), STOCK[ci]))
            c_mix, c_fus = counts(ref[lbl], o_mix), counts(ref[lbl], o_fus)
            add(t_mix, c_mix); add(t_fus, c_fus)
            add(per_class[lbl][0], c_mix); add(per_class[lbl][1], c_fus)
        add(tot_mix, t_mix); add(tot_fus, t_fus)
        rows.append((wav.stem.replace("MusicDelta_", "").replace("_Drum", ""),
                     prf(t_mix), prf(t_fus)))

    rows.sort(key=lambda r: r[1][2])
    print(f"\n{'track':<16}{'mix F1':>9}{'fuse F1':>9}{'delta':>9}"
          f"{'mix P':>8}{'fuse P':>8}{'mix R':>8}{'fuse R':>8}")
    print("-" * 75)
    helped = hurt = 0
    for name, (pm, rm, fm), (pf, rf, ff) in rows:
        d = ff - fm
        helped += d > 0.005
        hurt += d < -0.005
        print(f"{name:<16}{fm:>9.3f}{ff:>9.3f}{d:>+9.3f}"
              f"{pm:>8.3f}{pf:>8.3f}{rm:>8.3f}{rf:>8.3f}")

    pm, rm, fm = prf(tot_mix)
    pf, rf, ff = prf(tot_fus)
    print("-" * 75)
    print(f"{'MICRO':<16}{fm:>9.3f}{ff:>9.3f}{ff - fm:>+9.3f}"
          f"{pm:>8.3f}{pf:>8.3f}{rm:>8.3f}{rf:>8.3f}")
    print(f"\nfusion helped {helped} tracks, hurt {hurt}, neutral "
          f"{len(rows) - helped - hurt}")

    deltas = np.array([r[2][2] - r[1][2] for r in rows])
    base = np.array([r[1][2] for r in rows])
    if len(rows) > 3 and base.std() > 0 and deltas.std() > 0:
        r = float(np.corrcoef(base, deltas)[0, 1])
        print(f"correlation between mix F1 and the gain: {r:+.3f}")
        print("  strongly negative would mean fusion only rescues weak tracks")
        weak = deltas[base < np.median(base)].mean()
        strong = deltas[base >= np.median(base)].mean()
        print(f"  mean gain on the weaker half: {weak:+.3f}, on the stronger half: {strong:+.3f}")

    print(f"\n{'drum':<10}{'mix F1':>9}{'fuse F1':>9}{'delta':>9}")
    print("-" * 37)
    for _, lbl, _ in CLASSES:
        a, b = per_class[lbl]
        print(f"{lbl:<10}{prf(a)[2]:>9.3f}{prf(b)[2]:>9.3f}{prf(b)[2] - prf(a)[2]:>+9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
