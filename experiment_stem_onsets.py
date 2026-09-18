"""Stems need onset detection, not classification - measuring how best to get it.

The earlier experiment fed isolated stems to ADTOF and tom F1 collapsed to 0.000.
That was the wrong tool: on a tom stem the instrument is already known, so a 5-class
classifier is solving a problem that does not exist, out of its training distribution.

Here we compare ways of getting onsets out of a stem, and whether fusing them with the
mix-based transcription raises recall without wrecking precision.

  mix         ADTOF class activation on the mix (current behaviour)
  stem_any    ADTOF on the stem, max over all 5 classes - using it as a generic
              onset detector and taking the class from which stem it is
  stem_flux   SuperFlux-style spectral flux on the stem (librosa, max_size=3)
  fuse_any    mix onsets, plus stem_any onsets that the mix missed
  fuse_flux   mix onsets, plus stem_flux onsets that the mix missed

    python experiment_stem_onsets.py --limit 12
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

CLASSES = [(0, "KD", "kick"), (1, "SD", "snare"), (2, "TT", "toms"),
           (3, "HH", "hihat"), (4, "CY", "cymbals")]
STOCK = [0.22, 0.14, 0.32, 0.22, 0.30]
TOM_PCT, TOM_FLOOR = 98.5, 0.25
# flux envelopes are normalised to 0..1, so these are fractions of the track maximum
FLUX_THR = {"KD": 0.18, "SD": 0.16, "TT": 0.28, "HH": 0.16, "CY": 0.20}
VARIANTS = ["mix", "stem_any", "stem_flux", "fuse_any", "fuse_flux"]


@contextlib.contextmanager
def _in_dir(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def pick(col: np.ndarray, thr: float) -> np.ndarray:
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    p = NotePeakPickingProcessor(threshold=float(thr), pre_avg=0.1, post_avg=0.01,
                                 pre_max=0.02, post_max=0.01, combine=0.02, fps=100)
    return np.array([t for t, _ in p.process(col)])


def flux_envelope(sig: np.ndarray) -> np.ndarray:
    import librosa
    env = librosa.onset.onset_strength(y=sig, sr=SR, hop_length=441, max_size=3)
    peak = float(env.max())
    return (env / peak) if peak > 0 else env


def read_ann(path: Path) -> dict:
    out = {k: [] for _, k, _ in CLASSES}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) >= 2 and p[1] in out:
            out[p[1]].append(float(p[0]))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def fuse(base: np.ndarray, extra: np.ndarray, gap: float = 0.05) -> np.ndarray:
    keep = [t for t in extra if not len(base) or np.min(np.abs(base - t)) > gap]
    return np.array(sorted(list(base) + keep))


def tally(ref, est, acc):
    acc["ref"] += len(ref)
    acc["est"] += len(est)
    if len(ref) and len(est):
        _, p, _ = mir_eval.onset.f_measure(ref, est, window=WINDOW)
        acc["tp"] += int(round(p * len(est)))


def f1(a):
    p = a["tp"] / max(a["est"], 1); r = a["tp"] / max(a["ref"], 1)
    return 2 * p * r / max(p + r, 1e-9)


def separate_uvr_batch(tracks, tmp_root: Path) -> dict:
    """Load the MDX23C model once and push every track through it. The CLI reloads
    a 400 MB model per file, which dominates runtime on a handful of tracks."""
    from audio_separator.separator import Separator

    out = {}
    sep = Separator(output_dir=str(tmp_root), model_file_dir=str(ROOT / "uvr" / "models"),
                    log_level=40)
    sep.load_model(model_filename="MDX23C-DrumSep-aufr33-jarredou.ckpt")
    for wav in tracks:
        files = sep.separate(str(wav))
        stems = {}
        for f in files:
            path = tmp_root / f if not Path(f).is_absolute() else Path(f)
            if not path.exists():
                continue
            for raw, name in (("kick", "kick"), ("snare", "snare"), ("toms", "toms"),
                              ("hh", "hihat"), ("crash", "cymbals"), ("ride", "ride")):
                if f"({raw})" in path.name:
                    y, _ = sf.read(str(path), always_2d=True)
                    stems[name] = y.T.astype(np.float32)
        out[wav.stem] = stems
        print(f"    UVR done: {wav.stem} ({len(stems)} stems)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--separator", default="larsnet", choices=["larsnet", "uvr"])
    args = ap.parse_args()

    import torch
    from adtof_pytorch import (calculate_n_bins, create_frame_rnn_model,
                               get_default_weights_path, load_audio_for_model,
                               load_pytorch_weights)

    model = create_frame_rnn_model(calculate_n_bins())
    model.eval()
    w = get_default_weights_path()
    if w and Path(w).exists():
        model = load_pytorch_weights(model, w, strict=False)

    def activations(path):
        with torch.no_grad():
            return model(load_audio_for_model(str(path))).cpu().numpy()[0]

    tracks = sorted(AUDIO.glob("*.wav"))[: args.limit]
    totals = {v: {lbl: {"tp": 0, "ref": 0, "est": 0} for _, lbl, _ in CLASSES}
              for v in VARIANTS}

    uvr_stems = {}
    uvr_tmp = None
    if args.separator == "uvr":
        uvr_tmp = Path(tempfile.mkdtemp(prefix="uvrbatch_"))
        print(f"separating {len(tracks)} tracks with MDX23C (slow) ...")
        uvr_stems = separate_uvr_batch(tracks, uvr_tmp)

    sys.path.insert(0, str(LARSNET_DIR))
    with _in_dir(LARSNET_DIR):
        net = None
        if args.separator == "larsnet":
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
            if net is not None:
                with torch.no_grad():
                    stems = {k: v.cpu().numpy()
                             for k, v in net.separate_wiener(torch.from_numpy(audio)).items()}
            else:
                stems = uvr_stems.get(wav.stem, {})

            act_mix = activations(wav)
            tmp = Path(tempfile.mkdtemp(prefix="stemons_"))
            try:
                for ci, lbl, name in CLASSES:
                    s = stems.get(name)
                    if s is None:
                        continue
                    mono = s.mean(axis=0) if s.ndim > 1 else s
                    p = tmp / f"{name}.wav"
                    sf.write(str(p), s.T, SR)

                    thr = STOCK[ci]
                    if lbl == "TT":
                        thr = max(float(np.percentile(act_mix[:, ci], TOM_PCT)), TOM_FLOOR)
                    o_mix = pick(act_mix[:, ci], thr)

                    a_stem = activations(p)
                    o_any = pick(a_stem.max(axis=1), STOCK[ci])
                    o_flux = pick(flux_envelope(mono), FLUX_THR[lbl])

                    cand = {"mix": o_mix, "stem_any": o_any, "stem_flux": o_flux,
                            "fuse_any": fuse(o_mix, o_any),
                            "fuse_flux": fuse(o_mix, o_flux)}
                    for v in VARIANTS:
                        tally(ref.get(lbl, np.array([])), cand[v], totals[v][lbl])
            finally:
                import shutil
                shutil.rmtree(tmp, ignore_errors=True)
            print(f"  [{i:2d}/{len(tracks)}] {wav.stem}")

    sys.path.remove(str(LARSNET_DIR))
    if uvr_tmp is not None:
        import shutil
        shutil.rmtree(uvr_tmp, ignore_errors=True)

    print(f"\nF1 per class over {len(tracks)} MDB tracks "
          f"(stems from {args.separator})")
    print(f"{'variant':<11}" + "".join(f"{l:>9}" for _, l, _ in CLASSES) + f"{'MICRO':>9}")
    print("-" * (11 + 9 * 6))
    for v in VARIANTS:
        micro = {"tp": 0, "ref": 0, "est": 0}
        cells = []
        for _, lbl, _ in CLASSES:
            c = totals[v][lbl]
            for k in micro:
                micro[k] += c[k]
            cells.append(f"{f1(c):>9.3f}")
        print(f"{v:<11}" + "".join(cells) + f"{f1(micro):>9.3f}")

    print(f"\ntoms in detail (our worst class):")
    print(f"{'variant':<11}{'ref':>7}{'est':>7}{'match':>7}{'prec':>8}{'rec':>8}{'F1':>8}")
    for v in VARIANTS:
        c = totals[v]["TT"]
        p = c["tp"] / max(c["est"], 1); r = c["tp"] / max(c["ref"], 1)
        print(f"{v:<11}{c['ref']:>7}{c['est']:>7}{c['tp']:>7}{p:>8.3f}{r:>8.3f}{f1(c):>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
