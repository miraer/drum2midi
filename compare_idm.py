"""Tests the Inverse Drum Machine as a replacement for MDX23C in the velocity stage.

The pipeline splits work in two: ADTOF says what and when, a separator says how loud.
The separator is MDX23C, 417 MB and the slowest part of a run. IDM (TASLP 2026) does
separation by analysis-by-synthesis and ships Apache-2.0 weights of 2.4 MB -- two orders
of magnitude smaller -- and it accepts an external transcription, which is exactly the
shape of our pipeline: we already have onsets, we only want per-hit loudness.

Whether it is good enough is a measurement, not an argument. GMD carries the velocity
the drum module actually recorded, so both separators can be scored against it: take
each stem, read its level at the known hit times, and correlate with the true velocity
within each track.

Per-track correlation is used because velocity is relative -- a model that gets every
hit right but scales the whole track differently is still correct musically.

    python compare_idm.py --limit 8
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
IDM_DIR = ROOT / "inverse-drum-machine"
GMD = ROOT / "gmd" / "groove"
SR = 44100

sys.path.insert(0, str(ROOT))
from drum2midi import _mono, peak_at  # noqa: E402

# GMD pitch -> the coarse instrument both separators can be compared on
FAMILY = {
    **{p: "kick" for p in (35, 36)},
    **{p: "snare" for p in (38, 40, 37)},
    **{p: "hihat" for p in (42, 22, 46, 26, 44)},
    **{p: "toms" for p in (48, 50, 45, 47, 43, 58)},
    **{p: "cymbals" for p in (49, 57, 52, 55, 51, 59, 53)},
}

# IDM's own class names, from model.train_classes
IDM_MAP = {"KD": "kick", "SD": "snare",
           "HH_CHH": "hihat", "HH_OHH": "hihat",
           "TT_HFT": "toms", "TT_HMT": "toms", "TT_LMT": "toms",
           "CY_CR": "cymbals", "CY_RD": "cymbals"}


def load_tracks(limit: int | None):
    info = GMD / "info.csv"
    if not info.exists():
        return []
    rows = [r for r in csv.DictReader(info.open(encoding="utf-8"))
            if r.get("split") == "test" and float(r.get("duration", 0)) > 8]
    rows.sort(key=lambda r: r["id"])
    return rows[:limit] if limit else rows


def true_hits(midi_path: Path):
    import pretty_midi
    out = []
    for inst in pretty_midi.PrettyMIDI(str(midi_path)).instruments:
        for n in inst.notes:
            fam = FAMILY.get(n.pitch)
            if fam:
                out.append((float(n.start), fam, int(n.velocity)))
    return sorted(out)


def correlate(levels: dict, hits: list) -> dict:
    """Per-instrument Pearson r between stem level at the hit and true velocity."""
    out = {}
    for fam in set(f for _, f, _ in hits):
        sig = levels.get(fam)
        if sig is None:
            continue
        xs, ys = [], []
        for t, f, vel in hits:
            if f != fam:
                continue
            level = peak_at(sig, t)
            if level > 0:
                xs.append(20 * np.log10(level + 1e-9))
                ys.append(vel)
        if len(xs) >= 6 and np.std(xs) > 1e-6:
            out[fam] = float(np.corrcoef(xs, ys)[0, 1])
    return out


def patch_idm_device(device: str) -> None:
    """Makes IDM runnable without CUDA.

    idm/synthesis_conditioning/embedding.py declares
    `device: torch.device = torch.device("cuda")` as a default argument, and
    decoder.py calls it without passing one, so every call allocates on CUDA. On a
    machine without it, inference dies. Patched here rather than in their tree so the
    checkout stays pristine and updatable.
    """
    import torch
    from idm.decoder import decoder as dec
    from idm.synthesis_conditioning import embedding as emb

    original = emb.get_conditioning_vector

    def patched(*args, **kwargs):
        kwargs.setdefault("device", torch.device(device))
        return original(*args, **kwargs)

    emb.get_conditioning_vector = patched
    if hasattr(dec, "get_conditioning_vector"):
        dec.get_conditioning_vector = patched


def run_idm(audio_path: Path, model, device: str) -> dict:
    """IDM stems, keyed by our family names.

    A temporary mono file is written rather than passing a waveform, because their
    separate() applies torch.from_numpy() to both branches (so the tensor branch
    raises) and the model's first convolution expects a single channel.
    """
    import librosa
    import soundfile as sf
    from idm.inference import separate

    y, _ = librosa.load(str(audio_path), sr=44100, mono=True)
    with tempfile.TemporaryDirectory(prefix="idm_") as tmp:
        mono = Path(tmp) / "mono.wav"
        sf.write(mono, y, 44100)
        stems = separate(str(mono), model, save_to_disk=False,
                         masking="wiener", alpha=1.0)
    out = {}
    if not isinstance(stems, dict):
        # separate() returns a single tensor of shape (classes, samples) ordered by
        # model.train_classes, not the dict the README implies
        arr = stems.detach().cpu().numpy() if hasattr(stems, "detach") else np.asarray(stems)
        arr = np.squeeze(arr)
        names = list(getattr(model, "train_classes", []))
        if arr.ndim == 2 and len(names) == arr.shape[0]:
            items = list(zip(names, arr))
        else:
            raise RuntimeError(f"unexpected IDM output {arr.shape} for classes {names}")
    else:
        items = list(stems.items())

    for name, data in items:
        fam = IDM_MAP.get(name)
        if fam is None:
            continue
        arr = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
        arr = np.squeeze(arr)
        if arr.ndim > 1:
            arr = arr.mean(axis=0)
        out[fam] = out[fam] + arr if fam in out else arr
    return out

def run_uvr(audio_path: Path) -> dict:
    """MDX23C stems, via the pipeline's own code path."""
    from drum2midi import separate_uvr
    import librosa

    y, _ = librosa.load(str(audio_path), sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    stems = separate_uvr(audio_path, "auto", y.shape[-1])
    merged = {}
    for name, arr in stems.items():
        fam = {"kick": "kick", "snare": "snare", "hihat": "hihat", "toms": "toms",
               "cymbals": "cymbals", "ride": "cymbals"}.get(name)
        if fam is None:
            continue
        mono = _mono(arr)
        merged[fam] = merged[fam] + mono if fam in merged else mono
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rows = load_tracks(args.limit)
    if not rows:
        print(f"Groove MIDI not found at {GMD}; see the Install section of README.md.")
        return 1

    sys.path.insert(0, str(IDM_DIR))
    import os
    prev = os.getcwd()
    os.chdir(IDM_DIR)
    try:
        from idm.inference import load_model
        model, name = load_model("idm-44-train-kits", args.device, log_dir="pretrained")
        patch_idm_device(args.device)
        print(f"IDM checkpoint: {name}")
        print(f"IDM classes: {getattr(model, 'train_classes', '?')}\n")
    finally:
        os.chdir(prev)

    results = {"idm": {}, "uvr": {}}
    timing = {"idm": 0.0, "uvr": 0.0}
    for i, r in enumerate(rows, 1):
        wav = GMD / r["audio_filename"]
        mid = GMD / r["midi_filename"]
        if not wav.exists() or not mid.exists():
            continue
        hits = true_hits(mid)
        if len(hits) < 20:
            continue
        for tag, fn in (("idm", lambda: run_idm(wav, model, args.device)),
                        ("uvr", lambda: run_uvr(wav))):
            t0 = time.time()
            try:
                levels = fn()
            except Exception as exc:
                print(f"  ! {tag} failed on {r['id']}: {str(exc)[:90]}")
                continue
            timing[tag] += time.time() - t0
            for fam, corr in correlate(levels, hits).items():
                results[tag].setdefault(fam, []).append(corr)
        print(f"  [{i}/{len(rows)}] {r['id']}", flush=True)

    print(f"\n{'instrument':<12}{'IDM r':>9}{'MDX23C r':>11}{'tracks':>9}")
    print("-" * 42)
    for fam in ("kick", "snare", "toms", "hihat", "cymbals"):
        a, b = results["idm"].get(fam, []), results["uvr"].get(fam, [])
        if not a and not b:
            continue
        print(f"{fam:<12}{(np.mean(a) if a else float('nan')):>9.3f}"
              f"{(np.mean(b) if b else float('nan')):>11.3f}{max(len(a), len(b)):>9}")
    print(f"\ntime: IDM {timing['idm']:.0f}s, MDX23C {timing['uvr']:.0f}s")
    print("Correlation is computed within each track and then averaged; velocity is "
          "relative,\nso a constant offset per track is not an error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
