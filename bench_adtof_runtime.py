"""Does the OpenVINO speedup survive on the real ADTOF model, and is the output identical?

A synthetic GRU ran 12-15x faster under OpenVINO than under torch on the same CPU, with
outputs agreeing to 1.5e-07. That is only interesting if it carries to the model actually
in the pipeline, which is not a bare GRU: ADTOF puts a convolutional front end in front of
one, and the front end is the part an alternative runtime is least likely to improve.

Transcription is the slowest stage in this pipeline, so the measurement is worth making
carefully:

  * the real checkpoint, loaded the way drum2midi loads it
  * real audio, not noise, at several lengths
  * activations compared element-wise against torch, because a faster number is
    meaningless if the peak picker would then see different frames
  * repeats with a range, since a single timing is not a measurement

A negative result is useful: if the gain vanishes on the real model, that closes the
question and nobody needs to try again.

    python bench_adtof_runtime.py
    python bench_adtof_runtime.py --audio input/demo.wav --repeats 5
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def load_model():
    from adtof_pytorch import (calculate_n_bins, create_frame_rnn_model,
                               get_default_weights_path, load_pytorch_weights)
    model = create_frame_rnn_model(calculate_n_bins())
    model.eval()
    weights = get_default_weights_path()
    if weights and Path(weights).exists():
        model = load_pytorch_weights(model, weights, strict=False)
    return model


def find_audio(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for candidate in (ROOT / "input" / "demo.wav",
                      ROOT / "restem_in", ROOT / "mdbdrums"):
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            wav = next(candidate.rglob("*.wav"), None)
            if wav:
                return wav
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default="")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seconds", default="5,15,30",
                    help="lengths of audio to feed, in seconds")
    args = ap.parse_args()

    if importlib.util.find_spec("openvino") is None:
        print("openvino is not installed; nothing to compare against")
        return 1

    import numpy as np
    import torch
    from adtof_pytorch import load_audio_for_model

    audio = find_audio(args.audio or None)
    if audio is None:
        print("no audio found; pass --audio")
        return 1
    print(f"audio      : {audio.name}")

    model = load_model()
    print(f"checkpoint : loaded, {sum(p.numel() for p in model.parameters()):,} parameters")

    full = load_audio_for_model(str(audio))
    # (batch, frames, bins) or (batch, frames, bins, channels) - frames is dim 1 either way
    total_frames = full.shape[1]
    print(f"input      : {tuple(full.shape)}, {total_frames / 100:.1f}s\n")

    import openvino as ov
    core = ov.Core()
    print(f"{args.repeats} repeats per configuration, median and range\n")

    for secs in [float(s) for s in args.seconds.split(",")]:
        frames = min(int(secs * 100), total_frames)
        x = full[:, :frames]
        got = x.shape[1]
        if got < 100:
            print(f"{secs:g}s requested but only {got / 100:.1f}s available; skipping")
            continue

        print(f"{got / 100:.1f}s of audio  {tuple(x.shape)}")

        with torch.no_grad():
            for _ in range(2):
                model(x)
            t_samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                want = model(x).numpy()
                t_samples.append((time.perf_counter() - start) * 1000)

        t_med = statistics.median(t_samples)
        print(f"  {'torch cpu':<18}{t_med:9.1f} ms   "
              f"[{min(t_samples):7.1f}, {max(t_samples):7.1f}]")

        try:
            m = ov.convert_model(model, example_input=x)
            m.reshape({0: ov.PartialShape(list(x.shape))})
            compiled = core.compile_model(m, "CPU")
            arr = x.numpy()
            out = compiled(arr)[compiled.output(0)]
            err = float(np.abs(out - want).max())

            for _ in range(2):
                compiled(arr)
            o_samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                compiled(arr)
                o_samples.append((time.perf_counter() - start) * 1000)
            o_med = statistics.median(o_samples)
            print(f"  {'openvino cpu':<18}{o_med:9.1f} ms   "
                  f"[{min(o_samples):7.1f}, {max(o_samples):7.1f}]   max err {err:.1e}")

            # The peak picker thresholds activations, so what matters is not the raw
            # difference but whether any frame would cross a threshold differently.
            crossings = 0
            for thr in (0.14, 0.22, 0.30, 0.32):
                crossings += int(np.sum((want >= thr) != (out >= thr)))
            print(f"  {'':18}{'':9}      frames that would threshold "
                  f"differently: {crossings}")
            if err < 1e-3 and crossings == 0:
                print(f"  {'':18}{'':9}      -> {t_med / o_med:.1f}x faster, "
                      f"same decisions")
            else:
                print(f"  {'':18}{'':9}      -> FASTER BUT NOT IDENTICAL; "
                      f"speed is not usable as is")
        except Exception as exc:
            print(f"  openvino cpu       conversion or compile failed:")
            print(f"    {str(exc).strip().splitlines()[-1][:88]}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
