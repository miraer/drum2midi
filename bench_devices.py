"""Where should each stage run: measured per component, not assumed.

The pipeline has two neural stages with opposite hardware profiles:

  * the separator (MDX23C) is convolution + attention over spectrograms -- exactly what
    a GPU is built for;
  * the transcriber (ADTOF) is a small CNN feeding a GRU, and recurrent layers are a
    sequence of tiny dependent kernels, which is the case where dispatch overhead can
    exceed the work itself.

So "use the GPU" is not one decision. This script times each stage on each available
device and prints what the defaults should be.

    python bench_devices.py
    python bench_devices.py --audio input/demo.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def available() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():
        devs.append("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        devs.append("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devs.append("mps")
    return devs


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()


def time_adtof(audio: Path, device: str, repeats: int = 3) -> float:
    """Wall time of one ADTOF forward pass, best of `repeats`."""
    from drum2midi import _adtof_model
    from adtof_pytorch import load_audio_for_model

    model = _adtof_model(device)
    x = load_audio_for_model(str(audio)).to(device)

    with torch.no_grad():
        model(x)          # warm up: first call pays for kernel compilation
        sync(device)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            sync(device)
            best = min(best, time.perf_counter() - t0)
    return best


def time_conv(device: str, repeats: int = 5) -> float:
    """A separator-shaped workload: 2D convolution over a spectrogram."""
    net = torch.nn.Sequential(
        torch.nn.Conv2d(4, 64, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 64, 3, padding=1), torch.nn.ReLU(),
        torch.nn.Conv2d(64, 4, 3, padding=1),
    ).to(device).eval()
    x = torch.randn(2, 4, 512, 256, device=device)
    with torch.no_grad():
        net(x)
        sync(device)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            net(x)
            sync(device)
            best = min(best, time.perf_counter() - t0)
    return best


def time_gru(device: str, repeats: int = 5) -> float:
    """A transcriber-shaped workload: a recurrent layer over a long sequence."""
    net = torch.nn.GRU(128, 128, num_layers=2, batch_first=True).to(device).eval()
    x = torch.randn(1, 2000, 128, device=device)
    with torch.no_grad():
        net(x)
        sync(device)
        best = float("inf")
        for _ in range(repeats):
            t0 = time.perf_counter()
            net(x)
            sync(device)
            best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", default=str(ROOT / "input" / "demo.wav"))
    args = ap.parse_args()

    devs = available()
    print(f"torch {torch.__version__}")
    print(f"devices: {', '.join(devs)}\n")

    rows = []
    for name, fn in (("Conv2d stack (separator-like)", time_conv),
                     ("GRU (transcriber-like)", time_gru)):
        times = {}
        for d in devs:
            try:
                times[d] = fn(d)
            except Exception as exc:
                times[d] = float("nan")
                print(f"  {name} on {d}: failed ({type(exc).__name__}: {exc})")
        rows.append((name, times))

    audio = Path(args.audio)
    if audio.exists():
        times = {}
        for d in devs:
            try:
                times[d] = time_adtof(audio, d)
            except Exception as exc:
                times[d] = float("nan")
                print(f"  ADTOF on {d}: failed ({type(exc).__name__}: {exc})")
        rows.append(("ADTOF, real model", times))
    else:
        print(f"(skipping the real ADTOF pass: {audio} not found)")

    width = max(len(r[0]) for r in rows) + 2
    header = f"{'workload':<{width}}" + "".join(f"{d:>12}" for d in devs) + f"{'verdict':>22}"
    print(header)
    print("-" * len(header))
    for name, times in rows:
        line = f"{name:<{width}}"
        for d in devs:
            t = times.get(d, float("nan"))
            line += f"{t*1000:>11.1f}m" if t == t else f"{'-':>12}"
        base = times.get("cpu", float("nan"))
        best = min((d for d in devs if times.get(d, float("nan")) == times.get(d, float("nan"))),
                   key=lambda d: times[d], default="cpu")
        if base == base and times.get(best, float("nan")) == times.get(best, float("nan")):
            factor = base / times[best] if times[best] else float("nan")
            line += f"{best} ({factor:.1f}x vs cpu)".rjust(22)
        print(line)

    print("\nTimes are milliseconds, best of several runs after a warm-up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
