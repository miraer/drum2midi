"""Can this machine train a model - measured, not guessed.

Two separate questions:
  1. The MSG-LD latent diffusion separator (5.4 GB checkpoint) - clearly no.
  2. A small onset detector working on a stem - that is what needs measuring.

The second is fundamentally easier: the instrument is already known, only the moment of
the hit has to be found. For scale, the entire ADTOF model is 3.6 MB, i.e. under a
million parameters.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

print("=== devices ===")
print(f"  torch {torch.__version__}")
print(f"  CPU threads: {torch.get_num_threads()}")
print(f"  CUDA: {torch.cuda.is_available()}")
xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
print(f"  XPU (Intel Arc): {xpu}")
if xpu:
    print(f"    {torch.xpu.get_device_name(0)}")
try:
    import torch.backends.mkldnn as mkldnn
    print(f"  MKL-DNN: {mkldnn.is_available()}")
except Exception:
    pass


class StemOnsetNet(nn.Module):
    """Onset detector for a single stem.

    Input is a log-mel window around a frame, output is the hit probability.
    in advance, so no classification is needed and the model can stay small.
    """

    def __init__(self, n_mels: int = 64, context: int = 25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((1, 3)),
            nn.Conv2d(16, 32, (3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((1, 3)),
            nn.Conv2d(32, 64, (3, 3), padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.net(torch.zeros(1, 1, context, n_mels)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(flat, 128), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(flat := 128, 1))

    def forward(self, x):
        return self.head(self.net(x))


class SeqOnsetNet(nn.Module):
    """The same, but fully convolutional in time: takes a multi-second chunk and
    emits a hit probability for every frame at once.

    Convolutions reuse work between neighbouring frames, which makes this orders of
    magnitude cheaper than running a separate window per frame. ADTOF itself is
    built this way.
    """

    def __init__(self, n_mels: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(16, 32, (3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((1, 4)),
            nn.Conv2d(32, 32, (3, 3), padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.head = nn.Conv1d(32 * (n_mels // 16), 1, 1)

    def forward(self, x):                       # (B, 1, T, mels)
        h = self.body(x)                        # (B, C, T, mels/16)
        b, c, t, f = h.shape
        return self.head(h.permute(0, 2, 1, 3).reshape(b, t, c * f).transpose(1, 2))


def bench(device: str, batch: int = 256, n_mels: int = 64, context: int = 25,
          steps: int = 30) -> float:
    """Returns training examples per second."""
    model = StemOnsetNet(n_mels, context).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    x = torch.randn(batch, 1, context, n_mels, device=device)
    y = torch.randint(0, 2, (batch, 1), device=device).float()

    for _ in range(3):                      # warm-up
        opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
    if device == "xpu":
        torch.xpu.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        lossf(model(x), y).backward()
        opt.step()
    if device == "xpu":
        torch.xpu.synchronize()
    return batch * steps / (time.perf_counter() - t0)


def bench_seq(device: str, batch: int = 8, seconds: float = 10.0, fps: int = 100,
              n_mels: int = 64, steps: int = 20) -> tuple:
    """Frames per second for the fully convolutional scheme."""
    T = int(seconds * fps)
    model = SeqOnsetNet(n_mels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    x = torch.randn(batch, 1, T, n_mels, device=device)
    y = torch.randint(0, 2, (batch, 1, T), device=device).float()

    for _ in range(2):
        opt.zero_grad(); lossf(model(x), y).backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(steps):
        opt.zero_grad()
        lossf(model(x), y).backward()
        opt.step()
    dt = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    return batch * T * steps / dt, n_params


model = StemOnsetNet()
n_params = sum(p.numel() for p in model.parameters())
print(f"\n=== onset detector model ===")
print(f"  window per frame:    {n_params:,} parameters")
seq_rate, seq_params = bench_seq("cpu")
print(f"  fully convolutional: {seq_params:,} parameters")
print(f"  for comparison, ADTOF: ~900,000 (3.6 MB of weights)")

print(f"\n=== training throughput ===")
devices = ["cpu"] + (["xpu"] if xpu else [])
rates = {}
for dev in devices:
    try:
        r = bench(dev)
        rates[dev] = r
        print(f"  window per frame,    {dev}: {r:>12,.0f} frames/s")
    except Exception as exc:
        print(f"  window per frame,    {dev}: failed ({str(exc)[:50]})")
for dev in devices:
    try:
        r, _ = bench_seq(dev)
        rates[f"seq_{dev}"] = r
        print(f"  fully convolutional, {dev}: {r:>12,.0f} frames/s")
    except Exception as exc:
        print(f"  fully convolutional, {dev}: failed ({str(exc)[:50]})")

FPS = 100
GMD_HOURS = 10.76
frames = GMD_HOURS * 3600 * FPS
print(f"\n=== what that means at our scale ===")
print(f"  GMD: {GMD_HOURS} h of audio = {frames:,.0f} frames at {FPS} fps")
print(f"  five stems per track -> {frames * 5:,.0f} frames per epoch")

seq = rates.get("seq_cpu", 0)
naive = rates.get("cpu", 0)
if seq and naive:
    print(f"\n{'scheme':<20}{'epoch, 1 stem':>16}{'epoch, 5 stems':>18}{'30 epochs':>12}")
    print("-" * 66)
    for name, rate in (("window per frame", naive), ("fully convolutional", seq)):
        e1 = frames / rate / 60
        e5 = frames * 5 / rate / 60
        print(f"{name:<20}{e1:>14.1f} m{e5:>16.1f} m{e5*30/60:>10.1f} h")
    print(f"\n  fully convolutional speedup: {seq/naive:.0f}x")

print(f"\n=== for comparison: what MSG-LD requires ===")
print("  latent diffusion, a 5.4 GB checkpoint, plus VAE and HiFi-GAN;")
print("  inference needs dozens of denoising steps per chunk.")
