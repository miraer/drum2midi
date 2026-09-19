"""Is OpenVINO really 18x faster than torch at the recurrent stage, on the same CPU?

A single run of bench_npu_stages.py put a bidirectional GRU at 813 ms under torch and
44.5 ms under OpenVINO, on the same CPU, with a maximum absolute difference of 1.5e-07
between their outputs. If that holds it matters far more than anything about the NPU,
because transcription is the slowest stage in the pipeline and it runs on the CPU.

One run is not a result. This repeats each configuration and reports a median with a
range, because a difference this large is exactly the kind that turns out to be a warm-up
artefact, a thread-count accident, or a model that quietly got converted to something
cheaper.

It also varies the sequence length, since the whole reason this stage is on the CPU is
per-step launch overhead: if the gap is constant in absolute terms it is fixed cost, and
if it scales with length it is the arithmetic.

    python bench_rnn_runtime.py
    python bench_rnn_runtime.py --repeats 7 --frames 500,1000,3000
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build():
    """ADTOF's shape: a 3-layer bidirectional GRU over 84 mel bins, 5 classes out."""
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(84, 60, num_layers=3, bidirectional=True,
                              batch_first=True)
            self.out = nn.Linear(120, 5)

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.out(h)

    return Net().eval()


def torch_ms(net, x, repeats: int) -> list[float]:
    import torch
    out = []
    with torch.no_grad():
        for _ in range(2):
            net(x)
        for _ in range(repeats):
            start = time.perf_counter()
            net(x)
            out.append((time.perf_counter() - start) * 1000)
    return out


def ov_ms(net, x, device: str, repeats: int):
    import numpy as np
    import openvino as ov
    import torch

    model = ov.convert_model(net, example_input=x)
    model.reshape({0: ov.PartialShape(list(x.shape))})
    compiled = ov.Core().compile_model(model, device)
    arr = x.numpy()
    with torch.no_grad():
        want = net(x).numpy()
    got = compiled(arr)[compiled.output(0)]
    err = float(np.abs(got - want).max())
    for _ in range(2):
        compiled(arr)
    out = []
    for _ in range(repeats):
        start = time.perf_counter()
        compiled(arr)
        out.append((time.perf_counter() - start) * 1000)
    return out, err


def summarise(label: str, samples: list[float], err: float | None = None) -> float:
    med = statistics.median(samples)
    lo, hi = min(samples), max(samples)
    e = f"  max err {err:.1e}" if err is not None else ""
    print(f"  {label:<20}{med:9.1f} ms   [{lo:7.1f}, {hi:7.1f}]{e}")
    return med


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--frames", default="250,1000,3000",
                    help="sequence lengths; 100 frames is one second of audio")
    args = ap.parse_args()

    import torch

    have_ov = importlib.util.find_spec("openvino") is not None
    print(f"torch {torch.__version__}, threads {torch.get_num_threads()}, "
          f"openvino {'present' if have_ov else 'absent'}")
    print(f"{args.repeats} repeats per configuration, median and range\n")

    net = build()
    for frames in [int(f) for f in args.frames.split(",")]:
        x = torch.randn(1, frames, 84)
        print(f"{frames} frames  ({frames / 100:.1f}s of audio)")
        t = summarise("torch cpu", torch_ms(net, x, args.repeats))
        if have_ov:
            try:
                samples, err = ov_ms(net, x, "CPU", args.repeats)
                o = summarise("openvino cpu", samples, err)
                if err > 1e-3:
                    print("    ^ outputs differ materially; the speed is not comparable")
                else:
                    print(f"    openvino is {t / o:.1f}x faster on the same CPU")
            except Exception as exc:
                print(f"  openvino cpu        failed: "
                      f"{str(exc).strip().splitlines()[-1][:50]}")
        print()

    print("If the ratio is roughly constant across lengths, the difference is in the\n"
          "kernels rather than in per-call overhead, and it would carry to the real\n"
          "model. If it shrinks as the sequence grows, it is fixed cost and matters\n"
          "much less on a full song.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
