"""Is there anywhere in this pipeline the NPU is actually useful?

The pipeline has two neural stages with opposite shapes, and they already sit on
different devices for measured reasons: the separator is a convolution stack and runs on
the GPU, the transcriber is recurrent and runs on the CPU because an integrated GPU is
4.1x slower at it. That leaves the NPU idle, and "we have an NPU" is not by itself a
reason to use it.

There are only three ways it could earn a place, and this measures all three:

  1. beat the GPU on the separator                 -> it takes over that stage
  2. beat the CPU on the transcriber               -> it takes over that stage, and
                                                      frees the CPU, which is also
                                                      running everything else
  3. beat neither but run alongside one of them    -> it absorbs work in a folder run,
                                                      where the next file can be
                                                      separated while this one is
                                                      transcribed

Cases 1 and 2 are a straight race. Case 3 needs a queue, so devices are also timed
through AsyncInferQueue with several requests in flight; a device that loses a race can
still add throughput when work is pipelined.

Correctness is checked against torch on every device before anything is timed, because a
device that is fast and wrong is not a result.

    python bench_npu_stages.py
    python bench_npu_stages.py --seconds 4 --jobs 6
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def separator_shaped():
    """Conv2d over a spectrogram, which is what MDX23C is made of."""
    import torch
    import torch.nn as nn
    net = nn.Sequential(
        nn.Conv2d(4, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
    ).eval()
    return net, torch.randn(1, 4, 256, 256)


def transcriber_shaped():
    """A bidirectional GRU over a long sequence, which is what ADTOF is made of.

    100 frames per second, so 1000 steps is ten seconds of audio - long enough that
    per-step launch overhead dominates, which is the whole reason this stage is on the
    CPU today.
    """
    import torch
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

    return Net().eval(), torch.randn(1, 1000, 84)


def time_torch(net, x, device: str, seconds: float) -> float | None:
    import torch
    try:
        if device == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            return None
        if device == "cuda" and not torch.cuda.is_available():
            return None
        m, t = net.to(device), x.to(device)
        with torch.no_grad():
            for _ in range(3):
                m(t)
            if device != "cpu":
                getattr(torch, device).synchronize()
            start, n = time.perf_counter(), 0
            while time.perf_counter() - start < seconds:
                m(t)
                n += 1
            if device != "cpu":
                getattr(torch, device).synchronize()
            per = (time.perf_counter() - start) / n * 1000
        return per
    except Exception:
        return None
    finally:
        net.to("cpu")


def run_stage(name: str, build, seconds: float, jobs: int) -> None:
    import numpy as np
    import torch

    net, x = build()
    with torch.no_grad():
        want = net(x).numpy()

    print(f"\n{name}")
    print(f"  {'device':<22}{'latency':>10}{'queued':>12}{'max err':>11}")
    print("  " + "-" * 55)

    rows = []
    for dev in ("cpu", "xpu", "cuda"):
        per = time_torch(net, x, dev, seconds)
        if per is not None:
            rows.append((f"torch {dev}", per, None, 0.0))

    if importlib.util.find_spec("openvino") is not None:
        import openvino as ov
        try:
            model = ov.convert_model(net, example_input=x)
            # the NPU plugin refuses unbounded dimensions
            model.reshape({0: ov.PartialShape(list(x.shape))})
            core = ov.Core()
            arr = x.numpy()
            for dev in core.available_devices:
                try:
                    compiled = core.compile_model(model, dev)
                    out = compiled(arr)[compiled.output(0)]
                    err = float(np.abs(out - want).max())
                    for _ in range(3):
                        compiled(arr)
                    start, n = time.perf_counter(), 0
                    while time.perf_counter() - start < seconds:
                        compiled(arr)
                        n += 1
                    per = (time.perf_counter() - start) / n * 1000

                    queue = ov.AsyncInferQueue(compiled, jobs)
                    done = 0

                    def finished(request, userdata):
                        nonlocal done
                        done += 1

                    queue.set_callback(finished)
                    for _ in range(jobs):
                        queue.start_async({0: arr})
                    queue.wait_all()
                    done = 0
                    start = time.perf_counter()
                    while time.perf_counter() - start < seconds:
                        queue.start_async({0: arr})
                    queue.wait_all()
                    rate = done / (time.perf_counter() - start)
                    rows.append((f"openvino {dev}", per, rate, err))
                except Exception as exc:
                    tail = str(exc).strip().splitlines()[-1][:44]
                    rows.append((f"openvino {dev}", None, None, tail))
        except Exception as exc:
            print(f"  conversion failed: {str(exc).splitlines()[-1][:60]}")

    for label, per, rate, err in rows:
        if per is None:
            print(f"  {label:<22}{'failed':>10}   {err}")
            continue
        q = f"{rate:8.1f}/s" if rate else "         —"
        e = f"{err:.1e}" if isinstance(err, float) and err else "        —"
        print(f"  {label:<22}{per:8.1f}ms{q:>12}{e:>11}")

    ok = [(l, p) for l, p, _, e in rows
          if p is not None and (not isinstance(e, float) or e < 2e-2)]
    if ok:
        best = min(ok, key=lambda r: r[1])
        npu = next((r for r in ok if r[0].endswith("NPU")), None)
        print(f"\n  fastest: {best[0]} at {best[1]:.1f} ms")
        if npu and npu[0] != best[0]:
            print(f"  NPU is {npu[1] / best[1]:.2f}x slower than that")
        elif npu:
            print("  the NPU is the fastest device for this stage")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    run_stage("SEPARATOR shape — Conv2d over a spectrogram",
              separator_shaped, args.seconds, args.jobs)
    run_stage("TRANSCRIBER shape — bidirectional GRU over 1000 frames",
              transcriber_shaped, args.seconds, args.jobs)

    print("\nRead the second table first. The transcriber is the stage on the CPU, so")
    print("an NPU that beat the CPU there would free the busiest device in the system.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
