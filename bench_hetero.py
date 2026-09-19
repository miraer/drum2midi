"""Does running the separator's workload on the GPU *and* the NPU together beat the GPU?

Measured separately, the NPU is slower than the Arc iGPU on this workload, so it is not a
replacement. But the separator processes audio in chunks, which makes each chunk an
independent inference request -- and OpenVINO's MULTI plugin spreads a queue of requests
across several devices at once. Two devices that are each slower than one fast device can
still beat it on total throughput, so the question is worth a measurement rather than an
opinion.

The comparison has to be like for like, and that is the whole difficulty:

  * MULTI only helps a *queue*. Timing one synchronous request at a time measures
    latency, where a second device cannot help by construction, and would report MULTI as
    worthless for reasons that have nothing to do with MULTI.
  * so every device is timed the same way, through AsyncInferQueue with the same number
    of requests in flight, and scored on completed passes per second.
  * and the single-device numbers are re-measured here rather than copied from
    check_npu.py, because that script times synchronously and the two are not comparable.

    python bench_hetero.py
    python bench_hetero.py --seconds 4 --jobs 8
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def conv_stack():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(4, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
    ).eval()


def throughput(core, model, device: str, arr, seconds: float, jobs: int):
    """Completed passes per second with `jobs` requests in flight."""
    import openvino as ov

    compiled = core.compile_model(model, device)
    queue = ov.AsyncInferQueue(compiled, jobs)
    done = 0

    def finished(request, userdata):
        nonlocal done
        done += 1

    queue.set_callback(finished)

    # warm up: first requests on a fresh device include allocation
    for _ in range(jobs):
        queue.start_async({0: arr})
    queue.wait_all()
    done = 0

    start = time.perf_counter()
    deadline = start + seconds
    while time.perf_counter() < deadline:
        queue.start_async({0: arr})
    queue.wait_all()
    elapsed = time.perf_counter() - start
    return done / elapsed, done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--jobs", type=int, default=4,
                    help="requests in flight; MULTI needs at least as many as devices")
    args = ap.parse_args()

    if importlib.util.find_spec("openvino") is None:
        print("openvino is not installed; nothing here can reach an NPU")
        return 1

    import numpy as np
    import openvino as ov
    import torch

    net = conv_stack()
    x = torch.randn(1, 4, 256, 256)
    with torch.no_grad():
        want = net(x).numpy()

    model = ov.convert_model(net, example_input=x)
    # the NPU plugin refuses unbounded dimensions, and MULTI inherits that
    model.reshape({0: ov.PartialShape(list(x.shape))})
    arr = x.numpy()

    core = ov.Core()
    devices = core.available_devices
    print(f"devices: {', '.join(devices)}")
    print(f"{args.jobs} requests in flight, {args.seconds:g}s per configuration\n")

    plans = [d for d in devices if d != "CPU"]
    if len(plans) > 1:
        plans.append("MULTI:" + ",".join(plans))
        plans.append("HETERO:" + ",".join(plans[:-1]))
    plans.append("CPU")

    print(f"  {'configuration':<26}{'passes/s':>10}{'vs best single':>16}")
    print("  " + "-" * 52)
    best_single = 0.0
    results = {}
    for dev in plans:
        try:
            compiled = core.compile_model(model, dev)
            out = compiled(arr)[compiled.output(0)]
            err = float(np.abs(out - want).max())
            if err > 2e-2:
                print(f"  {dev:<26}{'WRONG':>10}   max abs err {err:.3g}")
                continue
            rate, n = throughput(core, model, dev, arr, args.seconds, args.jobs)
            results[dev] = rate
            if ":" not in dev and dev != "CPU":
                best_single = max(best_single, rate)
            ratio = f"{rate / best_single:+.2f}x" if best_single and ":" in dev else ""
            print(f"  {dev:<26}{rate:10.1f}{ratio:>16}")
        except Exception as exc:
            tail = str(exc).strip().splitlines()[-1][:60]
            print(f"  {dev:<26}{'failed':>10}   {tail}")

    combos = {k: v for k, v in results.items() if ":" in k}
    if combos and best_single:
        best_combo = max(combos, key=combos.get)
        gain = combos[best_combo] / best_single
        print()
        if gain > 1.05:
            print(f"  {best_combo} beats the best single device by {gain:.2f}x.")
            print("  Worth pursuing only if the separator can be made to queue chunks")
            print("  asynchronously, which it currently does not.")
        else:
            print(f"  No combination beats the best single device ({gain:.2f}x).")
            print("  Both devices share memory bandwidth, which is what this workload")
            print("  is actually limited by, so splitting it moves the bottleneck")
            print("  rather than widening it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
