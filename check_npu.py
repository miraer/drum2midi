"""Can this machine's NPU run the separator, and would it be worth it?

**Answer: it is reachable, and there is nowhere useful to put it.** Measured on an Intel
Core Ultra 7 165H with an Arc iGPU and an AI Boost NPU. Four scripts were needed and the
order matters, because the cheap measurements pointed the wrong way twice.

This script reports what is installed and times a toy convolution stack:

    torch    xpu       14.7 ms per pass     <- what the pipeline uses today
    openvino NPU       32.0 ms
    openvino GPU       34.5 ms
    openvino CPU       94.1 ms
    torch    cpu       78.8 ms

`bench_npu_stages.py` adds the recurrent shape and a queued measurement. Two things there
look like openings and neither survives:

  * on a queue the NPU beats the GPU, 84.3 against 67.9 passes per second, because an NPU
    is a throughput device and the latency table understates it
  * a bare GRU runs 12-15x faster under OpenVINO than under torch on the *same CPU*,
    which would be enormous, since transcription is the slowest stage in the pipeline

`bench_rnn_runtime.py` confirms the 12-15x with repeats and a range, outputs agreeing to
1.5e-07. `bench_adtof_runtime.py` then runs the **real ADTOF checkpoint** and it comes out
at **0.9x** -- slightly slower than torch -- at 5, 15 and 36 seconds of audio, with zero
frames crossing a peak-picking threshold differently. The toy GRU was measuring the part
of ADTOF that is not the bottleneck; its convolutional front end dominates, and OpenVINO
does not improve it.

`bench_hetero.py` tries both devices together. `MULTI:GPU,NPU` is far worse than either
alone (29.9 against 84.3 passes per second) and `HETERO` loses too. They share memory
bandwidth, which is what this workload is limited by, so splitting it moves the bottleneck
rather than widening it.

What stands between here and using the NPU, none of it a flag:

  * torch has no backend for an Intel NPU, so reaching it means leaving torch for the run
  * the NPU plugin refuses dynamic shapes, so a fixed chunk shape must be committed to
  * the stock onnxruntime wheel reaches only CPU and Azure

**What has not been measured** is MDX23C itself through OpenVINO. The toy convolution says
the runtime penalty would swamp any device gain, but toy evidence already misled us once
here, so that is a gap rather than a conclusion. The one axis where an NPU should genuinely
win is power draw, which is not measured anywhere in this project.

    python check_npu.py
    python check_npu.py --bench          # also time it, if a runtime can reach it
"""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANDIDATES = [
    ("openvino", "Intel's own runtime; its NPU plugin is the supported path"),
    ("onnxruntime", "needs the OpenVINO or QNN execution provider built in"),
    ("torch_directml", "DirectX 12 compute; NPU coverage is partial"),
    ("intel_npu_acceleration_library", "Intel's PyTorch-facing NPU wrapper"),
]


def windows_npu() -> list[str]:
    """NPUs Windows itself reports, independent of any Python package."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-PnpDevice -PresentOnly -Class ComputeAccelerator | "
             "Select-Object -ExpandProperty FriendlyName"],
            capture_output=True, text=True, timeout=30)
        return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="store_true",
                    help="time a convolution stack wherever it can be run")
    args = ap.parse_args()

    print("hardware Windows reports")
    npus = windows_npu()
    for n in npus:
        print(f"  {n}")
    if not npus:
        print("  no compute accelerator class devices")

    print("\nruntimes that could address it")
    reachable = []
    for name, why in CANDIDATES:
        spec = importlib.util.find_spec(name)
        print(f"  {name:<34} {'installed' if spec else 'not installed':<14} {why}")
        if spec:
            reachable.append(name)

    if "openvino" in reachable:
        try:
            import openvino as ov
            devices = ov.Core().available_devices
            print(f"\n  openvino sees: {', '.join(devices)}")
            if not any(d.startswith("NPU") for d in devices):
                print("  ...but no NPU plugin among them")
        except Exception as exc:
            print(f"\n  openvino present but unusable: {exc}")

    if "onnxruntime" in reachable:
        try:
            import onnxruntime as ort
            provs = ort.get_available_providers()
            print(f"  onnxruntime {ort.__version__} providers: {', '.join(provs)}")
            if not any(p.startswith(("OpenVINO", "QNN", "Dml")) for p in provs):
                print("  ...none of which can reach an NPU; the stock wheel is CPU-only "
                      "plus Azure")
        except Exception as exc:
            print(f"  onnxruntime unusable: {exc}")

    print("\nwhat torch offers, for comparison")
    try:
        import torch
        print(f"  torch {torch.__version__}")
        print(f"    cuda {torch.cuda.is_available()}   "
              f"xpu {hasattr(torch, 'xpu') and torch.xpu.is_available()}")
        print("    torch has no backend for an Intel NPU; reaching it means leaving "
              "torch for the run")
    except Exception as exc:
        print(f"  torch unusable: {exc}")

    if args.bench:
        bench()

    print("\nverdict")
    if npus and not any(k in reachable for k in
                        ("openvino", "intel_npu_acceleration_library")):
        print("  The NPU exists and nothing installed can address it. Answering the")
        print("  question costs an install plus an ONNX export of the separator; it is")
        print("  not a flag that can be flipped.")
    elif not npus:
        print("  No NPU on this machine. Nothing to measure.")
    return 0


def bench(seconds: float = 2.0) -> None:
    """Time a separator-shaped Conv2d stack on every device that can run it."""
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return
    print("\nconvolution stack, the shape the separator is made of")
    net = nn.Sequential(
        nn.Conv2d(4, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
    ).eval()
    x = torch.randn(1, 4, 256, 256)
    for dev in ("cpu", "xpu", "cuda"):
        if dev == "xpu" and not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            continue
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        try:
            m, t = net.to(dev), x.to(dev)
            with torch.no_grad():
                for _ in range(3):
                    m(t)
                if dev != "cpu":
                    getattr(torch, dev).synchronize()
                start, n = time.perf_counter(), 0
                while time.perf_counter() - start < seconds:
                    m(t)
                    n += 1
                if dev != "cpu":
                    getattr(torch, dev).synchronize()
                per = (time.perf_counter() - start) / n * 1000
            print(f"  torch {dev:<10} {per:8.1f} ms per pass   ({n} passes)")
        except Exception as exc:
            print(f"  torch {dev:<10} unavailable: {exc}")
    net.to("cpu")
    _bench_openvino(net, x, seconds)


def _bench_openvino(net, x, seconds: float) -> None:
    """Same graph through OpenVINO, which is the only thing here that reaches the NPU.

    Compiled per device from the same converted model, so the comparison is between
    devices rather than between toolchains. Correctness is checked against torch first:
    a device that is fast and wrong is not a result.

    The reshape to a static shape is not tidying. The NPU plugin refuses a model with
    unbounded dimensions -- "Upper bounds were not specified" -- and a torch conversion
    leaves them dynamic by default. Any real use of the NPU inherits that constraint,
    which for a separator means committing to a fixed chunk shape.
    """
    if importlib.util.find_spec("openvino") is None:
        return
    import numpy as np
    import openvino as ov
    import torch

    try:
        model = ov.convert_model(net, example_input=x)
        model.reshape({0: ov.PartialShape(list(x.shape))})
    except Exception as exc:
        print(f"  openvino   conversion failed: {exc}")
        return

    with torch.no_grad():
        want = net(x).numpy()
    core = ov.Core()
    arr = x.numpy()

    for dev in core.available_devices:
        try:
            t0 = time.perf_counter()
            compiled = core.compile_model(model, dev)
            compile_s = time.perf_counter() - t0
            out = compiled(arr)[compiled.output(0)]
            err = float(np.abs(out - want).max())
            for _ in range(3):
                compiled(arr)
            start, n = time.perf_counter(), 0
            while time.perf_counter() - start < seconds:
                compiled(arr)
                n += 1
            per = (time.perf_counter() - start) / n * 1000
            flag = "" if err < 2e-2 else f"   <-- DIFFERS from torch by {err:.3g}"
            print(f"  openvino {dev:<8} {per:8.1f} ms per pass   "
                  f"(compile {compile_s:.1f}s, max abs err {err:.1e}){flag}")
        except Exception as exc:
            tail = str(exc).strip().splitlines()[-1][:78]
            print(f"  openvino {dev:<8} unavailable: {tail}")


if __name__ == "__main__":
    raise SystemExit(main())
