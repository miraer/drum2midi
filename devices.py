"""Picks the right device for each stage of the pipeline.

"Use the GPU if there is one" is the wrong rule here, because the two neural stages
have opposite hardware profiles. Measured on this machine (Intel Arc, torch 2.14+xpu):

    workload                          CPU        Intel GPU
    separator-like Conv2d stack      86.7 ms      23.8 ms     GPU 3.6x faster
    MDX23C, whole track             543.5 s       44.7 s      GPU 12.2x faster
    GRU over a long sequence        225.7 ms    1378.8 ms     GPU 6.1x SLOWER
    ADTOF, the real model          1408.4 ms    5731.3 ms     GPU 4.1x SLOWER

A recurrent network is a long chain of tiny dependent kernels: each step waits for the
previous one, so launch overhead dominates and an integrated GPU loses to the CPU.
Convolution over a spectrogram is the opposite -- large independent tiles.

Hence the default is split: separation on the GPU, transcription on the CPU. On the
measured track that turns ~9.5 minutes into ~1.5 minutes without changing a single note
of the output (stem correlation 1.000000).

CUDA is treated as fast for both stages because cuDNN provides fused recurrent kernels
that XPU and MPS do not. That is the documented behaviour of those libraries rather than
something measured here -- this machine has no NVIDIA GPU. `--device cuda` forces it.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# Stages that can run somewhere other than the CPU.
SEPARATOR = "separator"
TRANSCRIBER = "transcriber"

# Which backends are worth using for a recurrent model. Everything else keeps the
# transcriber on the CPU, because dispatch overhead outweighs the arithmetic.
_FAST_FOR_RNN = {"cuda"}


def available_devices() -> List[str]:
    """Accelerators torch can actually use right now, best first."""
    try:
        import torch
    except Exception:
        return ["cpu"]

    found = []
    if torch.cuda.is_available():
        found.append("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        found.append("xpu")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        found.append("mps")
    found.append("cpu")
    return found


def describe_device(device: str) -> str:
    """A human-readable name, e.g. 'Intel(R) Arc(TM) Graphics'."""
    try:
        import torch
        if device == "cuda" and torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.xpu.get_device_properties(0).name
        if device == "mps":
            return "Apple GPU (Metal)"
    except Exception:
        pass
    if device == "cpu":
        import platform
        return platform.processor() or "CPU"
    return device


def pick_devices(requested: str = "auto") -> Dict[str, str]:
    """Maps each stage to a device.

    `auto` splits the work by what each stage is made of. Any explicit value is
    obeyed everywhere, including a deliberately slow one -- being able to force a
    device is what makes the automatic choice checkable.
    """
    if requested and requested != "auto":
        return {SEPARATOR: requested, TRANSCRIBER: requested}

    devices = available_devices()
    accelerator = next((d for d in devices if d != "cpu"), "cpu")
    return {
        SEPARATOR: accelerator,
        TRANSCRIBER: accelerator if accelerator in _FAST_FOR_RNN else "cpu",
    }


def unused_gpu() -> Optional[Tuple[str, str]]:
    """A GPU the operating system can see and torch cannot, with what to do about it.

    `available: cpu` on a machine with a GeForce 4070 reads as a statement about the
    hardware when it is really a statement about the torch build: the default wheel on
    PyPI for Windows carries no CUDA at all. Looking through the OS rather than through
    torch is the whole point, since torch is the thing that is blind here.

    Returns (what the OS reports, how to fix it) or None.
    """
    import os
    import subprocess

    try:
        import torch
        version = torch.__version__
        if torch.cuda.is_available():
            return None
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return None
    except Exception:
        version = "not installed"

    names = ""
    if os.name == "nt":
        try:
            names = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=30).stdout
        except Exception:
            names = ""
    else:
        try:
            names = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                   text=True, timeout=30).stdout
        except Exception:
            names = ""

    for line in names.splitlines():
        card = line.strip()
        low = card.lower()
        if any(k in low for k in ("nvidia", "geforce", "quadro", "rtx", "tesla")):
            why = ("this torch has no CUDA support"
                   if "+cpu" in version or "cu" not in version
                   else "torch reports CUDA unavailable; check the driver")
            return (card,
                    f"{why} (torch {version}). Install the CUDA build:\n"
                    f"       pip install --force-reinstall torch "
                    f"--index-url https://download.pytorch.org/whl/cu124\n"
                    f"       or pick your version at "
                    f"https://pytorch.org/get-started/locally/")
        if "intel" in low and ("arc" in low or "iris" in low):
            return (card,
                    f"this torch is not the XPU build (torch {version}). Install it "
                    f"with:\n       python setup_env.py --with-intel-gpu")
    return None


def summary(requested: str = "auto") -> str:
    """One line for the log: what was chosen and why."""
    picked = pick_devices(requested)
    sep, trans = picked[SEPARATOR], picked[TRANSCRIBER]
    if sep == trans:
        return f"device: {sep} ({describe_device(sep)})"
    return (f"device: separation on {sep} ({describe_device(sep)}), "
            f"transcription on {trans} - recurrent layers are faster there")


if __name__ == "__main__":
    import sys
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        raise SystemExit(0)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"available: {', '.join(available_devices())}")
    for d in available_devices():
        print(f"  {d:<6} {describe_device(d)}")
    print()
    print(summary("auto"))
    idle = unused_gpu()
    if idle:
        card, how = idle
        print(f"\nnote: {card} is installed and is not being used.")
        print(f"      {how}")
