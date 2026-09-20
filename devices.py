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
that XPU and MPS do not. That was the documented behaviour of those libraries rather
than a measurement until an RTX 4070 SUPER was benchmarked: the real transcriber runs
74.9 ms there against 4766.0 ms on the CPU, 63.6x faster, where the same model is 4.1x
slower on the Intel GPU. `--device cuda` forces it.
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


def interpreter_here() -> str:
    """The interpreter this is running in, as something that can be typed back.

    Advice that starts with a bare `python` or `pip3` is advice about whichever
    interpreter PATH happens to point at, which on Windows is usually not the
    project's virtualenv. Naming the one in hand removes the guess.
    """
    import sys
    from pathlib import Path

    exe = Path(sys.executable)
    try:
        return str(Path(".") / exe.relative_to(Path(__file__).resolve().parent))
    except ValueError:
        return str(exe)


def pip_here() -> str:
    """The pip invocation that reaches the interpreter this is running in.

    pytorch.org hands you `pip3 install torch --index-url ...`, which is right in
    general and wrong for a project that lives in a virtualenv. A bare `pip3` on
    Windows is usually the system Python, so the CUDA wheel installs perfectly into an
    interpreter the pipeline never looks at, and the only symptom is the message this
    module prints -- a GPU the OS can see and torch cannot.

    That is not hypothetical: it produced torch 2.14.0+cu132 with CUDA working in one
    interpreter and 2.14.0+cpu in the project's .venv at the same time. So every
    instruction here names an interpreter rather than trusting PATH.
    """
    return f"{interpreter_here()} -m pip"


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
                    f"{why} (torch {version}). Install a CUDA build into this\n"
                    f"interpreter:\n"
                    f"       https://pytorch.org/get-started/locally/ gives the exact\n"
                    f"       command for your driver. The CUDA version in the index\n"
                    f"       URL matters and is not named here: it goes stale faster\n"
                    f"       than this file does.\n"
                    f"       Run it as `{pip_here()} install ...`, not as a bare\n"
                    f"       `pip3 install ...`. pytorch.org gives the bare form and\n"
                    f"       on Windows that is usually a different Python: the wheel\n"
                    f"       installs perfectly into an interpreter this project never\n"
                    f"       opens, and the only symptom is this message again.")
        if "intel" in low and ("arc" in low or "iris" in low):
            return (card,
                    f"this torch is not the XPU build (torch {version}). Install it\n"
                    f"with:  {interpreter_here()} setup_env.py --with-intel-gpu")
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
        # line by line: the hint is several lines and only the first was indented,
        # which broke the alignment exactly where a command has to be copied out
        for line in how.splitlines():
            print(f"      {line.strip()}")
