"""Teaches audio-separator to use an Intel GPU (XPU), and measures whether it helps.

Yesterday's conclusion was that the separator cannot use this GPU. That was half right:
the library picks its device with a plain `if cuda / elif mps / else cpu`, so it never
considers XPU. But the rest of it is already device-agnostic -- the hot path always goes
through `self.torch_device`, there is a working precedent for a third backend
(DirectML), and `device_utils._supports_complex_spectral_ops` probes STFT/iSTFT on
whatever device it is given and falls back on its own.

So the change needed is four lines, applied here as a monkey patch so the installed
package stays untouched and upgradable.

The same hook does a second job: `force_cpu_audio_separator()` pins the separator to
the CPU, because `if cuda / elif mps / else cpu` has no way to be told "the CPU,
please", and `drum2midi --device cpu` therefore did not apply to this stage at all.

What is deliberately NOT assumed: that it will be faster. Conv2d measured 7.7x faster on
this GPU, but the model also runs STFT, complex arithmetic and attention, and an
integrated GPU shares bandwidth with the CPU. This script measures end-to-end time and
checks that the stems come out the same.

    python xpu_separate.py --list          # show what this machine offers
    python xpu_separate.py audio.wav       # CPU vs XPU, timed and compared
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"


def xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def patch_audio_separator() -> bool:
    """Adds an XPU branch to the device selection. Returns whether XPU will be used."""
    from audio_separator.separator.separator import Separator

    if getattr(Separator, "_xpu_patched", False):
        return xpu_available()

    original = Separator.setup_torch_device

    def setup_torch_device(self, system_info):
        if xpu_available():
            self.torch_device_cpu = torch.device("cpu")
            self.torch_device = torch.device("xpu")
            # ONNX Runtime has no XPU provider here; only the torch path is accelerated,
            # which is what MDX23C uses.
            self.onnx_execution_provider = ["CPUExecutionProvider"]
            self.logger.info("XPU (Intel GPU) is available in Torch, using it")
            return
        return original(self, system_info)

    Separator.setup_torch_device = setup_torch_device
    Separator._xpu_patched = True
    return xpu_available()


def force_cpu_audio_separator() -> None:
    """Pins audio-separator to the CPU whatever the machine has.

    The opposite errand to the patch above and the same hook, which is why it lives
    here: this module is the one place that knows the name of the method
    audio-separator picks its device in, and a rename upstream should break one file.

    It is needed because the library has no CPU option at all -- `audio-separator
    --help` offers DirectML and fp16 and nothing for the CPU -- and chooses for itself
    with `if cuda / elif mps / else cpu`. So `drum2midi --device cpu` reached every
    stage except the separator, which is the longest of them.

    CUDA_VISIBLE_DEVICES is the usual advice and is not good enough. The empty string
    that gets quoted everywhere leaves torch.cuda.is_available() True on torch 2.14 --
    measured, not assumed -- `-1` does work for CUDA but says nothing about MPS or
    XPU, and patching the choice covers all three.
    """
    from audio_separator.separator.separator import Separator

    if getattr(Separator, "_cpu_patched", False):
        return

    def setup_torch_device(self, system_info):
        self.torch_device_cpu = torch.device("cpu")
        self.torch_device = torch.device("cpu")
        self.onnx_execution_provider = ["CPUExecutionProvider"]
        self.logger.info("pinned to CPU by drum2midi --device cpu")

    Separator.setup_torch_device = setup_torch_device
    Separator._cpu_patched = True


def describe() -> None:
    print(f"torch {torch.__version__}")
    print(f"  CUDA : {torch.cuda.is_available()}")
    print(f"  XPU  : {xpu_available()}")
    if xpu_available():
        for i in range(torch.xpu.device_count()):
            props = torch.xpu.get_device_properties(i)
            total = getattr(props, "total_memory", 0) / 1024 ** 3
            print(f"    [{i}] {props.name}  {total:.1f} GB")
    from audio_separator.separator.uvr_lib_v5 import device_utils
    for name in ("cpu", "xpu"):
        if name == "xpu" and not xpu_available():
            continue
        dev = torch.device(name)
        ok = not device_utils.should_fallback_to_cpu_for_complex_ops(dev)
        print(f"  complex STFT/iSTFT on {name}: {'yes' if ok else 'NO -> would fall back'}")


def run(src: Path, device: str, outdir: Path) -> tuple[float, dict]:
    """Separates one file and returns elapsed seconds plus the stems."""
    import soundfile as sf
    from audio_separator.separator import Separator

    if device == "xpu":
        patch_audio_separator()
    outdir.mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(outdir), model_file_dir=str(ROOT / "uvr" / "models"),
                    log_level=40)
    sep.load_model(model_filename=MODEL)
    actual = getattr(sep, "torch_device", None)
    print(f"  requested {device}, separator is using: {actual}")

    t0 = time.time()
    files = sep.separate(str(src))
    elapsed = time.time() - t0

    stems = {}
    for f in files:
        p = outdir / f if not Path(f).is_absolute() else Path(f)
        if p.exists():
            audio, _ = sf.read(p, dtype="float32")
            # "demo_(crash)_MDX23C-DrumSep-aufr33-jarredou.wav" -> "crash"
            key = p.stem.split("_(")[-1].split(")")[0] if "_(" in p.stem else p.stem
            stems[key] = audio
    return elapsed, stems


def compare(a: dict, b: dict) -> None:
    print(f"\n{'stem':<16}{'max abs diff':>14}{'correlation':>14}")
    print("-" * 44)
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        n = min(len(x), len(y))
        x, y = np.asarray(x[:n]).ravel(), np.asarray(y[:n]).ravel()
        diff = float(np.max(np.abs(x - y)))
        denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        corr = float(np.dot(x, y) / denom) if denom else float("nan")
        print(f"{key:<16}{diff:>14.6f}{corr:>14.6f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="?", help="file to separate")
    ap.add_argument("--list", action="store_true", help="show devices and exit")
    ap.add_argument("--only", choices=["cpu", "xpu"], help="run just one device")
    args = ap.parse_args()

    if args.list or not args.audio:
        describe()
        return 0

    src = Path(args.audio).resolve()
    if not src.exists():
        print(f"not found: {src}")
        return 1
    describe()

    devices = [args.only] if args.only else ["cpu", "xpu"]
    if "xpu" in devices and not xpu_available():
        print("\nXPU not available in this interpreter; nothing to compare.")
        return 1

    times, results = {}, {}
    for dev in devices:
        print(f"\n=== {dev} ===")
        times[dev], results[dev] = run(src, dev, ROOT / "bench" / f"xpu_{dev}")
        print(f"  {times[dev]:.1f} s")

    if len(times) == 2:
        speedup = times["cpu"] / times["xpu"] if times["xpu"] else float("nan")
        print(f"\nCPU {times['cpu']:.1f} s   XPU {times['xpu']:.1f} s   "
              f"speedup {speedup:.2f}x")
        compare(results["cpu"], results["xpu"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
