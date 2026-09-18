"""Is the drum extractor actually using the GPU, and does it help?

The pipeline patches audio-separator's device choice to include XPU, and htdemucs goes
through the same code path -- but Demucs is a different architecture from MDX23C, with
its own buffer handling, so "it should work" is not evidence. This times one extraction
on each device and checks the output is the same.

    python check_extractor_gpu.py
    python check_extractor_gpu.py --audio path/to/song.wav
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from drum2midi import HTDEMUCS_MODEL, extract_drums, xpu_available  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default = ROOT / "mdbdrums" / "MDB Drums" / "audio" / "full_mix"
    ap.add_argument("--audio", default=None)
    ap.add_argument("--model", default=HTDEMUCS_MODEL)
    args = ap.parse_args()

    src = Path(args.audio) if args.audio else next(default.glob("*.wav"), None)
    if src is None or not src.exists():
        print("no input audio found")
        return 1

    print(f"input : {src.name}")
    print(f"model : {args.model}")
    print(f"XPU   : {'available' if xpu_available() else 'not available'}\n")

    import soundfile as sf

    results = {}
    tmp = Path(tempfile.mkdtemp(prefix="extgpu_"))
    try:
        for device in (["cpu", "auto"] if xpu_available() else ["cpu"]):
            dest = tmp / f"{device}.wav"
            t0 = time.time()
            try:
                extract_drums(src, device, dest, args.model)
            except Exception as exc:
                print(f"  {device:<5} failed: {str(exc)[:90]}")
                continue
            elapsed = time.time() - t0
            audio, _ = sf.read(dest, dtype="float32", always_2d=True)
            results[device] = (elapsed, audio)
            print(f"  {device:<5} {elapsed:>7.1f} s")

        if len(results) == 2:
            (t_cpu, a_cpu), (t_gpu, a_gpu) = results["cpu"], results["auto"]
            n = min(len(a_cpu), len(a_gpu))
            x, y = a_cpu[:n].ravel(), a_gpu[:n].ravel()
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            corr = float(np.dot(x, y) / denom) if denom else float("nan")
            print(f"\nspeedup     : {t_cpu / t_gpu:.2f}x")
            print(f"correlation : {corr:.6f}  (1.0 means identical output)")
            if corr < 0.999:
                print("WARNING: the two stems differ audibly; the GPU path is not "
                      "equivalent.")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
