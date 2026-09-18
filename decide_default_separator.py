"""Which separator should be the default - a head-to-head run.

Compares larsnet and uvr end to end on MDB Drums so the default rests on numbers rather
than habit. MDX23C stems come from the cache, so only transcription is recomputed.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "mdbdrums" / "MDB Drums" / "audio" / "drum_only"
PY = sys.executable


def run(sep: str, outdir: Path, limit: int | None) -> float:
    outdir.mkdir(parents=True, exist_ok=True)
    wavs = sorted(AUDIO.glob("*.wav"))[:limit]
    t0 = time.time()
    for i, wav in enumerate(wavs, 1):
        out = outdir / f"{wav.stem}.mid"
        if out.exists():
            continue
        cmd = [PY, str(ROOT / "drum2midi.py"), str(wav), "-o", str(out),
               "--device", "auto", "--separator", sep]
        subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
        print(f"  [{i:2d}/{len(wavs)}] {sep:<8} {wav.stem}", flush=True)
    return time.time() - t0


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    timings = {}
    for sep in ("larsnet", "uvr"):
        outdir = ROOT / "bench" / f"default_{sep}"
        timings[sep] = run(sep, outdir, limit)
        print(f"{sep}: {timings[sep]:.0f} s")

    print("\nto compare, run:")
    print("  python compare_with_restem.py --ours bench/default_larsnet")
    print("  python compare_with_restem.py --ours bench/default_uvr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
