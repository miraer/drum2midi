"""Cache MDX23C stems for the whole MDB Drums set, once.

MDX23C is roughly 10x slower than real time on CPU, and every experiment so far has
paid that cost again from scratch. Separating once and keeping the stems on disk makes
all later fusion analysis effectively free.

Resumable: tracks that already have a full set of stems are skipped.

    python cache_uvr_stems.py
    python cache_uvr_stems.py --limit 4      # partial run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "mdbdrums" / "MDB Drums" / "audio" / "drum_only"
CACHE = ROOT / "bench" / "uvr_stems"
MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
STEMS = {"kick": "kick", "snare": "snare", "toms": "toms",
         "hh": "hihat", "crash": "cymbals", "ride": "ride"}


def complete(dest: Path) -> bool:
    return dest.is_dir() and all((dest / f"{n}.flac").exists() for n in STEMS.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tracks = sorted(AUDIO.glob("*.wav"))[: args.limit]
    todo = [w for w in tracks if not complete(CACHE / w.stem)]
    print(f"{len(tracks)} tracks, {len(tracks) - len(todo)} already cached, "
          f"{len(todo)} to separate")
    if not todo:
        return 0

    from audio_separator.separator import Separator
    from xpu_separate import patch_audio_separator

    if patch_audio_separator():
        print("  separator: Intel GPU (XPU)")

    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="uvrcache_"))
    t0 = time.time()
    try:
        sep = Separator(output_dir=str(tmp), model_file_dir=str(ROOT / "uvr" / "models"),
                        log_level=40)
        sep.load_model(model_filename=MODEL)
        for i, wav in enumerate(todo, 1):
            dest = CACHE / wav.stem
            dest.mkdir(parents=True, exist_ok=True)
            produced = sep.separate(str(wav))
            kept = 0
            for f in produced:
                path = Path(f) if Path(f).is_absolute() else tmp / f
                if not path.exists():
                    continue
                for raw, name in STEMS.items():
                    if f"({raw})" in path.name:
                        data, sr = sf.read(str(path), always_2d=True)
                        sf.write(str(dest / f"{name}.flac"), data, sr)
                        kept += 1
                path.unlink(missing_ok=True)
            done = i
            rate = (time.time() - t0) / done
            left = rate * (len(todo) - done) / 60
            print(f"  [{i:2d}/{len(todo)}] {wav.stem:<34} {kept} stems  "
                  f"({rate:.0f}s/track, ~{left:.0f} min left)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    size = sum(f.stat().st_size for f in CACHE.rglob("*.flac")) / 1024 / 1024
    print(f"\ncached {len(list(CACHE.iterdir()))} tracks, {size:.0f} MB -> {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
