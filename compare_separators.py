"""Compare LarsNet and DrumSep on your own drum track.

Neither separator wins everywhere, so measure instead of guessing. For each drum
the script reports contrast: the median stem level at hits the transcriber found,
divided by the median level at random times. Roughly:

    > 10x   stem is clean, safe to drive velocity and articulation from it
    3-10x   usable but bleedy
    < 3x    the stem is mostly bleed - treat that instrument as unverified

Usage:  python compare_separators.py path\\to\\drums.wav
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SR = 44100
ROOT = Path(__file__).resolve().parent
PY = sys.executable

# ADTOF class -> (LarsNet stem, DrumSep stem). DrumSep has no separate hi-hat:
# both hats and cymbals come out of "platillos", which is the main reason the
# default pipeline uses LarsNet.
PAIRS = {
    35: ("kick", "bombo"),
    38: ("snare", "redoblante"),
    47: ("toms", "toms"),
    42: ("hihat", "platillos"),
    49: ("cymbals", "platillos"),
}
NAMES = {35: "kick", 38: "snare", 47: "toms", 42: "hi-hat", 49: "cymbals"}


def mono(path: Path) -> np.ndarray:
    x, _ = sf.read(str(path))
    return x.mean(axis=1) if x.ndim > 1 else x


def peak(sig: np.ndarray, t: float, w: float = 0.08) -> float:
    a, b = max(0, int(t * SR)), min(len(sig), int((t + w) * SR))
    return float(np.max(np.abs(sig[a:b]))) if b > a else 0.0


def contrast(sig: np.ndarray, times: list[float], rng) -> tuple[float, float, float]:
    if not times or len(sig) < SR:
        return 0.0, 0.0, 0.0
    hit = float(np.median([peak(sig, t) for t in times]))
    ref = float(np.median([peak(sig, float(t))
                           for t in rng.uniform(0, len(sig) / SR - 0.1, 300)]))
    return hit, ref, hit / max(ref, 1e-9)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0 if len(sys.argv) > 1 else 1
    src = Path(sys.argv[1]).resolve()
    if not src.exists():
        print(f"not found: {src}")
        return 1

    work = ROOT / "cmp" / src.stem
    work.mkdir(parents=True, exist_ok=True)
    mid = work / "ref.mid"

    print(f"[1/3] Transcribing {src.name} to locate the hits ...")
    subprocess.run([PY, str(ROOT / "drum2midi.py"), str(src), "-o", str(mid),
                    "--device", "auto", "--keep-stems", str(work / "larsnet")],
                   check=True, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")

    print("[2/3] Separating with DrumSep ...")
    ds_root = ROOT / "drumsep"
    subprocess.run([PY, "-m", "demucs", "--repo", "model", "-n", "49469ca8",
                    "-d", "cpu", "-o", str(work / "drumsep"), str(src)],
                   cwd=ds_root, check=True, capture_output=True, text=True,
                   encoding="utf-8", errors="replace")
    ds_dir = work / "drumsep" / "49469ca8" / src.stem

    print("[3/3] Measuring stem contrast ...\n")
    notes = pretty_midi.PrettyMIDI(str(mid)).instruments[0].notes
    families = {35: [35, 36], 38: [38, 40], 47: [41, 43, 45, 47, 48, 50],
                42: [42, 44, 46], 49: [49, 51]}
    rng = np.random.default_rng(3)

    print(f"{'drum':<10}{'hits':>6}{'LarsNet':>22}{'DrumSep':>22}")
    print(f"{'':<10}{'':>6}{'level':>11}{'contrast':>11}{'level':>11}{'contrast':>11}")
    print("-" * 60)

    verdict = []
    for cls, (l_stem, d_stem) in PAIRS.items():
        times = sorted(n.start for n in notes if n.pitch in families[cls])
        row = [f"{NAMES[cls]:<10}{len(times):>6}"]
        scores = {}
        for label, path in (("lars", work / "larsnet" / f"{l_stem}.wav"),
                            ("ds", ds_dir / f"{d_stem}.wav")):
            if not path.exists() or not times:
                row.append(f"{'-':>11}{'-':>11}")
                scores[label] = 0.0
                continue
            hit, _, c = contrast(mono(path), times, rng)
            row.append(f"{hit:>11.4f}{c:>10.1f}x")
            scores[label] = c
        print("".join(row))
        if times:
            verdict.append((NAMES[cls], scores["lars"], scores["ds"]))

    print("\nPer-drum winner (higher contrast = cleaner stem):")
    for name, lars, ds in verdict:
        if max(lars, ds) < 3.0:
            print(f"  {name:<9} neither ({lars:.1f}x / {ds:.1f}x) - both stems are mostly bleed, "
                  f"so hits on this drum are likely transcriber false positives")
        elif lars >= ds * 1.15:
            print(f"  {name:<9} LarsNet ({lars:.1f}x vs {ds:.1f}x)")
        elif ds >= lars * 1.15:
            print(f"  {name:<9} DrumSep ({ds:.1f}x vs {lars:.1f}x)")
        else:
            print(f"  {name:<9} tie ({lars:.1f}x vs {ds:.1f}x)")

    print("\nNote: DrumSep emits one 'platillos' stem for BOTH hi-hat and cymbals, so its "
          "\n      numbers for those two rows come from the same file and it cannot drive "
          "\n      open/closed hi-hat detection. That is why drum2midi.py defaults to LarsNet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
