"""Does tilting the spectrum let ADTOF hear the onsets it is blind to?

The deaf passages (ROADMAP, "Passages the transcriber cannot hear at all") have a dull
spectrum, centroid 2129 Hz against 4037 Hz in loud sections, and ADTOF's activations
there are noise, 0.008 to 0.08. One untried direction is pre-emphasis ahead of ADTOF. This
is the cheap necessary condition for it: if a tilt does not lift the blind onsets to
threshold, there is nothing for a full-pipeline run to find.

Fixed in advance, with no tuning:

- audible onsets come from `blind_spots.onset_peaks` on the ORIGINAL audio, so every arm
  shares a denominator;
- the arms are y[n] = x[n] - a*x[n-1] with a = 0.5 and 0.97, RMS-matched to the original
  so only the spectral shape changes;
- an onset is blind when every class stays under half its threshold (the
  `blind_spots` definition), and recovered when an originally blind onset reaches
  threshold in the arm;
- an arm passes when it recovers at least half the blind onsets, pooled over MDB (23)
  and the ENST default population (210).

Passing ships nothing. It only earns a full-pipeline F1 comparison with its own rule.

    python preemphasis_gate.py
    python preemphasis_gate.py --limit 5
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

import blind_spots  # noqa: E402
import drum2midi  # noqa: E402
from benchmark_enst import DEFAULT_KINDS, DEFAULT_POPULATION, recordings  # noqa: E402
from benchmark_mdb import AUDIO  # noqa: E402

ARMS = (0.5, 0.97)
FPS = 100
WINDOW = 0.06
PASS_SHARE = 0.5


def preemphasis(audio: np.ndarray, a: float) -> np.ndarray:
    """First-order tilt per channel, RMS-matched so only the shape changes."""
    out = audio.copy()
    out[1:] = audio[1:] - a * audio[:-1]
    rms_in = float(np.sqrt(np.mean(audio ** 2)))
    rms_out = float(np.sqrt(np.mean(out ** 2)))
    if rms_out > 0:
        out *= rms_in / rms_out
    peak = float(np.abs(out).max())
    if peak > 0.999:
        out *= 0.999 / peak
    return out


def ratios(act: np.ndarray, times: np.ndarray, thr: np.ndarray) -> np.ndarray:
    half = int(round(WINDOW * FPS))
    out = []
    for t in times:
        c = int(round(t * FPS))
        lo, hi = max(0, c - half), min(act.shape[0], c + half + 1)
        out.append(float((act[lo:hi].max(axis=0) / thr).max()) if lo < hi else np.nan)
    return np.array(out)


def measure(wav: Path, thr: np.ndarray, cache: Path, tmp: Path) -> dict:
    if cache.exists():
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    times, _s, _d = blind_spots.onset_peaks(wav)
    res = {"base": ratios(blind_spots.activations(wav), times, thr)}
    audio, sr = sf.read(str(wav), always_2d=True)
    for a in ARMS:
        f = tmp / f"{wav.stem}_{a}.wav"
        sf.write(str(f), preemphasis(audio.astype(np.float64), a), sr, subtype="FLOAT")
        res[f"a{a}"] = ratios(blind_spots.activations(f), times, thr)
        f.unlink()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **res)
    return res


def boot_share(num: np.ndarray, den: np.ndarray, rounds: int = 4000, seed: int = 0):
    """Pooled share with a 95% interval over recordings, not onsets."""
    keep = den > 0
    num, den = num[keep], den[keep]
    if not den.size:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, den.size, size=(rounds, den.size))
    s = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return float(num.sum() / den.sum()), (float(np.percentile(s, 2.5)),
                                          float(np.percentile(s, 97.5)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--cache", type=Path, default=ROOT / "bench" / "preemph_gate")
    args = ap.parse_args()

    thr = np.array(drum2midi.DEFAULT_THRESHOLDS, dtype=float)
    enst = recordings(ROOT / "enst" / "enst_drums_public", DEFAULT_KINDS, "wet_mix")
    items = [("mdb", w.stem, w) for w in sorted(AUDIO.glob("*.wav"))]
    items += [("enst", f"{a.stem}_d{d}", w) for d, a, w, _k in enst]
    if args.limit:
        items = items[:args.limit]
    n_mdb = sum(1 for c, _n, _w in items if c == "mdb")
    print(f"{n_mdb} MDB + {len(items) - n_mdb} ENST recordings, "
          f"thresholds {[float(x) for x in thr]}, arms a={list(ARMS)}")
    if not args.limit and len(enst) != DEFAULT_POPULATION["recordings"]:
        print(f"!! POPULATION MISMATCH: {len(enst)} ENST recordings, not "
              f"{DEFAULT_POPULATION['recordings']}")
    print(f"\n{'recording':<44}{'onsets':>7}{'blind':>6}"
          + "".join(f"{'rec a=' + str(a):>11}{'lost':>6}" for a in ARMS))

    rows = []
    with tempfile.TemporaryDirectory() as td:
        for corpus, name, wav in items:
            r = measure(wav, thr, args.cache / f"{corpus}_{name}.npz", Path(td))
            base = r["base"]
            blind = base < 0.5
            seen = base >= 1.0
            row = {"corpus": corpus, "name": name, "n": int(np.isfinite(base).sum()),
                   "blind": int(blind.sum()), "seen": int(seen.sum())}
            line = f"{corpus + ' ' + name:<44}{row['n']:>7}{row['blind']:>6}"
            for a in ARMS:
                arm = r[f"a{a}"]
                row[f"rec{a}"] = int((blind & (arm >= 1.0)).sum())
                row[f"lost{a}"] = int((seen & (arm < 1.0)).sum())
                line += f"{row[f'rec{a}']:>11}{row[f'lost{a}']:>6}"
            rows.append(row)
            print(line, flush=True)

    print()
    for corpus in ("mdb", "enst", "both"):
        sel = [r for r in rows if corpus == "both" or r["corpus"] == corpus]
        blind = np.array([r["blind"] for r in sel], dtype=int)
        seen = np.array([r["seen"] for r in sel], dtype=int)
        print(f"{corpus}: {len(sel)} recordings, {blind.sum()} blind onsets "
              f"in {int((blind > 0).sum())} of them, {seen.sum()} seen")
        for a in ARMS:
            rec = np.array([r[f"rec{a}"] for r in sel], dtype=int)
            lost = np.array([r[f"lost{a}"] for r in sel], dtype=int)
            rs, rci = boot_share(rec, blind)
            ls, lci = boot_share(lost, seen)
            verdict = ""
            if corpus == "both":
                verdict = "  PASS" if rs >= PASS_SHARE else "  FAIL"
            print(f"  a={a:<5} recovered {rec.sum():>5} = {rs:6.1%} "
                  f"[{rci[0]:.1%}, {rci[1]:.1%}]   seen->unseen {lost.sum():>5} = "
                  f"{ls:6.1%} [{lci[0]:.1%}, {lci[1]:.1%}]{verdict}")
    print(f"\nPASS needs >= {PASS_SHARE:.0%} of the blind onsets recovered, pooled over "
          f"both corpora. The seen->unseen column is a cost hint, not the cost:\n"
          f"a class falling under threshold is not a wrong note, and a note above "
          f"threshold is not a right one. Only full-pipeline F1 settles that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
