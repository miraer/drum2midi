"""Does confirming a tom against its own separated stem survive being held out?

The finding it tests: at an emitted tom onset, energy in the separated tom stem --
normalised by that stem's own overall level -- separates real toms from misclassified
ghost notes almost completely. On MDB that takes tom F1 from 0.589 to 0.766.

That number is worthless on its own, because the threshold producing it was chosen by
looking at the same tracks it was scored on. Every threshold policy in this project has
had to survive being fitted on one corpus and tested on the other, and the tom ceiling
bug is what happens when one does not.

So: fit on ENST, report on MDB, and the reverse. The number that means anything is the
held-out one, and it is printed next to the fitted one so the gap between them is visible
rather than quietly omitted.

    python stem_confirm_validate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark_mdb import ANN as MDB_ANN, WINDOW, read_annotation, read_estimate  # noqa: E402
from benchmark_enst import IGNORED, LABEL_TO_CLASS  # noqa: E402

ENST = ROOT / "enst" / "enst_drums_public"
SWEEP = np.round(np.arange(0.0, 4.01, 0.05), 2)


def energy(a: np.ndarray, sr: int, t: float, pre: float = 0.005,
           post: float = 0.045) -> float:
    i0, i1 = max(0, int((t - pre) * sr)), min(len(a), int((t + post) * sr))
    return float(np.sqrt((a[i0:i1] ** 2).mean())) if i1 > i0 else 0.0


def collect_mdb() -> list[tuple[np.ndarray, np.ndarray, int]]:
    out = []
    ship, stems = ROOT / "bench" / "ceiling", ROOT / "bench" / "uvr_stems"
    for mid in sorted(ship.glob("*.mid")):
        ann = MDB_ANN / f"{mid.stem.replace('_Drum', '')}_class.txt"
        sd = stems / mid.stem
        if not ann.exists() or not sd.is_dir():
            continue
        ref = read_annotation(ann)["TT"]
        est = read_estimate(mid)["TT"]
        rows = score_rows(est, ref, sd)
        if rows is not None:
            out.append((*rows, len(ref)))
    return out


def enst_toms(name: str) -> np.ndarray | None:
    for d in (1, 2, 3):
        p = ENST / f"drummer_{d}" / "annotation" / f"{name}.txt"
        if p.exists():
            return np.array(sorted(
                float(l.split()[0]) for l in p.read_text(encoding="utf-8",
                                                         errors="replace").splitlines()
                if len(l.split()) >= 2 and l.split()[1] not in IGNORED
                and LABEL_TO_CLASS.get(l.split()[1]) == "TT"))
    return None


def collect_enst(names: list[str]) -> list[tuple[np.ndarray, np.ndarray, int]]:
    import pretty_midi
    out = []
    sep, stems = ROOT / "bench" / "enst60_sep", ROOT / "bench" / "enst_stems"
    for name in names:
        mids = list(sep.glob(f"{name}*.mid"))
        sd = stems / name
        ref = enst_toms(name)
        if not mids or not sd.is_dir() or ref is None or not len(ref):
            continue
        pm = pretty_midi.PrettyMIDI(str(mids[0]))
        est = np.array(sorted(n.start for i in pm.instruments for n in i.notes
                              if n.pitch in (41, 43, 45, 47, 48, 50)))
        rows = score_rows(est, ref, sd)
        if rows is not None:
            out.append((*rows, len(ref)))
    return out


def score_rows(est: np.ndarray, ref: np.ndarray, sd: Path):
    """Per emitted tom: its stem energy, and whether it is real."""
    if not len(est):
        return None
    try:
        tm, sr = sf.read(str(sd / "toms.flac"))
    except Exception:
        return None
    if tm.ndim > 1:
        tm = tm.mean(axis=1)
    overall = float(np.sqrt((tm ** 2).mean())) or 1e-9
    e = np.array([energy(tm, sr, float(t)) / overall for t in est])
    ok = np.array([bool(len(ref)) and bool(np.abs(ref - t).min() <= WINDOW) for t in est])
    return e, ok


def curve(data) -> tuple[np.ndarray, np.ndarray]:
    ref_total = sum(d[2] for d in data)
    scores = []
    for k in SWEEP:
        tp = sum(int(np.sum(d[1] & (d[0] >= k))) for d in data)
        est = sum(int(np.sum(d[0] >= k)) for d in data)
        p = tp / max(est, 1)
        r = tp / max(ref_total, 1)
        scores.append(2 * p * r / (p + r) if p + r else 0.0)
    return SWEEP, np.array(scores)


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter
                            ).parse_args()

    fit_list = ROOT / "bench" / "enst_tomfit.txt"
    names = [n.strip() for n in fit_list.read_text().splitlines() if n.strip()] \
        if fit_list.exists() else []

    mdb, enst = collect_mdb(), collect_enst(names)
    if not mdb or not enst:
        print(f"need both corpora: MDB {len(mdb)}, ENST {len(enst)}", file=sys.stderr)
        return 1

    sets = {"MDB": mdb, "ENST": enst}
    for label, data in sets.items():
        print(f"{label}: {len(data)} recordings, {sum(d[2] for d in data)} annotated "
              f"toms, {sum(len(d[0]) for d in data)} emitted")

    best = {}
    for label, data in sets.items():
        ks, sc = curve(data)
        best[label] = (float(ks[int(np.argmax(sc))]), float(sc.max()), float(sc[0]))

    print(f"\n{'':<8}{'no rule':>10}{'own best':>11}{'at own k':>10}"
          f"{'held out':>11}{'fitted elsewhere':>19}")
    print("-" * 70)
    for label, other in (("MDB", "ENST"), ("ENST", "MDB")):
        ks, sc = curve(sets[label])
        k_other = best[other][0]
        at_other = float(sc[int(np.argmin(np.abs(ks - k_other)))])
        print(f"{label:<8}{best[label][2]:>10.3f}{best[label][1]:>11.3f}"
              f"{best[label][0]:>10.2f}{at_other:>11.3f}{k_other:>19.2f}")

    print("\n'held out' is the score this corpus reaches at the threshold chosen on the")
    print("other one, and it is the only column that can carry a claim. 'own best' is")
    print("what fitting and testing on the same data produces, printed so the gap")
    print("between them is visible.")

    gains = []
    for label, other in (("MDB", "ENST"), ("ENST", "MDB")):
        ks, sc = curve(sets[label])
        at_other = float(sc[int(np.argmin(np.abs(ks - best[other][0])))])
        gains.append(at_other - best[label][2])
    print()
    if min(gains) > 0.02:
        print(f"Held out, the rule gains {gains[0]:+.3f} on MDB and {gains[1]:+.3f} on "
              f"ENST.\nIt survives transfer in both directions, which is the test the tom "
              f"ceiling\npolicy failed. Worth proposing as a default.")
    elif max(gains) <= 0:
        print("Held out, the rule gains nothing in either direction. The separation seen "
              "when\nfitting was the threshold memorising the corpus it was chosen on.")
    else:
        print(f"Held out it gains {gains[0]:+.3f} on MDB and {gains[1]:+.3f} on ENST -- "
              f"it transfers\none way and not the other, so it is a property of one "
              f"corpus and not a policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
