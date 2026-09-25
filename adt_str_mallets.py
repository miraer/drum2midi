"""Are ADT_STR's notes on mallet recordings better than ours, or only present?

`second_transcriber_gate.py` found that ADT_STR writes a note within 60 ms of 82% of the
mallet onsets ADTOF is blind to, against 36% by chance. "A note" is not "the right note",
so this scores both transcribers properly on those eight recordings.

Fixed before scoring:

- the recordings are the eight ENST mallet recordings. They are the same recordings that
  produced the hypothesis, so this test can close it but cannot establish it;
- MICRO F1 over five classes, 50 ms, mir_eval. Ours comes from the published export
  `bench/enst`, where a missing file counts as an empty estimate. ADT_STR's 26 classes are
  folded with `score_adt_str.WIDE`, as in the published 0.673; folding can only help it;
- a paired bootstrap over recordings, with the difference taken as ADT_STR minus ours;
- if the lower bound is above zero, the exception stays open as a hypothesis that needs
  mallet recordings from outside ENST, and nothing ships. Otherwise it closes as heard
  but not transcribed.

The other ENST recordings that have ADT_STR output are printed as context, not gated.

    python adt_str_mallets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import mir_eval  # noqa: E402
import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402

from benchmark_enst import DEFAULT_KINDS, IGNORED, LABEL_TO_CLASS, recordings  # noqa: E402
from benchmark_mdb import CLASSES, ORDER, WINDOW  # noqa: E402


def refs(ann: Path) -> dict:
    ref = {k: [] for k in ORDER}
    for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
        p = line.split()
        if len(p) < 2 or p[1] in IGNORED:
            continue
        cls = LABEL_TO_CLASS.get(p[1])
        if cls is not None:
            ref[cls].append(float(p[0]))
    return {k: np.array(sorted(v)) for k, v in ref.items()}


def counts(ref: dict, mid: Path | None, classes: dict) -> np.ndarray:
    """(tp, ref, est) summed over the five classes, and per class, for one recording."""
    notes = ([n for inst in pretty_midi.PrettyMIDI(str(mid)).instruments for n in inst.notes]
             if mid is not None and mid.exists() else [])
    out = np.zeros((len(ORDER), 3), dtype=int)
    for i, k in enumerate(ORDER):
        r = ref[k]
        e = np.array(sorted(n.start for n in notes if n.pitch in classes[k]))
        tp = 0
        if r.size and e.size:
            _f, p, _r = mir_eval.onset.f_measure(r, e, window=WINDOW)
            tp = int(round(p * e.size))
        out[i] = (tp, r.size, e.size)
    return out


def micro(c: np.ndarray) -> float:
    tp, ref, est = c[..., 0].sum(-1), c[..., 1].sum(-1), c[..., 2].sum(-1)
    p = np.divide(tp, est, out=np.zeros_like(tp, dtype=float), where=est > 0)
    r = np.divide(tp, ref, out=np.zeros_like(tp, dtype=float), where=ref > 0)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)


def compare(label: str, ours: np.ndarray, theirs: np.ndarray, rounds: int, seed: int,
            gated: bool) -> None:
    """Pooled MICRO for both, and a paired interval over recordings."""
    n = ours.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(rounds, n))
    o = micro(ours[idx].sum(axis=1).reshape(rounds, -1, 3))
    t = micro(theirs[idx].sum(axis=1).reshape(rounds, -1, 3))
    lo, hi = np.percentile(t - o, [2.5, 97.5])
    mo, mt = float(micro(ours.sum(0))), float(micro(theirs.sum(0)))
    verdict = ""
    if gated:
        verdict = ("   lower bound > 0: open, needs mallets from outside ENST" if lo > 0
                   else "   CLOSED: heard, not transcribed")
    print(f"{label}: {n} recordings\n"
          f"  MICRO  ours {mo:.3f}   ADT_STR {mt:.3f}   ADT_STR - ours {mt - mo:+.3f} "
          f"[{lo:+.3f}, {hi:+.3f}]{verdict}")
    for i, k in enumerate(ORDER):
        oc, tc = ours[:, i].sum(0), theirs[:, i].sum(0)
        if not oc[1]:
            continue
        of, tf = float(micro(oc[None])), float(micro(tc[None]))
        print(f"    {k:<3} ref {oc[1]:>5}   ours est {oc[2]:>5} F1 {of:.3f}   "
              f"ADT_STR est {tc[2]:>5} F1 {tf:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, default=ROOT / "bench" / "enst")
    ap.add_argument("--theirs", type=Path, default=ROOT / "bench" / "adtstr_gate" / "midi")
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # imported after parsing: score_adt_str answers --help itself at import time
    from score_adt_str import WIDE

    data = ROOT / "enst" / "enst_drums_public"
    groups = {"mallets": ([], []), "other": ([], [])}
    for d, ann, _wav, _kind in recordings(data, DEFAULT_KINDS, "wet_mix"):
        name = f"{ann.stem}_d{d}"
        theirs = sorted((args.theirs / f"enst_{name}").glob("*.mid"))
        if not theirs:
            continue
        ref = refs(ann)
        g = "mallets" if "_mallets" in ann.stem else "other"
        groups[g][0].append(counts(ref, args.ours / f"{name}.mid", CLASSES))
        groups[g][1].append(counts(ref, theirs[0], WIDE))
    print(f"ours from {args.ours}, ADT_STR from {args.theirs}\n")
    for g, gated in (("mallets", True), ("other", False)):
        ours, theirs = groups[g]
        if ours:
            compare(f"ENST {g}" + ("" if gated else " (context, not gated)"),
                    np.array(ours), np.array(theirs), args.rounds, args.seed, gated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
