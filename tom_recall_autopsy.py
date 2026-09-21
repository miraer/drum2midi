"""Why does every missed tom get missed? One reason per onset, no residue.

Our ENST tom recall is 0.463 against ReStem's 0.882, and that difference is the only
thing either machine has established about this class with an interval that clears zero.
Knowing it is recall says nothing about what to do; 753 onsets go missing and the useful
question is which mechanism drops each one.

Every missed onset falls into exactly one of these, checked in this order:

  the model is silent      the tom column never rises at all near the onset. Nothing
                           downstream can recover it.
  another class won        a pick exists at that instant on a different pitch. The
                           detector fired and the label is wrong.
  below the threshold      the picker thresholds `act - moving_average`, and that value
                           does not reach the tom threshold even though the raw
                           activation may be well above it.
  not a local maximum      it clears the threshold but a neighbouring frame within the
                           20 ms max window is higher, so it is not a peak.
  merged into a neighbour  it is a peak and clears the threshold, but sits within the
                           20 ms combine window of a stronger pick and is absorbed.

The order matters and is not arbitrary: a cause earlier in the list makes the later ones
unreachable, so attributing to the first that applies is the only assignment that does
not double-count. The counts below are therefore a partition, and they sum to the misses.

    python tom_recall_autopsy.py
    python tom_recall_autopsy.py --limit 20
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from benchmark_enst import IGNORED, LABEL_TO_CLASS  # noqa: E402

ENST = ROOT / "enst" / "enst_drums_public"
FPS = 100
W = 0.05
TOM_COL = 2
TOM_PITCHES = (41, 43, 45, 47, 48, 50)
NEAR = 3          # frames either side of the annotation to look for the response


def annotated(name: str) -> np.ndarray | None:
    for d in (1, 2, 3):
        p = ENST / f"drummer_{d}" / "annotation" / f"{name}.txt"
        if p.exists():
            return np.array(sorted(
                float(l.split()[0]) for l in p.read_text(encoding="utf-8",
                                                         errors="replace").splitlines()
                if len(l.split()) >= 2 and l.split()[1] not in IGNORED
                and LABEL_TO_CLASS.get(l.split()[1]) == "TT"))
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import drum2midi
    import pretty_midi
    from adtof_pytorch import load_audio_for_model
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    import torch

    model = drum2midi._adtof_model(args.device)
    probe = NotePeakPickingProcessor(threshold=0.0, fps=FPS)
    pre_avg, post_avg = int(round(probe.pre_avg * FPS)), int(round(probe.post_avg * FPS))
    pre_max, post_max = int(round(probe.pre_max * FPS)), int(round(probe.post_max * FPS))
    combine = max(1, int(round(probe.combine * FPS)))

    names = [n.strip() for n in
             (ROOT / "bench" / "enst60.txt").read_text().splitlines() if n.strip()]
    if args.limit:
        names = names[: args.limit]

    reasons: Counter = Counter()
    matched = 0
    total = 0
    for name in names:
        ref = annotated(name)
        mids = list((ROOT / "bench" / "enst60_sep").glob(f"{name}*.mid"))
        wav = ROOT / "restem_in" / f"{name}.wav"
        if ref is None or not len(ref) or not mids or not wav.exists():
            continue

        pm = pretty_midi.PrettyMIDI(str(mids[0]))
        notes = sorted((n.start, n.pitch) for i in pm.instruments for n in i.notes)
        times = np.array([t for t, _ in notes])
        pitches = np.array([p for _, p in notes])
        toms = times[np.isin(pitches, TOM_PITCHES)]

        x = load_audio_for_model(str(wav)).to(args.device)
        with torch.no_grad():
            act = model(x).cpu().numpy()[0]
        col = act[:, TOM_COL]
        proc = np.maximum(0.0, col - probe._moving_average(col, pre_avg, post_avg))
        wmax = probe._local_maxima(proc, pre_max, post_max)
        thr = min(max(float(np.percentile(col, drum2midi.TOM_PERCENTILE)),
                      drum2midi.TOM_FLOOR), drum2midi.TOM_CEILING)
        picks = np.where((proc >= wmax) & (proc >= thr))[0]

        used = np.zeros(len(toms), dtype=bool)
        for onset in ref:
            total += 1
            if len(toms):
                cand = np.where((~used) & (np.abs(toms - onset) <= W))[0]
                if len(cand):
                    used[cand[np.argmin(np.abs(toms[cand] - onset))]] = True
                    matched += 1
                    continue

            fr = int(np.clip(round(onset * FPS), 0, len(col) - 1))
            lo, hi = max(0, fr - NEAR), min(len(col), fr + NEAR + 1)
            if float(col[lo:hi].max()) < 0.05:
                reasons["the model is silent"] += 1
                continue
            if len(times) and float(np.abs(times - onset).min()) <= W:
                reasons["another class won"] += 1
                continue
            if float(proc[lo:hi].max()) < thr:
                reasons["below the threshold after mean subtraction"] += 1
                continue
            band = np.arange(lo, hi)
            peaks = band[(proc[lo:hi] >= wmax[lo:hi]) & (proc[lo:hi] >= thr)]
            if not len(peaks):
                reasons["clears the threshold but is not a local maximum"] += 1
                continue
            if len(picks) and int(np.abs(picks - peaks[0]).min()) <= combine:
                reasons["merged into a neighbouring pick"] += 1
            else:
                reasons["a pick the export did not carry"] += 1

        # onsets our MIDI matched are not examined further

    missed = total - matched
    print(f"{total} reference toms over {len(names)} recordings: "
          f"{matched} matched ({matched / max(total, 1):.1%}), {missed} missed\n")
    print(f"{'why each missed tom was missed':<48}{'count':>8}{'of misses':>12}")
    print("-" * 68)
    for reason, n in reasons.most_common():
        print(f"{reason:<48}{n:>8}{n / max(missed, 1):>11.1%}")
    print("-" * 68)
    print(f"{'total':<48}{sum(reasons.values()):>8}")

    silent = reasons["the model is silent"]
    picker = (reasons["below the threshold after mean subtraction"]
              + reasons["clears the threshold but is not a local maximum"]
              + reasons["merged into a neighbouring pick"])
    other = reasons["another class won"]
    print()
    print(f"model never responds:      {silent:>5}  ({silent / max(missed, 1):.0%}) "
          f"-- unreachable without a different model")
    print(f"wrong class chosen:        {other:>5}  ({other / max(missed, 1):.0%}) "
          f"-- the confusion documented separately")
    print(f"discarded by the picker:   {picker:>5}  ({picker / max(missed, 1):.0%}) "
          f"-- the model responded and the decision stage dropped it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
