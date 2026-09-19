"""Sweep the tom threshold policy against both corpora at once.

Measured so far, and the reason this exists:

    MDB  (tom density 1.1%)   adaptive 0.605   fixed 0.322    adaptive wins by +0.282
    ENST (tom density 5.9%)   adaptive 0.342   fixed 0.590    fixed wins by +0.248

transcribe() sets the tom threshold to max(percentile(toms, TOM_PERCENTILE), TOM_FLOOR).
That is a proportional cap, so the denser the toms the higher the threshold it chooses --
self-defeating on exactly the material where toms matter. On MDB's Beatles, the
tom-richest track, it picks 0.572 against the fixed 0.32; on ENST solos it holds tom
recall to 0.227.

The obvious missing piece is a ceiling to match the floor, so the policy can still lift a
threshold on sparse material without being free to raise it without limit on dense
material. This sweeps that and the neighbouring parameters.

Doing it by re-running the pipeline would cost 35 minutes per candidate on ENST. The only
thing that varies is the peak-picking threshold, so activations are computed once per
recording, cached, and every candidate is then almost free.

That shortcut is only sound if it reproduces the pipeline, so the script calibrates
against the four numbers above before sweeping anything and refuses to continue if it
cannot. Run --calibrate alone to check that without sweeping.

    python sweep_tom_policy.py --calibrate
    python sweep_tom_policy.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--repo", default=os.environ.get("DRUM2MIDI_ROOT"))
ap.add_argument("--cache", default=None, help="where to keep activation .npy files")
ap.add_argument("--calibrate", action="store_true", help="check fidelity, sweep nothing")
ap.add_argument("--rounds", type=int, default=4000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tolerance", type=float, default=0.02)
args = ap.parse_args()
if not args.repo or not (Path(args.repo) / "drum2midi.py").exists():
    raise SystemExit("point --repo or $DRUM2MIDI_ROOT at a drum2midi checkout")

ROOT = Path(args.repo).resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import mir_eval  # noqa: E402
import numpy as np  # noqa: E402

import drum2midi  # noqa: E402
from benchmark_mdb import ANN, AUDIO, CLASSES, WINDOW, read_annotation  # noqa: E402
from benchmark_enst import IGNORED, LABEL_TO_CLASS  # noqa: E402

FPS = 100
TOM_COLUMN = 2          # LABELS_5 order: kick, snare, tom, hi-hat, cymbal
ENST_KINDS = {"phrase", "solo", "minus-one", "MIDI-minus-one"}
CACHE = Path(args.cache or (ROOT / "bench" / "act_cache"))


def activations(wav: Path) -> np.ndarray:
    """ADTOF activations for one recording, cached; this is the only slow step."""
    key = CACHE / f"{wav.parent.parent.parent.name}_{wav.stem}.npy"
    if key.exists():
        return np.load(key)
    import torch
    from adtof_pytorch import load_audio_for_model
    model = drum2midi._adtof_model("cpu")
    x = load_audio_for_model(str(wav)).to("cpu")
    with torch.no_grad():
        act = model(x).cpu().numpy()[0]
    if act.ndim == 2 and act.shape[0] < act.shape[1]:
        act = act.T
    key.parent.mkdir(parents=True, exist_ok=True)
    np.save(key, act.astype(np.float32))
    return act


def tom_times(act: np.ndarray, threshold: float) -> np.ndarray:
    """Peak-pick the tom column exactly as the pipeline's PeakPicker does."""
    from adtof_pytorch import PeakPicker
    thr = list(drum2midi.DEFAULT_THRESHOLDS)
    thr[TOM_COLUMN] = threshold
    picked = PeakPicker(thresholds=thr, fps=FPS).pick(
        act[None, ...], labels=__import__("adtof_pytorch").LABELS_5, label_offset=0)[0]
    out = []
    for pitch, times in picked.items():
        if pitch in CLASSES["TT"]:
            out.extend(times)
    return np.array(sorted(out))


def policy_threshold(act: np.ndarray, kind: str, **kw) -> float:
    col = act[:, TOM_COLUMN]
    if kind == "fixed":
        return kw["value"]
    value = float(np.percentile(col, kw.get("percentile", drum2midi.TOM_PERCENTILE)))
    value = max(value, kw.get("floor", drum2midi.TOM_FLOOR))
    ceiling = kw.get("ceiling")
    return min(value, ceiling) if ceiling is not None else value


def score(pairs, policy) -> tuple:
    """pairs is [(name, activations, reference tom times)]; returns per-recording counts."""
    rows = {}
    for name, act, ref in pairs:
        est = tom_times(act, policy_threshold(act, **policy))
        if ref.size and est.size:
            _f, prec, _r = mir_eval.onset.f_measure(ref, est, window=WINDOW)
            tp = int(round(prec * len(est)))
        else:
            tp = 0
        rows[name] = {"tp": tp, "ref": int(ref.size), "est": int(est.size)}
    return rows


def f1(rows, picks) -> float:
    tp = sum(rows[n]["tp"] for n in picks)
    ref = sum(rows[n]["ref"] for n in picks)
    est = sum(rows[n]["est"] for n in picks)
    if ref == 0 or est == 0:
        return float("nan")
    p, r = tp / est, tp / ref
    return 2 * p * r / max(p + r, 1e-9)


def interval(v, lo=0.025, hi=0.975):
    v = sorted(v)
    return v[int(lo * len(v))], v[int(hi * len(v))]


def load_mdb():
    pairs = []
    for wav in sorted(AUDIO.glob("*.wav")):
        ann = ANN / f"{wav.stem.replace('_Drum', '')}_class.txt"
        if not ann.exists():
            continue
        ref = np.array(sorted(read_annotation(ann).get("TT", [])))
        pairs.append((wav.stem, activations(wav), ref))
    return pairs


def load_enst():
    data = ROOT / "enst" / "enst_drums_public"
    pairs = []
    for d in (1, 2, 3):
        for ann in sorted((data / f"drummer_{d}" / "annotation").glob("*.txt")):
            parts = ann.stem.split("_")
            if len(parts) < 2 or parts[1] not in ENST_KINDS:
                continue
            wav = data / f"drummer_{d}" / "audio" / "wet_mix" / f"{ann.stem}.wav"
            if not wav.exists():
                continue
            ref = []
            for line in ann.read_text(encoding="utf-8", errors="replace").splitlines():
                p = line.split()
                if len(p) >= 2 and p[1] not in IGNORED \
                        and LABEL_TO_CLASS.get(p[1]) == "TT":
                    ref.append(float(p[0]))
            pairs.append((f"d{d}/{ann.stem}", activations(wav), np.array(sorted(ref))))
    return pairs


ADAPTIVE = {"kind": "adaptive"}
FIXED = {"kind": "fixed", "value": drum2midi.DEFAULT_THRESHOLDS[TOM_COLUMN]}

KNOWN = {("mdb", "adaptive"): 0.605, ("mdb", "fixed"): 0.322,
         ("enst", "adaptive"): 0.342, ("enst", "fixed"): 0.590}


def main() -> int:
    print(f"tom policy today: max(percentile(toms, {drum2midi.TOM_PERCENTILE}), "
          f"{drum2midi.TOM_FLOOR});  fixed comparison "
          f"{drum2midi.DEFAULT_THRESHOLDS[TOM_COLUMN]}")
    print(f"activation cache: {CACHE}\n")

    print("computing activations (cached after the first run) ...", flush=True)
    corpora = {"mdb": load_mdb(), "enst": load_enst()}
    for name, pairs in corpora.items():
        print(f"  {name}: {len(pairs)} recordings, "
              f"{sum(int(r.size) for _n, _a, r in pairs)} tom onsets")

    print(f"\ncalibration against the pipeline (tolerance {args.tolerance:.3f})")
    print(f"{'corpus':<8}{'policy':<10}{'here':>8}{'pipeline':>10}{'delta':>8}")
    print("-" * 44)
    ok = True
    for corpus, pairs in corpora.items():
        for label, policy in (("adaptive", ADAPTIVE), ("fixed", FIXED)):
            got = f1(score(pairs, policy), [n for n, _a, _r in pairs])
            want = KNOWN[(corpus, label)]
            delta = got - want
            flag = "" if abs(delta) <= args.tolerance else "   MISMATCH"
            ok &= abs(delta) <= args.tolerance
            print(f"{corpus:<8}{label:<10}{got:>8.3f}{want:>10.3f}{delta:>+8.3f}{flag}")
    if not ok:
        print("\nPeak picking here does not reproduce the pipeline. The sweep would be\n"
              "measuring this script rather than drum2midi; not continuing.")
        return 1
    print("\nreproduces the pipeline; the sweep is measuring drum2midi")
    if args.calibrate:
        return 0

    candidates = [("current (98.5, floor 0.25)", ADAPTIVE), ("fixed 0.32", FIXED)]
    for c in (0.28, 0.30, 0.32, 0.35, 0.40, 0.45):
        candidates.append((f"98.5, floor 0.25, ceiling {c:.2f}",
                           {"kind": "adaptive", "ceiling": c}))
    for p in (95.0, 96.5, 97.5, 99.0):
        candidates.append((f"percentile {p}, floor 0.25",
                           {"kind": "adaptive", "percentile": p}))
    for v in (0.25, 0.28, 0.36, 0.40):
        candidates.append((f"fixed {v:.2f}", {"kind": "fixed", "value": v}))

    rng = random.Random(args.seed)
    draws = {}
    for corpus, pairs in corpora.items():
        names = [n for n, _a, _r in pairs]
        draws[corpus] = [[names[rng.randrange(len(names))] for _ in names]
                         for _ in range(args.rounds)]

    print(f"\ntom F1, bootstrap over recordings, {args.rounds} resamples")
    print(f"{'policy':<32}{'MDB':>7}{'  95% CI':>17}{'ENST':>8}{'  95% CI':>17}")
    print("-" * 82)
    results = []
    for label, policy in candidates:
        line, vals = f"{label:<32}", {}
        for corpus, pairs in corpora.items():
            rows = score(pairs, policy)
            names = [n for n, _a, _r in pairs]
            point = f1(rows, names)
            ci = interval([v for v in (f1(rows, d) for d in draws[corpus]) if v == v])
            vals[corpus] = (point, ci)
            line += f"{point:>7.3f}   [{ci[0]:.3f},{ci[1]:.3f}]" if corpus == "mdb" \
                else f"{point:>8.3f}   [{ci[0]:.3f},{ci[1]:.3f}]"
        print(line, flush=True)
        results.append((label, vals))

    base_mdb = KNOWN[("mdb", "adaptive")]
    print(f"\nthe constraint: beat ENST's {KNOWN[('enst', 'adaptive')]:.3f} without "
          f"regressing MDB's {base_mdb:.3f}")
    print(f"{'policy':<32}{'MDB delta':>11}{'ENST delta':>12}{'both?':>8}")
    print("-" * 63)
    for label, vals in results:
        dm = vals["mdb"][0] - base_mdb
        de = vals["enst"][0] - KNOWN[("enst", "adaptive")]
        verdict = "yes" if dm >= -0.02 and de > 0 else ""
        print(f"{label:<32}{dm:>+11.3f}{de:>+12.3f}{verdict:>8}")

    print("\nA policy qualifies only if it does not cost MDB more than 0.02 -- that\n"
          "corpus is tom-sparse and represents the material most users bring.")

    # ---- intervals on the difference, and a held-out check ----------------------
    # Sixteen candidates were just scored on the same recordings they are selected on,
    # which is how a benchmark gets overfitted. Both sections below exist to say how
    # much of the winner's margin survives that.

    qualifying = [(label, vals) for label, vals in results
                  if vals["mdb"][0] - base_mdb >= -0.02
                  and vals["enst"][0] - KNOWN[("enst", "adaptive")] > 0]
    if not qualifying:
        print("\nno candidate satisfies the constraint; nothing further to test")
        return 0
    best_label = max(qualifying, key=lambda kv: kv[1]["enst"][0])[0]
    best = dict(candidates)[best_label]

    print(f"\n{'=' * 72}")
    print(f"paired bootstrap, '{best_label}' minus the current policy")
    print(f"{'corpus':<8}{'current':>9}{'candidate':>11}{'diff':>9}{'95% CI':>22}")
    print("-" * 60)
    for corpus, pairs in corpora.items():
        names = [n for n, _a, _r in pairs]
        cur, cand = score(pairs, ADAPTIVE), score(pairs, best)
        diffs = []
        for d in draws[corpus]:
            a, b = f1(cur, d), f1(cand, d)
            if a == a and b == b:
                diffs.append(b - a)
        lo, hi = interval(diffs)
        verdict = "significant" if lo > 0 or hi < 0 else "not significant"
        print(f"{corpus:<8}{f1(cur, names):>9.3f}{f1(cand, names):>11.3f}"
              f"{f1(cand, names) - f1(cur, names):>+9.3f}"
              f"   [{lo:+.3f}, {hi:+.3f}]  {verdict}")

    enst = corpora["enst"]
    fit = [(n, a, r) for n, a, r in enst if not n.startswith("d3/")]
    held = [(n, a, r) for n, a, r in enst if n.startswith("d3/")]
    print(f"\nheld out: choose the ceiling on drummers 1-2 ({len(fit)} recordings), "
          f"report on drummer 3 ({len(held)})")
    ceilings = [0.30, 0.32, 0.35, 0.40, 0.45, 0.50, 0.55]
    best_c, best_f = None, -1.0
    for c in ceilings:
        pol = {"kind": "adaptive", "ceiling": c}
        mdb_rows = score(corpora["mdb"], pol)
        if f1(mdb_rows, [n for n, _a, _r in corpora["mdb"]]) - base_mdb < -0.02:
            continue
        v = f1(score(fit, pol), [n for n, _a, _r in fit])
        print(f"  ceiling {c:.2f}   drummers 1-2 {v:.3f}"
              + ("   <- chosen" if v > best_f else ""))
        if v > best_f:
            best_c, best_f = c, v
    if best_c is None:
        print("  no ceiling satisfies the MDB constraint on the fitting split")
        return 0

    pol = {"kind": "adaptive", "ceiling": best_c}
    cur_h, cand_h = score(held, ADAPTIVE), score(held, pol)
    hn = [n for n, _a, _r in held]
    rng2 = random.Random(args.seed + 1)
    hd = [[hn[rng2.randrange(len(hn))] for _ in hn] for _ in range(args.rounds)]
    diffs = []
    for d in hd:
        a, b = f1(cur_h, d), f1(cand_h, d)
        if a == a and b == b:
            diffs.append(b - a)
    lo, hi = interval(diffs)
    print(f"\n  ceiling {best_c:.2f} chosen without seeing drummer 3, then applied to it:")
    print(f"    current   {f1(cur_h, hn):.3f}")
    print(f"    candidate {f1(cand_h, hn):.3f}")
    print(f"    diff      {f1(cand_h, hn) - f1(cur_h, hn):+.3f}   "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]   "
          f"{'significant' if lo > 0 or hi < 0 else 'not significant'}")
    print("\nA margin that survives being chosen on other drummers is a property of the\n"
          "policy. One that does not is a property of this sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
