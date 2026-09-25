"""Would a warning about silent passages be right often enough to show?

ADTOF goes deaf on some passages, and neither a fallback, pre-emphasis nor a second
transcriber fixed that. What is left is telling the user. A warning that is wrong half
the time teaches people to ignore it, so this measures one before it ships.

Everything was fixed before a single passage was scored:

- the detector is `blind_spots.py` word for word. Onsets come from the audio's own onset
  envelope. An onset is blind when ADTOF stays under half its threshold for every class
  within 60 ms. Blind onsets less than 2 s apart form a passage, and a passage has to
  span more than 0.5 s. Nothing is tuned here;
- a passage is right when it contains at least one reference onset and the published
  export misses at least half of them. The reference onsets are the five classes merged
  within 20 ms, and an onset is missed when no note of any pitch lies within 50 ms;
- precision is right passages over all passages, with a 95% interval from a bootstrap
  over recordings;
- the warning ships for a condition when the lower bound is at least 0.50 over at least
  10 passages. Drum-only input is MDB's 23 drum tracks plus ENST's 210. Full-mix input is
  MDB's 23 mixes through htdemucs, the default extractor. If drum-only fails, nothing
  ships and the full mix is not run, since it could not change that. If only the full
  mix fails, the warning is limited to input that was not pulled out of a song.

    python silence_warning_gate.py              # drum-only from cache; full mix if it passes
    python silence_warning_gate.py --full-mix   # run the full mix regardless
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402

import drum2midi  # noqa: E402
from adt_str_mallets import refs as enst_refs  # noqa: E402
from benchmark_enst import DEFAULT_KINDS, DEFAULT_POPULATION, recordings  # noqa: E402
from benchmark_mdb import AUDIO, ORDER, read_annotation  # noqa: E402
from preemphasis_gate import boot_share  # noqa: E402
from second_transcriber_gate import adtof_view, kill_tree  # noqa: E402

MDB = ROOT / "mdbdrums" / "MDB Drums"
ANN = MDB / "annotations" / "class"
FULL = MDB / "audio" / "full_mix"
GAP, SPAN, MERGE, TOL, PAD = 2.0, 0.5, 0.02, 0.05, 0.06


def passages(times: np.ndarray, base: np.ndarray) -> list:
    """Blind passages exactly as blind_spots.py reports them."""
    bt = [float(t) for t, r in zip(times, base) if r < 0.5]
    if not bt:
        return []
    runs, start, prev = [], bt[0], bt[0]
    for t in bt[1:]:
        if t - prev > GAP:
            runs.append((start, prev))
            start = t
        prev = t
    runs.append((start, prev))
    return [(a, b) for a, b in runs if b - a > SPAN]


def union(ref: dict) -> np.ndarray:
    """All reference onsets, a kick and a hi-hat struck together counted once."""
    t = np.sort(np.concatenate([np.asarray(ref[k], dtype=float) for k in ORDER]))
    if not t.size:
        return t
    keep = np.concatenate([[True], np.diff(t) > MERGE])
    return t[keep]


def notes(mid: Path) -> np.ndarray:
    if not mid.exists():
        return np.array([])
    return np.sort([n.start for inst in pretty_midi.PrettyMIDI(str(mid)).instruments
                    for n in inst.notes])


def missed(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    if not est.size:
        return np.ones(ref.size, dtype=bool)
    i = np.clip(np.searchsorted(est, ref), 1, est.size) - 1
    j = np.clip(i + 1, 0, est.size - 1)
    return np.minimum(np.abs(est[i] - ref), np.abs(est[j] - ref)) > TOL


def judge(times, base, ref, est) -> dict:
    ps = passages(times, base)
    miss = missed(ref, est)
    right, inside = 0, np.zeros(ref.size, dtype=bool)
    detail = []
    for a, b in ps:
        sel = (ref >= a - PAD) & (ref <= b + PAD)
        n, m = int(sel.sum()), int((sel & miss).sum())
        ok = n > 0 and m >= 0.5 * n
        right += ok
        if ok:
            inside |= sel
        detail.append((a, b, n, m, ok))
    return {"passages": len(ps), "right": right, "missed": int(miss.sum()),
            "covered": int((inside & miss).sum()), "detail": detail}


def extract(src: Path, dest: Path, device: str, timeout: float) -> bool:
    """htdemucs drums stem, killed as a tree if it hangs; the log goes to a file."""
    if dest.exists():
        return True
    tmp = dest.parent / f"_{src.stem}"
    tmp.mkdir(parents=True, exist_ok=True)
    cmd = drum2midi._separator_command(src, drum2midi.HTDEMUCS_MODEL, tmp, device)
    with (dest.parent / f"{src.stem}.log").open("w", encoding="utf-8") as log:
        kw = {"stdout": log, "stderr": subprocess.STDOUT}
        if os.name != "nt":
            kw["start_new_session"] = True
        proc = subprocess.Popen(cmd, **kw)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            return False
    hits = [p for p in tmp.iterdir() if "drums" in p.name.lower()]
    if not hits:
        return False
    hits[0].replace(dest)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    return True


def report(label: str, rows: list, rounds: int, seed: int) -> bool:
    n = np.array([r["passages"] for r in rows])
    k = np.array([r["right"] for r in rows])
    share, (lo, hi) = boot_share(k, n, rounds, seed)
    total = int(n.sum())
    if total < 10:
        verdict = "INSUFFICIENT (< 10 passages): does not ship here"
    elif lo >= 0.5:
        verdict = "PASS: ships"
    else:
        verdict = "FAIL: does not ship"
    miss = sum(r["missed"] for r in rows)
    cov = sum(r["covered"] for r in rows)
    warned = int((n > 0).sum())
    false_per = (total - int(k.sum())) / max(len(rows), 1)
    print(f"\n{label}: {len(rows)} recordings, {warned} warned, {total} passages")
    print(f"  precision {int(k.sum())}/{total} = {share:.1%} [{lo:.1%}, {hi:.1%}]   {verdict}")
    print(f"  false passages per recording {false_per:.2f}; export-missed reference onsets "
          f"inside right passages {cov}/{miss} = {cov / max(miss, 1):.1%}")
    return verdict.startswith("PASS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / "bench" / "adtstr_gate" / "adtof")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "silence_warning")
    ap.add_argument("--full-mix", action="store_true",
                    help="extract and score the full mixes even if drum-only fails")
    ap.add_argument("--stop-at", default="07:30",
                    help="no new extraction starts after this local time")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--rounds", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    thr = np.array(drum2midi.DEFAULT_THRESHOLDS, dtype=float)
    enst_data = ROOT / "enst" / "enst_drums_public"
    enst = recordings(enst_data, DEFAULT_KINDS, "wet_mix")
    if len(enst) != DEFAULT_POPULATION["recordings"]:
        print(f"!! POPULATION MISMATCH: {len(enst)} ENST recordings, not "
              f"{DEFAULT_POPULATION['recordings']}")

    rows = {"mdb": [], "enst": [], "mallets": []}
    for wav in sorted(AUDIO.glob("*.wav")):
        times, base = adtof_view(wav, thr, args.cache / f"mdb_{wav.stem}.npz")
        song = wav.stem.replace("_Drum", "")
        ref = union(read_annotation(ANN / f"{song}_class.txt"))
        r = judge(times, base, ref, notes(ROOT / "bench" / "ceiling" / f"{wav.stem}.mid"))
        r["name"] = f"mdb/{song}"
        rows["mdb"].append(r)
    for d, ann, wav, _k in enst:
        name = f"{ann.stem}_d{d}"
        times, base = adtof_view(wav, thr, args.cache / f"enst_{name}.npz")
        r = judge(times, base, union(enst_refs(ann)),
                  notes(ROOT / "bench" / "enst" / f"{name}.mid"))
        r["name"] = f"enst/{name}"
        rows["mallets" if "_mallets" in ann.stem else "enst"].append(r)

    print("recordings with a passage: name, passages [start-end s, ref onsets, missed, right]")
    for group in rows.values():
        for r in group:
            if r["passages"]:
                spans = "; ".join(f"{a:.1f}-{b:.1f} {n}/{m}{'' if ok else ' x'}"
                                  for a, b, n, m, ok in r["detail"])
                print(f"  {r['name']:<50} {spans}")

    passed = report("A. drum-only (pre-registered gate)",
                    rows["mdb"] + rows["enst"] + rows["mallets"], args.rounds, args.seed)
    for label, group in (("  MDB drum-only", rows["mdb"]),
                         ("  ENST without mallets", rows["enst"]),
                         ("  ENST mallets", rows["mallets"])):
        report(label + " (context)", group, args.rounds, args.seed)

    if not passed and not args.full_mix:
        print("\nB. full mix: not run. Drum-only failed, so nothing ships whatever it shows.")
        return 0

    device = "xpu" if drum2midi.xpu_available() else "cpu"
    stems = args.out / "stems"
    stems.mkdir(parents=True, exist_ok=True)
    stop = dt.datetime.combine(dt.date.today(), dt.time.fromisoformat(args.stop_at))
    if dt.datetime.now() > stop and dt.datetime.now().hour >= 12:
        stop += dt.timedelta(days=1)
    full = []
    print(f"\nfull mix: htdemucs on {device}")
    for mix in sorted(FULL.glob("*.wav")):
        dest = stems / f"{mix.stem}_drums.wav"
        if not dest.exists() and dt.datetime.now() > stop:
            print(f"  stop time {args.stop_at} reached; {mix.stem} and later not extracted")
            break
        t0 = dt.datetime.now()
        if not extract(mix, dest, device, args.timeout):
            print(f"  ! {mix.stem}: extraction failed or timed out, scored as missing")
            continue
        times, base = adtof_view(dest, thr, args.out / "adtof" / f"{mix.stem}.npz")
        song = mix.stem.replace("_MIX", "")
        ref = union(read_annotation(ANN / f"{song}_class.txt"))
        r = judge(times, base, ref,
                  notes(ROOT / "bench" / "extractors" / "htdemucs" / f"{mix.stem}.mid"))
        r["name"] = f"mdb-mix/{song}"
        full.append(r)
        spans = "; ".join(f"{a:.1f}-{b:.1f} {n}/{m}{'' if ok else ' x'}"
                          for a, b, n, m, ok in r["detail"])
        print(f"  {song:<24} {(dt.datetime.now() - t0).seconds:>4}s  {spans}", flush=True)
    if len(full) < 23:
        print(f"!! only {len(full)} of 23 mixes scored")
    report("B. full mix through htdemucs (pre-registered gate)", full, args.rounds, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
