"""Does a transcriber trained on different material hear what ADTOF is blind to?

ADTOF is blind in some passages: every class stays under half its threshold where the
audio has clear onsets. README and ROADMAP call this a property of the material rather
than of the transcriber. That claim rests on one recording, where ReStem failed on the
same passage. This counts it across every benchmark recording that has a blind onset,
using ADT_STR, which was trained on synthetic audio rendered from Lakh MIDI and so shares
none of ADTOF's training data.

Fixed before it ran:

- the population is every MDB (23) and default-ENST (210) recording with at least one
  ADTOF-blind onset, using the `blind_spots` definition on the original audio;
- an onset is heard when ADT_STR writes any note within 60 ms of it;
- the gate passes when at least half of the ADTOF-blind onsets are heard, pooled;
- the heard rate on onsets ADTOF does see, in the same recordings, is ADT_STR's own
  baseline. If seen-heard minus blind-heard has a lower bound above zero (95%, over
  recordings), the deafness travels with the material. If the interval contains zero, the
  claim is not supported;
- a recording that fails or runs past the timeout counts as unheard and is listed.

Passing ships nothing, because the fallback ceiling of 1.27% recall on MDB still applies.
Two checks were added after the run, timeouts scored from disk and a chance baseline, and
they print under their own heading (`secondary`), apart from the verdict.

    python second_transcriber_gate.py --limit 2
    python second_transcriber_gate.py
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
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
from preemphasis_gate import boot_share, ratios  # noqa: E402

ADT = ROOT / "ADT_STR"
WINDOW = 0.06
PASS_SHARE = 0.5


def adtof_view(wav: Path, thr: np.ndarray, cache: Path):
    """Onset times from the audio, and ADTOF's ratio to threshold at each."""
    if cache.exists():
        z = np.load(cache)
        return z["times"], z["base"]
    times, _s, _d = blind_spots.onset_peaks(wav)
    base = ratios(blind_spots.activations(wav), times, thr)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, times=times, base=base)
    return times, base


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop a child and everything it started.

    On Windows a venv's python.exe is a launcher that runs the real interpreter as its own
    child. Killing only the launcher leaves that child running. With pipes attached, the
    parent then waits for it, which is how a 20-minute timeout became 525 minutes on
    24 September.
    """
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True)
    else:
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def run_adt_str(wav: Path, dest: Path, config: Path, timeout: int):
    """(MIDI path or None, whether it hit the timeout); reuses work already on disk.

    A timed-out recording is scored as unheard, as fixed in advance. Its MIDI can still
    appear later if the old, unkillable timeout let the render finish, and it is returned
    so that the secondary analysis can report it separately.
    """
    timed_out = (dest / "TIMEOUT").exists()
    done = sorted(dest.glob("*.mid"))
    if done or timed_out:
        return (str(done[0]) if done else None), timed_out
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "adt_str.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "import torchaudio_shim; "
             "import sys; sys.argv[0] = 'inference.py'; "
             "exec(open('inference.py', encoding='utf-8').read())",
             "--input", str(wav), "--config", str(config), "-o", str(dest)],
            cwd=str(ADT), stdout=log, stderr=subprocess.STDOUT,
            start_new_session=(os.name != "nt"))
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            (dest / "TIMEOUT").write_text("timed out\n", encoding="utf-8")
            timed_out = True
    done = sorted(dest.glob("*.mid"))
    return (str(done[0]) if done else None), timed_out


def heard(times: np.ndarray, mid: str | None) -> np.ndarray:
    if mid is None:
        return np.zeros(len(times), dtype=bool)
    import pretty_midi
    starts = np.array(sorted(n.start for inst in pretty_midi.PrettyMIDI(mid).instruments
                             for n in inst.notes))
    if not starts.size:
        return np.zeros(len(times), dtype=bool)
    idx = np.clip(np.searchsorted(starts, times), 1, starts.size - 1)
    gap = np.minimum(np.abs(starts[idx] - times), np.abs(starts[idx - 1] - times))
    if starts.size == 1:
        gap = np.abs(starts[0] - times)
    return gap <= WINDOW


def coverage(mid: str | None, duration: float) -> float:
    """Share of the recording within WINDOW of some ADT_STR note: chance of 'heard'."""
    if mid is None or duration <= 0:
        return 0.0
    return float(heard(np.arange(0.0, duration, 0.005), mid).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--tau", default="0.8", choices=["0.4", "0.6", "0.8"])
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per recording")
    ap.add_argument("--out", type=Path, default=ROOT / "bench" / "adtstr_gate")
    ap.add_argument("--rounds", type=int, default=4000)
    args = ap.parse_args()

    checkpoint = ADT / "checkpoints" / f"setting-tau-{args.tau}"
    if not (checkpoint / "model.safetensors").exists():
        print(f"weights missing: run  python fetch_adt_str.py --tau {args.tau}")
        return 1
    from run_adt_str import write_config
    config = write_config(checkpoint)

    thr = np.array(drum2midi.DEFAULT_THRESHOLDS, dtype=float)
    enst = recordings(ROOT / "enst" / "enst_drums_public", DEFAULT_KINDS, "wet_mix")
    if len(enst) != DEFAULT_POPULATION["recordings"]:
        print(f"!! POPULATION MISMATCH: {len(enst)} ENST recordings, not "
              f"{DEFAULT_POPULATION['recordings']}")
    items = [("mdb", w.stem, w) for w in sorted(AUDIO.glob("*.wav"))]
    items += [("enst", f"{a.stem}_d{d}", w) for d, a, w, _k in enst]

    chosen = []
    for corpus, name, wav in items:
        times, base = adtof_view(wav, thr, args.out / "adtof" / f"{corpus}_{name}.npz")
        if (base < 0.5).any():
            chosen.append((corpus, name, wav, times, base))
    print(f"{len(chosen)} of {len(items)} recordings have an ADTOF-blind onset "
          f"({sum(int((c[4] < 0.5).sum()) for c in chosen)} blind onsets)")
    if args.limit:
        chosen = chosen[:args.limit]
        print(f"--limit: running {len(chosen)}")

    print(f"\n{'recording':<48}{'blind':>6}{'heard':>6}{'seen':>6}{'heard':>6}{'min':>6}")
    rows, extra, failed, t0 = [], [], [], time.time()
    for corpus, name, wav, times, base in chosen:
        t1 = time.time()
        mid, timed_out = run_adt_str(wav, args.out / "midi" / f"{corpus}_{name}", config,
                                     args.timeout)
        scored = None if timed_out else mid
        if scored is None:
            failed.append(f"{corpus} {name}")
        h = heard(times, scored)
        blind, seen = base < 0.5, base >= 1.0
        row = (corpus, int(blind.sum()), int((blind & h).sum()),
               int(seen.sum()), int((seen & h).sum()))
        rows.append(row)
        hd = heard(times, mid)
        extra.append({"name": name, "timed_out": timed_out,
                      "blind_heard_disk": int((blind & hd).sum()),
                      "seen_heard_disk": int((seen & hd).sum()),
                      "chance": coverage(mid, sf.info(str(wav)).duration)})
        print(f"{corpus + ' ' + name:<48}{row[1]:>6}{row[2]:>6}{row[3]:>6}{row[4]:>6}"
              f"{(time.time() - t1) / 60:>6.1f}" + ("  FAILED" if scored is None else ""),
              flush=True)
    print(f"\n{(time.time() - t0) / 60:.0f} min; {len(failed)} failed or timed out, "
          f"scored as unheard" + (": " + ", ".join(failed) if failed else ""))

    rng = np.random.default_rng(0)
    for corpus in ("mdb", "enst", "both"):
        sel = [r for r in rows if corpus == "both" or r[0] == corpus]
        if not sel:
            continue
        b, bh, s, sh = (np.array([r[i] for r in sel], dtype=int) for i in (1, 2, 3, 4))
        bs, bci = boot_share(bh, b, args.rounds)
        ss, sci = boot_share(sh, s, args.rounds)
        idx = rng.integers(0, len(sel), size=(args.rounds, len(sel)))
        with np.errstate(invalid="ignore", divide="ignore"):
            diff = sh[idx].sum(1) / s[idx].sum(1) - bh[idx].sum(1) / b[idx].sum(1)
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        verdict = ""
        if corpus == "both":
            verdict = ("   gate PASS" if bs >= PASS_SHARE else "   gate FAIL") + (
                "; deafness travels with the material" if lo > 0
                else "; material claim NOT supported")
        print(f"{corpus}: {len(sel)} recordings\n"
              f"  ADTOF-blind onsets heard by ADT_STR {bh.sum():>5} of {b.sum():<5} "
              f"= {bs:6.1%} [{bci[0]:.1%}, {bci[1]:.1%}]\n"
              f"  ADTOF-seen onsets heard by ADT_STR  {sh.sum():>5} of {s.sum():<5} "
              f"= {ss:6.1%} [{sci[0]:.1%}, {sci[1]:.1%}]\n"
              f"  seen minus blind {ss - bs:+.1%} [{lo:+.1%}, {hi:+.1%}]{verdict}")

    secondary(rows, extra, args.rounds)
    return 0


def secondary(rows: list, extra: list, rounds: int) -> None:
    """Checks added after the run, reported apart from the verdict they cannot change.

    Two things came up once the result was in. First, the timeout could not stop a render,
    so the three timed-out recordings eventually wrote MIDI. They are scored here, next to
    the pre-registered rule that counts them as unheard. Second, "a note within 60 ms"
    also happens by chance, and more often where ADT_STR writes densely. The chance rate
    is the share of a recording's duration within 60 ms of some ADT_STR note, weighted by
    where the onsets are. It is the heard rate of an instant chosen at random.
    """
    print("\n-- added after the run, not part of the pre-registered verdict --")
    groups = [
        ("all, timeouts scored from disk", lambda r, e: True),
        ("excluding the timeouts", lambda r, e: not e["timed_out"]),
        ("ENST mallets", lambda r, e: r[0] == "enst" and "_mallets" in e["name"]),
        ("everything else", lambda r, e: not (r[0] == "enst" and "_mallets" in e["name"])),
    ]
    for label, keep in groups:
        sel = [(r, e) for r, e in zip(rows, extra) if keep(r, e)]
        if not sel:
            continue
        a = np.array([[r[1], e["blind_heard_disk"], r[3], e["seen_heard_disk"],
                       e["chance"] * r[1], e["chance"] * r[3]] for r, e in sel], dtype=float)
        rng = np.random.default_rng(0)
        s = a[rng.integers(0, len(sel), size=(rounds, len(sel)))].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            blind_ci = np.nanpercentile(s[:, 1] / s[:, 0], [2.5, 97.5])
            above = np.nanpercentile((s[:, 1] - s[:, 4]) / s[:, 0], [2.5, 97.5])
        t = a.sum(axis=0)
        print(f"{label}: {len(sel)} recordings\n"
              f"  ADTOF-blind heard {int(t[1]):>5} of {int(t[0]):<5} = {t[1] / t[0]:6.1%} "
              f"[{blind_ci[0]:.1%}, {blind_ci[1]:.1%}]   chance {t[4] / t[0]:6.1%}   "
              f"above chance [{above[0]:+.1%}, {above[1]:+.1%}]\n"
              f"  ADTOF-seen heard  {int(t[3]):>5} of {int(t[2]):<5} = {t[3] / max(t[2], 1):6.1%}"
              f"   chance {t[5] / max(t[2], 1):6.1%}")


if __name__ == "__main__":
    raise SystemExit(main())
