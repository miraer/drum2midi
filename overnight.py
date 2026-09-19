"""Runs the remaining measurements unattended, in order, and writes one report.

Left to run overnight. Each step is independent: a failure is recorded and the next one
starts anyway, so a single broken task cannot waste the whole night. The machine is kept
awake for the duration by this process itself, rather than by a guard watching some
shell that may have exited.

Steps, cheapest first so partial results still arrive early:

  1. does the drum extractor benefit from the GPU, and is the output identical
  2. the live progress bar, on a real conversion
  3. re-transcribe the user's own track with the current defaults
  4. wait for the extractor comparison, then score it

    python overnight.py
    python overnight.py --skip 4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "bench" / "overnight.log"
PY = sys.executable


def say(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(title: str, args: list[str], timeout: float = 6 * 3600) -> bool:
    say(f"START  {title}")
    t0 = time.time()
    try:
        res = subprocess.run([PY, *args], cwd=str(ROOT), capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        say(f"TIMEOUT {title} after {timeout/3600:.1f} h")
        return False

    out = (res.stdout or "") + (res.returncode and (res.stderr or "") or "")
    tail = [l for l in out.splitlines() if l.strip()][-25:]
    for line in tail:
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"    | {line}\n")
    ok = res.returncode == 0
    say(f"{'DONE  ' if ok else 'FAILED'} {title}  ({(time.time()-t0)/60:.1f} min)")
    return ok


def extractor_comparison_running() -> bool:
    if sys.platform != "win32":
        return False
    probe = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
         "Where-Object { $_.CommandLine -match 'compare_extractors' }).Count"],
        capture_output=True, text=True)
    try:
        return int((probe.stdout or "0").strip() or 0) > 0
    except ValueError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", type=int, nargs="*", default=[],
                    help="step numbers to skip")
    ap.add_argument("--song", type=Path, default=os.environ.get("DRUM2MIDI_SONG"),
                    help="a full song to re-transcribe (default: $DRUM2MIDI_SONG)")
    ap.add_argument("--stem-dir", type=Path,
                    default=os.environ.get("DRUM2MIDI_STEM_DIR"),
                    help="folder holding a '*(Drums).wav' stem "
                         "(default: $DRUM2MIDI_STEM_DIR)")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from keep_awake import keep_awake

    # Kept out of the source: this is somebody's own music, and the path names it.
    song = Path(args.song) if args.song else None
    stem_dir = Path(args.stem_dir) if args.stem_dir else None
    stem = (next(stem_dir.glob("*(Drums).wav"), None)
            if stem_dir and stem_dir.exists() else None)

    steps = [
        (1, "extractor on GPU vs CPU", ["check_extractor_gpu.py"]),
        (2, "progress bar on a real conversion",
         ["drum2midi.py", "input/demo.wav", "-o", "bench/overnight_demo.mid"]),
        (3, "re-transcribe the user's track with current defaults",
         ["drum2midi.py", str(stem), "-o", "bench/overnight_track.mid"]
         if stem else None),
        # Does the transcriber's blind spot predict where it scores badly? If the
        # correlation is near zero the fallback idea dies and that is the result.
        (5, "blind spots across MDB, against per-track F1", ["blind_spots_mdb.py"]),
    ]

    with keep_awake() as granted:
        say(f"=== overnight run started, sleep {'blocked' if granted else 'NOT blocked'}")

        if extractor_comparison_running():
            say("extractor comparison is still running; the cheap steps go first")

        for number, title, argv in steps:
            if number in args.skip or argv is None:
                say(f"SKIP   {title}")
                continue
            run(title, argv)

        if 4 not in args.skip:
            say("waiting for the extractor comparison to finish")
            waited = 0
            while extractor_comparison_running() and waited < 5 * 3600:
                time.sleep(120)
                waited += 120
            if extractor_comparison_running():
                # The loop also exits on timeout, and the previous version treated
                # that as success: it announced "finished", started a second
                # compare_extractors over the first, and the two deadlocked on the
                # GPU for nine hours. Never launch anything while one is alive.
                say(f"extractor comparison STILL RUNNING after {waited/3600:.1f} h; "
                    "leaving it alone")
            else:
                say(f"extractor comparison finished (waited {waited/3600:.1f} h)")
                # scores the MIDI already on disk instead of transcribing again
                run("score the extractor comparison", ["score_extractors.py"])

        run("full test suite", ["test_smoke.py"])
        say("=== overnight run complete")

    print(f"\nreport: {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
