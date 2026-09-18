"""Second overnight pass: settles two claims that are currently unverified.

Both are debts this project owes its own standard.

1. The CPU thread count. bench_threads.py was run while the machine was simultaneously
   building a 417 MB training set, and the README says so plainly -- "the absolute
   values are noisy". The conclusion (6 threads, not torch's default 16) drives a real
   code path, so it deserves a clean measurement on an idle machine.

2. Onset accuracy after the note change. The kick now writes as General MIDI 36 instead
   of 35, and OUTPUT_NOTES is applied inside write_midi. benchmark_mdb.py accepts both
   numbers, so nothing should have moved -- but "should" is not a measurement, and a
   silent regression in the headline 0.882 would be the worst possible bug to ship.

Waits for the first pass to finish so the machine is genuinely idle for step 1.

    python overnight2.py
"""

from __future__ import annotations

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

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)


def say(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def busy() -> bool:
    """Is any of our own long-running work still going?"""
    if sys.platform != "win32":
        return False
    probe = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "(Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
         "Where-Object { $_.CommandLine -match 'overnight\\.py|compare_extractors|"
         "drum2midi\\.py' }).Count"],
        capture_output=True, text=True)
    try:
        return int((probe.stdout or "0").strip() or 0) > 0
    except ValueError:
        return False


def run(title: str, args: list[str], timeout: float = 4 * 3600) -> None:
    say(f"START  {title}")
    t0 = time.time()
    try:
        res = subprocess.run([PY, *args], cwd=str(ROOT), capture_output=True,
                             text=True, encoding="utf-8", errors="replace",
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        say(f"TIMEOUT {title}")
        return
    out = (res.stdout or "") + (res.stderr or "" if res.returncode else "")
    for line in [l for l in out.splitlines() if l.strip()][-30:]:
        with LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"    | {line}\n")
    say(f"{'DONE  ' if res.returncode == 0 else 'FAILED'} {title}  "
        f"({(time.time()-t0)/60:.1f} min)")


def main() -> int:
    sys.path.insert(0, str(ROOT))
    from keep_awake import keep_awake

    with keep_awake():
        say("=== second pass queued, waiting for the machine to go idle")
        waited = 0
        while busy() and waited < 6 * 3600:
            time.sleep(120)
            waited += 120
        say(f"machine idle after {waited/3600:.1f} h")

        # now the machine is quiet, so the thread sweep measures threads and not
        # contention with a separator
        run("CPU thread count on an idle machine", ["bench_threads.py", "--repeats", "5"])

        # The headline number, recomputed from scratch. A tag is used so the existing
        # cache stays intact for the ADT_STR comparison, and because --rescore would
        # only re-read the old MIDI and miss exactly the regression being looked for.
        run("MDB onsets, full regression check after the note change",
            ["benchmark_mdb.py", "--tag", "note36"])

        run("synthetic end-to-end check", ["evaluate.py"])
        say("=== second pass complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
