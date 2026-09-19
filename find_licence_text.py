"""Looks for licence text inside the ReStem installer, since none is published.

We benchmarked a commercial product and are preparing to publish the result. Their site
has no terms page (restemapp.com/terms is a 404) and the install directory carries no
EULA file, so the only place the terms can be is the installer or the application
itself. Before sending anything to the vendor it is worth knowing what we agreed to --
particularly whether driving the UI with a script, or publishing benchmark results, is
restricted.

Scans the installer for readable ASCII runs and reports any that look like licence
prose. Read-only: it opens the file and reads, nothing else.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Phrases that appear in licence agreements and rarely anywhere else.
INTEREST = re.compile(
    r"(?i)(end user licen[cs]e|licen[cs]e agreement|you may not|shall not|"
    r"reverse engineer|benchmark|decompile|disassemble|redistribut|"
    r"automat\w+ (?:the )?(?:software|interface)|publish.{0,30}result|"
    r"terms of (?:use|service)|all rights reserved|warrant)")

ASCII_RUN = re.compile(rb"[\x20-\x7e]{40,}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("installer", type=Path)
    ap.add_argument("--limit-mb", type=int, default=120,
                    help="how much of the file to scan; setup metadata sits near the "
                         "front, the payload after it")
    ap.add_argument("--context", type=int, default=0,
                    help="print this many characters around each hit")
    args = ap.parse_args()

    if not args.installer.exists():
        print(f"no such file: {args.installer}")
        return 1

    size = args.installer.stat().st_size
    scan = min(size, args.limit_mb * 1024 * 1024)
    print(f"{args.installer.name}: {size / 1024 / 1024:.0f} MB, scanning first "
          f"{scan / 1024 / 1024:.0f} MB")

    seen = set()
    hits = 0
    with args.installer.open("rb") as fh:
        data = fh.read(scan)

    for m in ASCII_RUN.finditer(data):
        text = m.group(0).decode("ascii", "replace")
        if not INTEREST.search(text):
            continue
        key = text[:60]
        if key in seen:
            continue
        seen.add(key)
        hits += 1
        print(f"\n  at {m.start():,}:")
        print("   " + (text if args.context == 0 else text[:args.context]))
        if hits >= 40:
            print("\n  (stopping after 40 distinct hits)")
            break

    if not hits:
        print("\nNo licence-like text in the scanned region. Either it sits further in,\n"
              "or it is compressed, which Inno Setup and similar installers do by default.\n"
              "In that case the terms are only visible when the installer runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
