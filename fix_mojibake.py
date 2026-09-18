"""Repairs text that was read as cp1252 and written back as UTF-8.

Editing a UTF-8 file with a tool that assumes the system code page turns every
non-ASCII character into two or three Latin-1 ones: an em dash becomes "a-EUR-", an
ellipsis becomes "a-EUR-|". The damage is reversible because the mangled text still
contains the original bytes -- encoding it back to cp1252 recovers them.

Only lines that round-trip cleanly are touched, so correct text is never altered.

    python fix_mojibake.py --check       # report, change nothing
    python fix_mojibake.py               # repair in place
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
# Latin-1 letters that appear when UTF-8 is misread. The first group covers Western
# punctuation (an em dash becomes "a-EUR-"), the second the Cyrillic range, whose
# leading bytes land on different characters again.
SUSPECT = "âÂ€¦œ†“”ÐÑÒÓ°±"


def repair(text: str) -> str:
    try:
        fixed = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text
    return fixed if fixed != text else text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report without writing")
    ap.add_argument("paths", nargs="*", help="files to scan (default: this project)")
    args = ap.parse_args()

    if args.paths:
        files = [Path(p) for p in args.paths]
    else:
        files = [p for pattern in ("*.py", "*.pyw", "*.md", "*.txt", "*.yml", "*.bat")
                 for p in ROOT.glob(pattern)]

    total = 0
    for path in sorted(set(files)):
        if path.name == Path(__file__).name or not path.is_file():
            continue
        try:
            src = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        out_lines = []
        hits = 0
        for line in src.splitlines(keepends=True):
            if any(c in line for c in SUSPECT):
                fixed = repair(line)
                if fixed != line:
                    hits += 1
                    out_lines.append(fixed)
                    continue
            out_lines.append(line)

        if hits:
            total += hits
            print(f"{path.name}: {hits} line(s)")
            if not args.check:
                path.write_text("".join(out_lines), encoding="utf-8", newline="")

    if not total:
        print("no mangled text found")
    elif args.check:
        print(f"\n{total} line(s) would be repaired; run without --check to fix")
    else:
        print(f"\nrepaired {total} line(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
