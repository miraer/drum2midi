"""Reads a PDF and looks for the things we could not find anywhere else.

We published a comparison against a commercial product, having stated that its terms
are not published -- restemapp.com/terms is a 404, the install directory carries no
EULA, the installer body is compressed and the application exposes no About view. A
user's guide is the one place left, and if it carries terms, or answers the open
question about which MIDI path carries hi-hat articulation, then something we wrote
needs correcting.

    python read_pdf.py guide.pdf --outline
    python read_pdf.py guide.pdf --find "hi-hat"
    python read_pdf.py guide.pdf --page 12
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TOPICS = {
    "licence / terms": r"(?i)\b(licen[cs]e|end.user|agreement|terms of|eula|"
                       r"you may not|shall not|reverse.engineer|redistribut|"
                       r"benchmark|all rights reserved|warrant|copyright)\b",
    "hi-hat articulation": r"(?i)(hi.?hat|hihat|open|closed|pedal|cc\s*4|articulat)",
    "MIDI export vs live": r"(?i)(export|offline|real.?time|loopback|virtual (midi|port)|"
                           r"midi (out|output|port|file)|\.mid\b|save.{0,15}midi)",
    "quality modes": r"(?i)(better|best|bleed reduction|quality|good \(|offline\))",
    "unclassified / note 60": r"(?i)(unclassified|other|note\s*60|middle c|c3\b)",
}


def pages(path: Path) -> list[str]:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return [(p.extract_text() or "") for p in reader.pages]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--outline", action="store_true", help="page count and headings")
    ap.add_argument("--find", help="regex to search for, with page numbers")
    ap.add_argument("--page", type=int, help="print one page in full")
    ap.add_argument("--topics", action="store_true", help="scan for the topics we care about")
    ap.add_argument("--context", type=int, default=240)
    args = ap.parse_args()

    text = pages(args.pdf)
    print(f"{args.pdf.name}: {len(text)} pages, "
          f"{sum(len(t) for t in text):,} characters of extractable text\n")

    if args.page:
        print(text[args.page - 1])
        return 0

    if args.outline:
        for i, t in enumerate(text, 1):
            lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
            head = lines[0][:70] if lines else "(no text)"
            print(f"  p{i:>3}  {head}")
        return 0

    if args.find:
        pat = re.compile(args.find, re.I)
        hits = 0
        for i, t in enumerate(text, 1):
            for m in pat.finditer(t):
                a = max(0, m.start() - args.context // 2)
                snippet = " ".join(t[a:a + args.context].split())
                print(f"  p{i}: ...{snippet}...\n")
                hits += 1
                if hits >= 30:
                    print("  (stopping at 30)")
                    return 0
        if not hits:
            print("  no match")
        return 0

    if args.topics:
        for label, pat in TOPICS.items():
            found = [i for i, t in enumerate(text, 1) if re.search(pat, t)]
            where = ", ".join(f"p{i}" for i in found[:14])
            more = f" (+{len(found) - 14} more)" if len(found) > 14 else ""
            print(f"  {label:<24} {len(found):>3} pages" +
                  (f"  {where}{more}" if found else ""))
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
