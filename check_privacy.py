"""Refuses to let private material reach a commit.

Written after a draft letter to a third party was committed by mistake. It held our
reasoning about how to approach that company -- when to send, what they might object to
-- which belongs nowhere near a repository that may become public. The history had to be
rewritten to remove it.

So this is a check rather than a good intention. It looks at what is staged, or at any
files given to it, for four kinds of leak:

  machine paths     C:\\Users\\<name>, /home/<name>, %APPDATA%
  personal names    whoever is logged in, and the repository's own account name
  private material  correspondence drafts, notes to self, anything under docs/private
  credentials       tokens, keys, passwords, connection strings

Song titles and other media filenames are harder: they are not a fixed pattern. What
this can do is flag absolute paths into media folders and any non-ASCII filename in an
audio or MIDI extension, which is what a personal recording usually looks like here.

    python check_privacy.py              # what is staged right now
    python check_privacy.py --all        # every tracked file
    python check_privacy.py f1.py f2.md  # specific files
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

USER = os.environ.get("USERNAME") or os.environ.get("USER") or ""

# (label, pattern, whether a hit is fatal). Non-fatal ones are worth a look but are
# routinely legitimate -- a DAW's name in a table of note conventions, for instance.
RULES = [
    ("windows user path", re.compile(r"[A-Za-z]:\\+Users\\+(?!<)[^\\\s\"']+", re.I), True),
    ("unix home path", re.compile(r"/(?:home|Users)/(?!<)[a-z][a-z0-9._-]+", re.I), True),
    ("appdata or roaming", re.compile(r"%APPDATA%|\\AppData\\|/Library/Application Support/",
                                      re.I), True),
    ("cloud drive path", re.compile(r"\b(OneDrive|Dropbox|Google Drive|iCloud Drive)\b",
                                    re.I), True),
    ("access token", re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{16,}|sk-[A-Za-z0-9]{20,}|"
                                r"xox[baprs]-[A-Za-z0-9-]{10,})"), True),
    ("secret assignment", re.compile(r"\b(password|passwd|secret|api[_-]?key|access[_-]?key|"
                                     r"private[_-]?key|token)\s*[=:]\s*[\"'][^\"']{6,}",
                                     re.I), True),
    ("private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), True),
    ("connection string", re.compile(r"\b\w+://[^/\s:@]+:[^/\s@]+@"), True),
    ("email address", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), False),
]

# Names that identify this machine or its owner. Checked separately so the message can
# say why, and so an empty username does not match everything.
IDENTITY = [n for n in {USER, ROOT.parent.name} if len(n) > 2]

# Files that must never be committed, whatever is in them.
FORBIDDEN = [
    (re.compile(r"(^|/)docs/letter-", re.I), "correspondence draft"),
    (re.compile(r"(^|/)docs/private/", re.I), "marked private"),
    (re.compile(r"\.draft\.md$", re.I), "draft"),
    (re.compile(r"(^|/)(notes|todo|scratch)\.(md|txt)$", re.I), "personal notes"),
    (re.compile(r"\.(env|pem|key|p12|pfx)$", re.I), "credential file"),
]

MEDIA = re.compile(r"\.(wav|mp3|flac|aiff?|m4a|ogg|opus|mid|midi)$", re.I)
TEXT = re.compile(r"\.(py|pyw|md|txt|json|ya?ml|ps1|sh|bat|cfg|ini|toml|gitignore)$", re.I)

ALLOW = re.compile(
    r"<user>|<name>|<username>|\$env:|%USERNAME%|~/|placeholder|example\.com|"
    r"your\.name|C:\\\\Users\\\\<|noreply\.github\.com|users\.noreply",
    re.I)


def staged_files() -> list[Path]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                         capture_output=True, text=True, cwd=ROOT,
                         encoding="utf-8", errors="replace")
    return [ROOT / line.strip() for line in out.stdout.splitlines() if line.strip()]


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT,
                         encoding="utf-8", errors="replace")
    return [ROOT / line.strip() for line in out.stdout.splitlines() if line.strip()]


def scan(paths: list[Path]) -> tuple[list, list]:
    fatal, warn = [], []

    for path in paths:
        rel = path.relative_to(ROOT).as_posix() if path.is_absolute() else str(path)

        for pattern, why in FORBIDDEN:
            if pattern.search(rel):
                fatal.append((rel, 0, why, rel))

        # a personal recording usually arrives as a non-ASCII audio or MIDI filename
        if MEDIA.search(rel) and any(ord(c) > 127 for c in rel):
            fatal.append((rel, 0, "media file with a non-ASCII name", rel))

        if not TEXT.search(rel) or not path.exists():
            continue
        # This file defines the patterns, so it necessarily contains them. It is the
        # one exemption, and it is by exact name rather than by a rule anything else
        # could match.
        if rel == "check_privacy.py":
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for n, line in enumerate(lines, 1):
            if ALLOW.search(line):
                continue
            for label, pattern, is_fatal in RULES:
                m = pattern.search(line)
                if m:
                    (fatal if is_fatal else warn).append(
                        (rel, n, label, m.group(0)[:70]))
            for name in IDENTITY:
                if re.search(rf"\b{re.escape(name)}\b", line, re.I):
                    fatal.append((rel, n, f"machine or account name '{name}'",
                                  line.strip()[:70]))
    return fatal, warn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--all", action="store_true", help="check every tracked file")
    args = ap.parse_args()

    if args.files:
        paths = [p if p.is_absolute() else ROOT / p for p in args.files]
    elif args.all:
        paths = tracked_files()
    else:
        paths = staged_files()

    if not paths:
        print("nothing to check")
        return 0

    fatal, warn = scan(paths)
    print(f"checked {len(paths)} file(s)")

    if warn:
        print(f"\n{len(warn)} thing(s) worth a look:")
        for rel, n, label, hit in warn[:20]:
            where = f"{rel}:{n}" if n else rel
            print(f"  {where}  {label}: {hit}")

    if fatal:
        print(f"\n{len(fatal)} thing(s) that must not be committed:")
        for rel, n, label, hit in fatal[:30]:
            where = f"{rel}:{n}" if n else rel
            print(f"  {where}  {label}: {hit}")
        print("\nFix them, or if one is a false positive, make that obvious in the text\n"
              "(a placeholder like <user>) rather than weakening the check.")
        return 1

    print("no machine paths, personal names, private drafts or credentials found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
