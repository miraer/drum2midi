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

def remote_owner() -> str | None:
    """The account this repository belongs to, from the configured remote.

    Needed because the other two sources of identity are environment-dependent and one
    of them is wrong on CI. The owner is the same on every machine that has the remote,
    which makes it the only identity worth guarding that a build server can also know.
    """
    try:
        out = subprocess.run(["git", "remote", "get-url", "origin"],
                             capture_output=True, text=True, cwd=ROOT, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?\s*$", out.stdout.strip())
    return m.group(1) if m else None


HOME_CONTAINERS = ("users", "home")


def checkout_account(root: Path) -> str | None:
    r"""The account name in the folder a checkout sits in -- when that folder is a home.

    A checkout in `C:\Users\<name>\drum2midi` or `/home/<name>/drum2midi` names its
    owner in the path itself, and that is worth guarding independently of the username
    environment variable, which can be missing or wrong.

    Only when the parent really is a home, though. This used to take the parent's name
    whatever it was, and the machine this project is developed on keeps its checkout in
    `C:\dev\drum2midi` -- so `dev` became a guarded identity and every
    `for dev in devices:` in bench_training.py was reported as a leaked account name.
    A container folder (`dev`, `src`, `code`, `projects`) identifies nobody, and
    guarding one flags ordinary variable names: the same failure the CI robot account
    `runner` caused, in the same direction, for the same reason.

    So a home is recognised by shape rather than assumed: the parent is either the
    running user's own home, or it sits directly inside a home container.
    """
    parent = root.parent
    if parent.parent.name.lower() in HOME_CONTAINERS:
        return parent.name
    try:
        if parent == Path.home():
            return parent.name
    except RuntimeError:
        pass  # no resolvable home; the username and the remote owner still apply
    return None


def identity_names(user: str, root: Path, owner: str | None = None,
                   ci: bool = False) -> list[str]:
    """Names that identify this machine or its owner.

    Three sources, and the reason there are three is that the first two were wrong in
    opposite directions on a build server.

    The folder above a checkout names the owner when it is a home directory, which is
    what checkout_account decides and why this is not simply the parent's name. On
    GitHub Actions the layout is `work/<repo>/<repo>`, so the parent is the
    repository's own name, and every `from drum2midi import ...` was reported as a
    leaked account. A repository cannot leak its own name, so it is excluded here too.

    The ambient username is the author on a developer machine and a shared robot
    account on CI -- where it is `runner`, an ordinary English word that appears in
    this repository in the phrase "the batch runner waits". Guarding it there flagged
    prose. On CI it identifies nobody and is dropped.

    That would leave CI guarding nothing, which is worse, because the real account name
    could be committed and only a developer's own run would notice. So the owner from
    the git remote is added: knowable everywhere, identical everywhere, and already
    exempted where it legitimately appears, in the clone URL.
    """
    names = {checkout_account(root)}
    if user and not ci:
        names.add(user)
    if owner:
        names.add(owner)
    return sorted(n for n in names
                  if n and len(n) > 2 and n.lower() != root.name.lower())


# Checked separately from RULES so the message can say why, and so an empty username
# does not match everything.
IS_CI = bool(os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"))
IDENTITY = identity_names(USER, ROOT, remote_owner(), IS_CI)


def own_repo_url() -> re.Pattern | None:
    r"""The one place this repository's own account name is not a leak: its clone URL.

    A public repository's URL necessarily contains its owner. Telling people to clone
    `github.com/<you>/drum2midi` gives them a URL that does not work, so the README
    carries the real one.

    This is deliberately derived from the configured remote rather than hard-coded, and
    it matches the full `owner/repo` slug rather than the bare name. A stray mention of
    the account anywhere else still fails, which is the point -- the exemption is for
    one specific string that is already public by definition, not for the name.

    The trailing `(?![\w.-])` is load-bearing and was not there first. With a plain `\b`
    the pattern also matched `github.com/<owner>/<repo>-coord`, because a word boundary
    sits happily before a hyphen -- so the private repository this project uses to
    coordinate with a second machine would have been exempted by the rule meant to
    protect it. A test asserts that specific URL is still rejected.
    """
    try:
        out = subprocess.run(["git", "remote", "get-url", "origin"],
                             capture_output=True, text=True, cwd=ROOT, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?\s*$", out.stdout.strip())
    if not m:
        return None
    return re.compile(rf"github\.com[/:]{re.escape(m.group(1))}/"
                      rf"{re.escape(m.group(2))}(?:\.git)?(?![\w.-])", re.I)


OWN_REPO = own_repo_url()

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

# Extensions are a fast path, not the decision. The repository's own git hooks are
# extensionless shell scripts, so a machine path written into one of them was invisible
# to a check that matched on name -- as was a commit message, which is what found this.
# Anything small enough to read and free of NUL bytes is treated as text regardless of
# what it is called.
TEXT_SNIFF_BYTES = 64 * 1024


def looks_textual(path: Path) -> bool:
    if TEXT.search(path.name):
        return True
    try:
        if path.stat().st_size > TEXT_SNIFF_BYTES:
            return False
        head = path.read_bytes()
    except OSError:
        return False
    return b"\0" not in head

# Publishing something is a decision, and this is where the decision gets made rather
# than assumed. A file of one of these kinds appearing in a commit for the first time
# is stopped until a human has put it in .publish-allow.
#
# This exists because `git add -A` swept a 3 MB pitch video into an unrelated commit and
# pushed it, twice after saying in writing that it would not ship unreviewed. Every
# content check passed, because the content was fine -- the question nothing asked was
# whether the file belonged in the repository at all.
PUBLISHED_DELIBERATELY = Path(".publish-allow")
BINARY = re.compile(
    r"\.(mp4|mov|mkv|webm|avi|wav|mp3|flac|m4a|ogg|opus|zip|7z|gz|tar|rar|"
    r"ckpt|onnx|pth|pt|safetensors|bin|exe|dll|msi|pdf|psd|sketch)$", re.I)
BIG_BYTES = 512 * 1024

ALLOW = re.compile(
    r"<user>|<name>|<username>|\$env:|%USERNAME%|~/|placeholder|example\.com|"
    r"your\.name|C:\\\\Users\\\\<|noreply\.github\.com|users\.noreply",
    re.I)


def staged_files() -> list[Path]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
                         capture_output=True, text=True, cwd=ROOT,
                         encoding="utf-8", errors="replace")
    return [ROOT / line.strip() for line in out.stdout.splitlines() if line.strip()]


def newly_added() -> list[Path]:
    """Files this commit would add to the repository for the first time."""
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=A"],
                         capture_output=True, text=True, cwd=ROOT,
                         encoding="utf-8", errors="replace")
    return [ROOT / line.strip() for line in out.stdout.splitlines() if line.strip()]


def allowed_to_publish() -> set[str]:
    path = ROOT / PUBLISHED_DELIBERATELY
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line.replace("\\", "/"))
    return out


def check_new_files(paths: list[Path]) -> list[tuple[str, str]]:
    """Which of these additions should not be published without someone saying so."""
    allowed = allowed_to_publish()
    stopped = []
    for path in paths:
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            continue
        if rel in allowed:
            continue
        size = path.stat().st_size if path.exists() else 0
        if BINARY.search(path.name):
            stopped.append((rel, f"binary or media, {size/1024:.0f} KB"))
        elif size > BIG_BYTES and not TEXT.search(path.name):
            stopped.append((rel, f"{size/1024:.0f} KB and not a text file"))
    return stopped


def hook_installed() -> bool:
    """Is the check wired to run by itself, or does someone have to remember it?"""
    out = subprocess.run(["git", "config", "--get", "core.hooksPath"],
                         capture_output=True, text=True, cwd=ROOT,
                         encoding="utf-8", errors="replace")
    configured = out.stdout.strip()
    if configured and (ROOT / configured / "pre-commit").exists():
        return True
    return (ROOT / ".git" / "hooks" / "pre-commit").exists()


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, cwd=ROOT,
                         encoding="utf-8", errors="replace")
    return [ROOT / line.strip() for line in out.stdout.splitlines() if line.strip()]


def scan(paths: list[Path], explicit: bool = False) -> tuple[list, list, int]:
    fatal, warn = [], []
    read = 0

    for path in paths:
        # A file outside the repository is a normal thing to check -- a draft held
        # elsewhere on purpose, inspected before anyone is tempted to move it in. This
        # used to raise ValueError from relative_to and print a traceback instead of a
        # verdict, which is the least useful moment for a tool like this to fail.
        if path.is_absolute():
            try:
                rel = path.relative_to(ROOT).as_posix()
            except ValueError:
                rel = path.as_posix()
        else:
            rel = str(path)

        for pattern, why in FORBIDDEN:
            if pattern.search(rel):
                fatal.append((rel, 0, why, rel))

        # a personal recording usually arrives as a non-ASCII audio or MIDI filename
        if MEDIA.search(rel) and any(ord(c) > 127 for c in rel):
            fatal.append((rel, 0, "media file with a non-ASCII name", rel))

        # The extension list is a fast path, not the decision. A file somebody named on
        # purpose is always read; anything else is read when it sniffs as text. A commit
        # message exposed the old behaviour -- COMMIT_EDITMSG has no extension, so an
        # explicit check of it reported "checked 1 file(s)" while reading nothing, and a
        # machine path in the message passed untouched. The repository's own git hooks
        # are extensionless too, and were equally invisible.
        if not path.exists():
            continue
        if not explicit and not looks_textual(path):
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
        read += 1

        for n, line in enumerate(lines, 1):
            if ALLOW.search(line):
                continue
            for label, pattern, is_fatal in RULES:
                m = pattern.search(line)
                if m:
                    (fatal if is_fatal else warn).append(
                        (rel, n, label, m.group(0)[:70]))
            for name in IDENTITY:
                for m in re.finditer(rf"\b{re.escape(name)}\b", line, re.I):
                    # inside this repository's own clone URL the owner is public by
                    # definition; anywhere else on the line it is still a leak
                    if OWN_REPO and any(u.start() <= m.start() and m.end() <= u.end()
                                        for u in OWN_REPO.finditer(line)):
                        continue
                    fatal.append((rel, n, f"machine or account name '{name}'",
                                  line.strip()[:70]))
                    break
    return fatal, warn, read


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

    fatal, warn, read = scan(paths, explicit=bool(args.files))
    # A path that is not there was folded into "not a text type" and the run still
    # exited 0, so asking about a file that had moved, or passing one relative to the
    # wrong directory, produced a clean bill for a check that never happened. Green has
    # to mean "looked and found nothing"; it cannot also mean "could not look".
    missing = [p for p in paths if not p.exists()]
    # Naming the skipped files rather than counting them. A commit reported "1 not a
    # text type" over two staged .py files and the run could not be reproduced
    # afterwards, because the summary said how many were skipped and never which. A
    # count cannot be investigated; a name can. It turned out to be this file: the
    # scanner exempts itself by exact name, and the summary filed that under "binary",
    # which is a different claim about a different reason.
    def rel_of(p: Path) -> str:
        if p.is_absolute():
            try:
                return p.relative_to(ROOT).as_posix()
            except ValueError:
                return p.as_posix()
        return str(p)

    exempt = [p for p in paths if p.exists() and rel_of(p) == "check_privacy.py"]
    skipped = [p for p in paths
               if p.exists() and not args.files and not looks_textual(p)
               and p not in exempt]
    # "checked N" used to count files the scanner had only looked at the name of, so a
    # skipped file read as an inspected one. Say how many were actually opened.
    if read == len(paths):
        print(f"checked {len(paths)} file(s)")
    else:
        parts = [f"{read} read for content"]
        if skipped:
            parts.append(f"{len(skipped)} not a text type")
        if exempt:
            parts.append(f"{len(exempt)} self-exempt")
        if missing:
            parts.append(f"{len(missing)} NOT FOUND")
        unaccounted = len(paths) - read - len(skipped) - len(exempt) - len(missing)
        if unaccounted:
            parts.append(f"{unaccounted} unaccounted")
        print(f"checked {len(paths)} file(s), " + ", ".join(parts))
        for p in skipped[:10]:
            try:
                size = p.stat().st_size
            except OSError:
                size = -1
            print(f"  skipped as binary: {p} ({size} bytes)")
    if missing:
        print(f"\n{len(missing)} path(s) could not be read, so nothing is known about "
              f"them:")
        for p in missing[:10]:
            print(f"  {p}")
        print("\nA check that could not run is not a check that passed.")
        return 1

    # Only meaningful for a real commit: an explicit file list or --all is someone
    # asking about content, not about what is being published.
    new_stopped = [] if (args.files or args.all) else check_new_files(newly_added())

    if warn:
        print(f"\n{len(warn)} thing(s) worth a look:")
        for rel, n, label, hit in warn[:20]:
            where = f"{rel}:{n}" if n else rel
            print(f"  {where}  {label}: {hit}")

    if new_stopped:
        print(f"\n{len(new_stopped)} file(s) this commit would publish for the first "
              f"time:")
        for rel, why in new_stopped:
            print(f"  {rel}  ({why})")
        print("\nNothing here is necessarily wrong. The point is that publishing a\n"
              "binary is a decision, and `git add -A` makes it by accident. If it\n"
              "should ship, say so once:\n")
        for rel, _why in new_stopped:
            print(f"    echo {rel} >> {PUBLISHED_DELIBERATELY}")
        print("\nIf it should not, unstage it and add it to .gitignore.")

    if fatal:
        print(f"\n{len(fatal)} thing(s) that must not be committed:")
        for rel, n, label, hit in fatal[:30]:
            where = f"{rel}:{n}" if n else rel
            print(f"  {where}  {label}: {hit}")
        print("\nFix them, or if one is a false positive, make that obvious in the text\n"
              "(a placeholder like <user>) rather than weakening the check.")

    if fatal or new_stopped:
        return 1

    print("no machine paths, personal names, private drafts or credentials found")
    if not (args.files or args.all) and not hook_installed():
        print("\nNote: this check is not wired to run by itself. One command fixes "
              "that:\n    git config core.hooksPath hooks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
