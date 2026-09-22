"""Fetches the Expanded Groove MIDI Dataset, and checks what is actually in it.

E-GMD is the one dataset that could fix both weaknesses this project has measured in
itself: toms and ghost notes. It is CC BY 4.0, so unlike ENST-Drums (CC BY-NC-ND) it may
be trained on and the resulting weights published.

The MIDI-only archive is 102 MB against 90 GB for the audio, and it carries the whole
annotation -- every onset, every velocity. So the claims worth checking can be checked
before committing to the big download: how many tom onsets there really are, and how the
velocities are distributed. That is the point of --survey.

Disk: the audio archive downloads 90 GB and unpacks to 131 GB. Both exist at once --
`extractall` finishes before the archive is deleted -- so the number a machine has to have
free is the **peak of about 221 GB**, not the 131 GB it settles to afterwards. An earlier
version of this line promised "about 135 GB" on the strength of the deletion, which
describes the state after the run and not the moment it fails: a machine with 150 GB free
gets four hours into the download and dies partway through the unpack. `--keep-archive`
does not change the peak, only what is left at the end. Checked before anything is fetched.
Measured on the second machine, which reported 45537 wavs matching the metadata row for row.

Caveat that survives any survey: the audio is a Roland TD-17 electronic kit. No room, no
mic bleed, no cymbal wash. Expect a domain gap to acoustic drums and measure it rather
than assuming it away -- a previous attempt to fix toms with synthetic audio, ADT_STR,
scored 0.140 on toms.

    python fetch_egmd.py --survey        # 102 MB, then count what is inside
    python fetch_egmd.py --audio         # the full 90 GB
"""

from __future__ import annotations

import argparse
import collections
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
BASE = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0"
MIDI_ZIP = f"{BASE}/e-gmd-v1.0.0-midi.zip"
FULL_ZIP = f"{BASE}/e-gmd-v1.0.0.zip"
HEADERS = {"User-Agent": "drum2midi/1.0 (research use)"}
CHUNK = 1 << 20

# Roland TD kits deviate from General MIDI, so the pads have to be mapped by hand.
# Taken from the Groove MIDI Dataset's own "Drum Mapping" table.
ROLAND = {
    36: ("kick", "kick"),
    38: ("snare head", "snare"), 40: ("snare rim", "snare"),
    37: ("snare x-stick", "snare"),
    48: ("tom 1", "tom"), 50: ("tom 1 rim", "tom"),
    45: ("tom 2", "tom"), 47: ("tom 2 rim", "tom"),
    43: ("tom 3 head", "tom"), 58: ("tom 3 rim", "tom"),
    46: ("hh open bow", "hi-hat"), 26: ("hh open edge", "hi-hat"),
    42: ("hh closed bow", "hi-hat"), 22: ("hh closed edge", "hi-hat"),
    44: ("hh pedal", "hi-hat"),
    49: ("crash 1 bow", "cymbal"), 55: ("crash 1 edge", "cymbal"),
    57: ("crash 2 bow", "cymbal"), 52: ("crash 2 edge", "cymbal"),
    51: ("ride bow", "cymbal"), 59: ("ride edge", "cymbal"),
    53: ("ride bell", "cymbal"),
    # Auxiliary pads that appear on some kits. Not part of the five classes this
    # project transcribes, but naming them stops them being reported as "unmapped",
    # which invites the suspicion that the mapping is wrong.
    54: ("tambourine", "auxiliary"), 39: ("hand clap", "auxiliary"),
    56: ("cowbell", "auxiliary"),
}


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


# Archive and unpacked size in GiB, measured on the second machine. The pair matters
# rather than the total, because both exist simultaneously partway through the run.
FOOTPRINT = {"audio": (89.8, 131.0), "midi": (0.1, 0.4)}


def enough_space(free_gib: float, kind: str, keep_archive: bool = False) -> tuple[bool, str]:
    """Whether a run can finish, judged on the high-water mark rather than the end state.

    Separated from the disk query so the decision can be tested with a number instead of a
    filesystem. The distinction it exists to make: `extractall()` completes before the
    archive is unlinked, so the zip and its contents coexist and the peak is their sum. The
    docstring used to quote the post-deletion figure as the requirement, which is the one
    number that cannot fail -- it is only ever observed after the risky part is over.

    `--keep-archive` does not raise the peak. It only decides whether the run gives the
    archive back afterwards, which is why it is reported separately.
    """
    archive, unpacked = FOOTPRINT[kind]
    peak = archive + unpacked
    settles_to = peak if keep_archive else unpacked
    ok = free_gib >= peak * 1.03

    def gib(n: float) -> str:
        return f"{n:.1f}" if n < 10 else f"{n:.0f}"

    note = (f"{free_gib:.1f} GiB free; needs about {gib(peak)} GiB at peak "
            f"({gib(archive)} archive + {gib(unpacked)} unpacked, both present during the "
            f"unpack), settling to {gib(settles_to)} GiB")
    return ok, note


def require_space(out: Path, kind: str, keep_archive: bool) -> None:
    free = shutil.disk_usage(out).free / (1 << 30)
    ok, note = enough_space(free, kind, keep_archive)
    print(note)
    if not ok:
        raise SystemExit(
            f"\nNot enough disk for the {kind} archive, so this would fail partway through "
            f"the unpack rather than now.\nFree space, or pass --out on another drive.")


def download(url: str, dest: Path) -> None:
    """Resumable, because the audio archive is 90 GB."""
    req = urllib.request.Request(url, headers=HEADERS, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        expect = int(r.headers.get("Content-Length", 0))
    done = dest.stat().st_size if dest.exists() else 0
    if done >= expect > 0:
        print(f"already have {dest.name} ({human(done)})")
        return

    print(f"{dest.name}: {human(expect)}")
    attempt = 0
    while done < expect:
        attempt += 1
        headers = dict(HEADERS)
        if done:
            headers["Range"] = f"bytes={done}-"
            print(f"  resuming at {human(done)} ({done * 100 / expect:.1f}%)")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp, \
                    dest.open("ab" if done else "wb") as fh:
                last = time.time()
                while block := resp.read(CHUNK):
                    fh.write(block)
                    done += len(block)
                    if time.time() - last > 30:
                        last = time.time()
                        print(f"  {human(done)} / {human(expect)} "
                              f"({done * 100 / expect:.1f}%)", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt >= 12:
                raise SystemExit(f"giving up after {attempt} attempts: {exc}")
            print(f"  interrupted ({exc}); retrying in 30 s")
            time.sleep(30)
            done = dest.stat().st_size if dest.exists() else 0
    print(f"  done, {human(done)}")


def survey(folder: Path) -> None:
    """Counts every note-on, so the published claims can be checked rather than quoted."""
    import pretty_midi

    files = sorted(folder.rglob("*.mid*"))
    if not files:
        print(f"no MIDI under {folder}")
        return
    print(f"\nparsing {len(files)} MIDI files (this takes a few minutes) ...")

    pitches = collections.Counter()
    vels = collections.Counter()
    bad = 0
    for i, path in enumerate(files, 1):
        try:
            pm = pretty_midi.PrettyMIDI(str(path))
        except Exception:
            bad += 1
            continue
        for inst in pm.instruments:
            for n in inst.notes:
                pitches[n.pitch] += 1
                vels[n.velocity] += 1
        if i % 5000 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    total = sum(pitches.values())
    print(f"\n{len(files) - bad} files parsed, {bad} unreadable, {total} note-ons")

    fams = collections.Counter()
    for pitch, n in pitches.items():
        fams[ROLAND.get(pitch, ("?", "unmapped"))[1]] += n
    print("\nby family")
    for fam, n in fams.most_common():
        print(f"  {fam:<10}{n:>10}  {100 * n / total:5.2f}%")

    print("\ntoms in detail — the class MDB has only 90 of")
    for pitch, n in sorted(pitches.items(), key=lambda kv: -kv[1]):
        if ROLAND.get(pitch, ("", ""))[1] == "tom":
            print(f"  {pitch:>3} {ROLAND[pitch][0]:<14}{n:>9}")

    if vels:
        lo = sum(n for v, n in vels.items() if v < 60)
        lower = sum(n for v, n in vels.items() if v < 40)
        vtot = sum(vels.values())
        mean = sum(v * n for v, n in vels.items()) / vtot
        print(f"\nvelocity: {min(vels)}..{max(vels)}, mean {mean:.1f}, "
              f"{len(vels)} distinct values")
        print(f"  below 60: {lo} ({100 * lo / vtot:.2f}%)   "
              f"below 40: {lower} ({100 * lower / vtot:.2f}%)")
        print("  (ghost-note material; MDB annotates ghosts but carries no velocity)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", action="store_true",
                    help="fetch the full 90 GB archive instead of MIDI only")
    ap.add_argument("--survey", action="store_true",
                    help="count onsets and velocities after unpacking")
    ap.add_argument("--out", type=Path, default=ROOT / "egmd")
    ap.add_argument("--keep-archive", action="store_true",
                    help="do not delete the .zip after unpacking")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    url = FULL_ZIP if args.audio else MIDI_ZIP
    require_space(args.out, "audio" if args.audio else "midi", args.keep_archive)
    archive = args.out / url.rsplit("/", 1)[1]
    download(url, archive)

    target = args.out / archive.stem
    if not target.exists():
        print(f"unpacking into {target} ...")
        with zipfile.ZipFile(archive) as z:
            z.extractall(target)
        print("unpacked")
        # The audio zip unpacks 90 GB into 131 GB and keeping both needs 220 GB, which
        # runs a machine out of disk near the end of a four-hour download. fetch_enst.py
        # has deleted its archive by default since it was written; this one did not.
        if not args.keep_archive:
            archive.unlink(missing_ok=True)
            print(f"removed {archive.name}")

    if args.survey:
        survey(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
