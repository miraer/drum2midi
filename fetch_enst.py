"""Fetches ENST-Drums, the acoustic tom and ghost-note set, from its new Zenodo record.

Why this dataset. Our two test sets are close to blind in the class we score worst:
MDB Drums holds 90 tom onsets (1.14% of it) and IDMT-SMT-Drums holds none, which is why
the tom comparison against ReStem came back statistically meaningless. ENST has 2758 tom
onsets across 4 tom classes on real kits in real rooms -- and, unusually, ships isolated
close-miked tom stems, so it can tell us whether toms are lost by the separator or missed
by the transcriber. Those are different repairs.

Licence, which decides what we may do with it: CC BY-NC-ND 4.0. Non-commercial is fine
here, but the ND matters -- it is a research-use evaluation set, not training data for a
model whose weights we publish. E-GMD (CC BY 4.0) is the one to train on.

Until July 2026 this needed a click-through subscription form. It is now a single 9.61 GB
file on Zenodo, so the download is resumable and verifiable instead of manual.

    python fetch_enst.py --check        # prove it is reachable, download nothing
    python fetch_enst.py                # download, verify, unpack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
RECORD = "21506051"
API = f"https://zenodo.org/api/records/{RECORD}"
# Zenodo rejects requests without a plausible User-Agent
HEADERS = {"User-Agent": "drum2midi/1.0 (research use; +https://github.com/miraer/drum2midi)"}
CHUNK = 1 << 20


def fetch_metadata() -> dict:
    req = urllib.request.Request(API, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def pick_file(meta: dict) -> dict:
    files = meta.get("files", [])
    archives = [f for f in files if f.get("key", "").endswith((".tar.zst", ".tar.gz",
                                                               ".zip"))]
    if not archives:
        raise SystemExit(f"no archive in record {RECORD}: {[f.get('key') for f in files]}")
    return max(archives, key=lambda f: f.get("size", 0))


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def download(url: str, dest: Path, expect: int) -> None:
    """Resumable: 9.6 GB over a home connection does not always arrive in one piece."""
    done = dest.stat().st_size if dest.exists() else 0
    if done and done >= expect:
        print(f"already complete: {dest.name} ({human(done)})")
        return

    attempt = 0
    while done < expect:
        attempt += 1
        headers = dict(HEADERS)
        if done:
            headers["Range"] = f"bytes={done}-"
            print(f"resuming at {human(done)} ({done * 100 / expect:.1f}%)")
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, \
                    dest.open("ab" if done else "wb") as fh:
                started, last = time.time(), time.time()
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    fh.write(block)
                    done += len(block)
                    if time.time() - last > 30:
                        last = time.time()
                        rate = (done - (0 if attempt == 1 else 0)) / max(time.time() - started, 1)
                        print(f"  {human(done)} / {human(expect)} "
                              f"({done * 100 / expect:.1f}%)  {human(rate)}/s", flush=True)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt >= 12:
                raise SystemExit(f"giving up after {attempt} attempts: {exc}")
            print(f"  interrupted ({exc}); retrying in 30 s")
            time.sleep(30)
            done = dest.stat().st_size if dest.exists() else 0
    print(f"downloaded {human(done)}")


def verify(path: Path, checksum: str) -> bool:
    """Zenodo publishes 'md5:...'. A truncated 9 GB file fails confusingly later."""
    algo, _, want = checksum.partition(":")
    if algo != "md5":
        print(f"unexpected checksum type {algo}; skipping verification")
        return True
    h = hashlib.md5()
    seen = 0
    total = path.stat().st_size
    with path.open("rb") as fh:
        while block := fh.read(CHUNK * 8):
            h.update(block)
            seen += len(block)
            if seen % (512 * CHUNK) < CHUNK * 8:
                print(f"  hashing {seen * 100 / total:.0f}%", flush=True)
    ok = h.hexdigest() == want
    print(f"md5 {'matches' if ok else 'MISMATCH'}: {h.hexdigest()}")
    return ok


def unpack(archive: Path, out: Path) -> None:
    """Windows ships bsdtar built with libzstd, so .tar.zst needs no extra package."""
    out.mkdir(parents=True, exist_ok=True)
    if not shutil.which("tar"):
        raise SystemExit("no tar on PATH; unpack manually")
    print(f"unpacking into {out} ...")
    res = subprocess.run(["tar", "-xf", str(archive), "-C", str(out)],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace")
    if res.returncode != 0:
        raise SystemExit(f"tar failed: {(res.stderr or res.stdout)[-400:]}")
    print("unpacked")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="report what is there and exit without downloading")
    ap.add_argument("--out", type=Path, default=ROOT / "enst")
    ap.add_argument("--keep-archive", action="store_true",
                    help="do not delete the .tar.zst after unpacking")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    try:
        meta = fetch_metadata()
    except Exception as exc:
        print(f"could not reach Zenodo record {RECORD}: {exc}")
        return 1

    entry = pick_file(meta)
    title = meta.get("metadata", {}).get("title", "")
    licence = meta.get("metadata", {}).get("license", {}).get("id", "unknown")
    url = entry["links"]["self"]
    size = entry["size"]

    print(f"record  : {title}")
    print(f"licence : {licence}")
    print(f"file    : {entry['key']}  {human(size)}")
    print(f"checksum: {entry.get('checksum', 'none')}")

    if licence and "nd" in licence.lower():
        print("\nNote: this licence carries NoDerivatives. Use it to evaluate, not to\n"
              "train a model whose weights you then publish.")

    if args.check:
        free = shutil.disk_usage(ROOT).free
        print(f"\nfree space: {human(free)} — "
              f"{'enough' if free > size * 2.2 else 'TIGHT, needs ~2.2x the archive'}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    archive = args.out / entry["key"]
    download(url, archive, size)

    if not args.skip_verify and entry.get("checksum"):
        if not verify(archive, entry["checksum"]):
            print("checksum failed; delete the file and retry")
            return 1

    unpack(archive, args.out)
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"removed {archive.name}")

    wavs = sum(1 for _ in args.out.rglob("*.wav"))
    print(f"\n{args.out}: {wavs} wav files")
    print("next: python benchmark_enst.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
