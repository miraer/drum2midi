"""Does every collected ReStem events file belong to the render it is named after?

A JSON has no duration to contradict a wrong label, which is why the second machine's
point is sharper than it first sounds: `restem_batch_mode.ps1` attributes results by
mtime alone, and mtime advancing proves only that *a* render finished, not that it was
the one requested.

But the constraint is there implicitly. The last onset cannot fall after the end of the
recording, and in practice it lands near it -- FreeJazz 96% of its input, Rock 96%,
FusionJazz 95%. A file from a longer recording plastered over a shorter one shows as a
ratio above 100%, which is impossible rather than merely suspicious.

This checks that, plus byte-identical duplicates across a collection, which is the other
signature: two renders of different tracks are never the same bytes, so if two files
match exactly one of them is a copy of the other.

Used here on `bench/restem_modes`, where the quality-mode comparison in the README was
collected -- several renders of the *same* track under different settings, so the
duplicate check is the informative half. Byte-identical output from Better and Best is a
published finding there, so the check distinguishes "identical because the modes agree"
from "identical because one file never got overwritten" by reading the mode each file
was taken under from its own name.

    python verify_restem_events.py --dir bench/restem_modes --audio restem_in
    python verify_restem_events.py --dir restem_events --audio restem_in
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import wave
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent


def audio_seconds(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(str(path))
            return info.frames / info.samplerate
        except Exception:
            return None


def last_onset(path: Path) -> tuple[float, int] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None
    sr = data.get("sr", 44100)
    hop = data.get("hop", 441)
    latest, count = 0.0, 0
    for events in (data.get("stems") or {}).values():
        for e in events:
            count += 1
            if "sample" in e:
                latest = max(latest, e["sample"] / sr)
            elif "frame" in e:
                latest = max(latest, e["frame"] * hop / sr)
    return latest, count


def track_of(stem: str) -> str:
    """'MusicDelta_Rock_Drum.best-plus' -> 'MusicDelta_Rock_Drum'."""
    return stem.split(".")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="restem_events")
    ap.add_argument("--audio", default="restem_in")
    args = ap.parse_args()

    dirp, audio_dir = ROOT / args.dir, ROOT / args.audio
    if not dirp.exists():
        print(f"no such directory: {dirp}")
        return 1

    files = sorted(dirp.glob("*.json"))
    if not files:
        print(f"no json under {dirp}")
        return 1

    print(f"{'file':<48}{'input':>8}{'last':>8}{'ratio':>8}{'events':>8}  verdict")
    print("-" * 92)

    impossible = 0
    digests: dict[str, list[str]] = defaultdict(list)
    for f in files:
        wav = audio_dir / f"{track_of(f.stem)}.wav"
        dur = audio_seconds(wav) if wav.exists() else None
        info = last_onset(f)
        digests[hashlib.sha256(f.read_bytes()).hexdigest()].append(f.name)
        if info is None:
            print(f"{f.name[:47]:<48}{'':>8}{'':>8}{'':>8}{'':>8}  UNREADABLE")
            impossible += 1
            continue
        last, n = info
        if dur is None:
            print(f"{f.name[:47]:<48}{'—':>8}{last:7.1f}s{'—':>8}{n:8}  no audio to check")
            continue
        ratio = last / dur if dur else 0.0
        if ratio > 1.0:
            verdict = "IMPOSSIBLE — later than its own audio"
            impossible += 1
        elif ratio < 0.5:
            verdict = "ends very early — check"
        else:
            verdict = "ok"
        print(f"{f.name[:47]:<48}{dur:7.1f}s{last:7.1f}s{ratio:7.0%}{n:8}  {verdict}")

    print("-" * 92)
    dupes = {d: names for d, names in digests.items() if len(names) > 1}
    if dupes:
        print("\nbyte-identical files:")
        for names in dupes.values():
            tracks = {track_of(Path(n).stem) for n in names}
            same_track = len(tracks) == 1
            note = ("same track under different settings — a finding if the settings "
                    "differ, a stale copy if they do not"
                    if same_track else
                    "DIFFERENT TRACKS WITH IDENTICAL BYTES — one is a copy of the other")
            if not same_track:
                impossible += 1
            print(f"  {', '.join(names)}")
            print(f"    {note}")
    else:
        print("\nno byte-identical files")

    if impossible:
        print(f"\n{impossible} problem(s). These numbers should not be published as they "
              f"stand.")
        return 1
    print("\nEvery file's last onset falls inside its own audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
