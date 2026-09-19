"""Draws MIDI tracks against the audio they came from, on a shared bar ruler.

Written to settle a "why is the output shorter than the song?" question, where the
answer turned out to depend on three things that are easy to conflate: where the last
note is, where the music actually stops, and where the bar lines fall. Numbers alone did
not settle it; a picture does.

Bar numbers come from ticks, not from seconds, so they are the same in any DAW. If a bar
ruler here disagrees with the one on screen, the project tempo differs from the tempo in
the file.

    python plot_timeline.py song.wav ours.mid
    python plot_timeline.py song.wav ours.mid theirs.mid --labels drum2midi ReStem
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pretty_midi  # noqa: E402
import soundfile as sf  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from drum_icons import colour  # noqa: E402

INK = (24, 28, 34)
DIM = (150, 158, 170)
PAPER = (250, 251, 252)
WAVE = (56, 132, 222)
GRID = (222, 226, 232)


def font(px: int, bold: bool = False):
    names = (("segoeuisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf") if bold
             else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"))
    for name in names:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def envelope(path: Path, columns: int) -> tuple:
    """Peak per column, plus the duration, without loading the file twice."""
    info = sf.info(str(path))
    total = info.frames
    step = max(1, total // columns)
    peaks = np.zeros(columns)
    with sf.SoundFile(str(path)) as fh:
        for i in range(columns):
            fh.seek(min(i * step, max(total - 1, 0)))
            block = fh.read(step, dtype="float32", always_2d=True)
            if len(block):
                peaks[i] = float(np.abs(block).max())
    return peaks, info.duration


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("midi", type=Path, nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "timeline.png")
    ap.add_argument("--width", type=int, default=1600)
    args = ap.parse_args()

    labels = args.labels or [p.stem[:24] for p in args.midi]
    tracks = []
    for path in args.midi:
        pm = pretty_midi.PrettyMIDI(str(path))
        notes = [(n.start, n.pitch) for inst in pm.instruments for n in inst.notes]
        _, tempi = pm.get_tempo_changes()
        tracks.append({"notes": sorted(notes), "bpm": float(tempi[0]) if len(tempi) else 120.0,
                       "path": path})

    W = args.width
    left, right = 150, 40
    plot_w = W - left - right
    wave_h, lane_h, ruler_h = 120, 130, 46
    H = ruler_h + wave_h + lane_h * len(tracks) + 70

    peaks, duration = envelope(args.audio, plot_w)
    span = max(duration, max((t["notes"][-1][0] if t["notes"] else 0) for t in tracks)) * 1.01

    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)

    def x_of(sec: float) -> float:
        return left + plot_w * sec / span

    # bar ruler, from the first track's tempo
    bpm = tracks[0]["bpm"]
    bar = 4 * 60 / bpm
    y0 = 10
    d.text((left, y0), f"bars at {bpm:.2f} BPM (4/4)", font=font(15, True), fill=INK)
    every = max(1, int(round(span / bar / 40)))
    n = 1
    while (n - 1) * bar <= span:
        x = x_of((n - 1) * bar)
        strong = (n - 1) % (every * 4) == 0
        d.line([x, y0 + 24, x, H - 34], fill=GRID, width=2 if strong else 1)
        if (n - 1) % every == 0:
            d.text((x + 3, y0 + 24), str(n), font=font(12), fill=DIM)
        n += 1

    # audio envelope
    wy = y0 + ruler_h
    mid = wy + wave_h / 2
    d.text((10, int(mid) - 8), "audio", font=font(14, True), fill=INK)
    for i, p in enumerate(peaks):
        h = p * (wave_h / 2 - 4)
        d.line([left + i, mid - h, left + i, mid + h], fill=WAVE)
    d.line([left, mid, left + plot_w, mid], fill=(200, 210, 224))
    d.text((x_of(duration) + 4, mid - 22), f"{duration:.1f}s",
           font=font(12), fill=DIM)
    d.line([x_of(duration), wy, x_of(duration), H - 34], fill=(230, 120, 120), width=2)

    # one lane per MIDI file
    pitches = sorted({p for t in tracks for _, p in t["notes"]})
    for row, track in enumerate(tracks):
        ly = wy + wave_h + row * lane_h
        d.rectangle([left, ly, left + plot_w, ly + lane_h - 14],
                    fill=(255, 255, 255), outline=(236, 239, 243))
        last = track["notes"][-1][0] if track["notes"] else 0
        d.text((10, ly + 6), labels[row], font=font(14, True), fill=INK)
        d.text((10, ly + 26), f"{len(track['notes'])} notes", font=font(12), fill=DIM)
        d.text((10, ly + 44), f"ends {last:.1f}s", font=font(12), fill=DIM)
        d.text((10, ly + 62), f"bar {last / (4 * 60 / track['bpm']) + 1:.1f}",
               font=font(12), fill=DIM)

        usable = lane_h - 24
        for start, pitch in track["notes"]:
            idx = pitches.index(pitch)
            y = ly + 6 + usable * (1 - idx / max(len(pitches) - 1, 1))
            c = tuple(int(colour(pitch)[i:i + 2], 16) for i in (1, 3, 5))
            x = x_of(start)
            d.line([x, y - 4, x, y + 4], fill=c, width=2)
        # where this track's last note falls
        d.line([x_of(last), ly, x_of(last), ly + lane_h - 14], fill=(120, 190, 120),
               width=2)

    d.text((left, H - 26),
           "red line: end of audio   green line: last note   "
           "bar numbers come from ticks, so they do not depend on the project tempo",
           font=font(13), fill=DIM)

    args.out.parent.mkdir(exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out}  ({args.out.stat().st_size // 1024} KB)")
    for label, track in zip(labels, tracks):
        last = track["notes"][-1][0] if track["notes"] else 0
        print(f"  {label:<14}{len(track['notes']):>6} notes   last {last:7.2f}s   "
              f"bar {last / (4 * 60 / track['bpm']) + 1:6.2f}   tempo {track['bpm']:.2f}")
    print(f"  {'audio':<14}{'':>6}         ends {duration:7.2f}s   "
          f"bar {duration / bar + 1:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
