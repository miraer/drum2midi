"""Draws the project logo: a drum hit turning into MIDI notes.

Kept as code rather than as a binary blob so it can be regenerated at any size, tweaked
in review, and diffed. The image says what the tool does: an audio transient on the left
(sharp attack, exponential decay, the shape of a struck drum), the same events on the
right as note blocks on a grid, and the arrow of the transcription between them.

    python make_logo.py                 # icon, banner and .ico into docs/
    python make_logo.py --size 512      # a single large PNG
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Pillow is imported inside the drawing functions rather than here, the way
# drum_icons.py does it. Pillow is optional in this project: the GUI degrades without
# it and requirements.txt says so. Importing it at module scope meant that reading the
# palette constants below required a library that drawing needs and reading does not,
# so a smoke test that compared them errored out on any install without Pillow instead
# of checking anything.

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
sys.path.insert(0, str(ROOT))

from drum_icons import _rgb, colour  # noqa: E402

INK = (24, 28, 34)
WAVE = (56, 132, 222)
DIM = (150, 158, 170)
PAPER = (250, 251, 252)

# A bar of a plain rock beat, as (start, length, lane, pitch). Lane 0 is the highest
# pitch, lane 3 the lowest, the way a piano roll orders them. The colours come from
# drum_icons so the logo and the results table always agree.
BLOCKS = [
    (0.00, 0.15, 0, 49),                                    # crash on the downbeat
    (0.00, 0.13, 1, 42), (0.26, 0.13, 1, 42),               # hi-hat
    (0.52, 0.13, 1, 42), (0.78, 0.13, 1, 42),
    (0.26, 0.18, 2, 38), (0.78, 0.18, 2, 38),               # snare on 2 and 4
    (0.00, 0.20, 3, 36), (0.52, 0.20, 3, 36),               # kick
]


def drum_envelope(x: float) -> float:
    """Three struck drums: near-instant attack, exponential decay, slight ring.

    One long decay read as a single smear at small sizes, and its tail collided with
    the arrow. Three hits look like an actual drum track and leave the middle clear.
    """
    peaks = ((0.00, 1.00), (0.38, 0.62), (0.66, 0.85))
    value = 0.0
    for start, height in peaks:
        if x < start:
            continue
        t = x - start
        if t < 0.015:
            value = max(value, height * (t / 0.015))
        else:
            decay = math.exp(-11.0 * (t - 0.015))
            value = max(value, height * decay * (0.80 + 0.20 * math.cos(46 * t)))
    return value


def draw_logo(size: int, transparent: bool = False) -> Image.Image:
    from PIL import Image, ImageDraw
    # supersample, then downscale: gives clean edges without any antialiasing code
    scale = 4
    s = size * scale
    bg = (0, 0, 0, 0) if transparent else PAPER + (255,)
    img = Image.new("RGBA", (s, s), bg)
    d = ImageDraw.Draw(img)

    pad = int(s * 0.10)
    radius = int(s * 0.22)
    if not transparent:
        d.rounded_rectangle([pad // 2, pad // 2, s - pad // 2, s - pad // 2],
                            radius=radius, fill=(255, 255, 255, 255),
                            outline=INK + (255,), width=max(2, int(s * 0.016)))

    mid_y = s // 2
    left_x0, left_x1 = pad + int(s * 0.05), int(s * 0.41)
    right_x0, right_x1 = int(s * 0.56), s - pad - int(s * 0.05)

    # left: the waveform, drawn as a filled envelope around the centre line
    bars = 30
    bar_w = max(2, int((left_x1 - left_x0) / (bars * 1.8)))
    span = int(s * 0.28)
    for i in range(bars):
        f = i / (bars - 1)
        x = left_x0 + int(f * (left_x1 - left_x0))
        h = int(span * drum_envelope(f))
        h = max(h, int(s * 0.007))
        d.rounded_rectangle([x, mid_y - h, x + bar_w, mid_y + h],
                            radius=bar_w // 2, fill=WAVE + (255,))

    # right: note blocks on a faint grid, the way a DAW shows them
    lanes = 4
    lane_h = int(s * 0.115)
    grid_top = mid_y - int(lanes * lane_h / 2)
    for i in range(lanes + 1):
        y = grid_top + i * lane_h
        d.line([right_x0, y, right_x1, y], fill=DIM + (90,), width=max(1, int(s * 0.004)))

    width = right_x1 - right_x0
    block_h = int(lane_h * 0.52)
    for start, length, lane, pitch in BLOCKS:
        x0 = right_x0 + int(start * width)
        x1 = x0 + max(int(length * width), int(s * 0.03))
        y = grid_top + lane * lane_h + (lane_h - block_h) // 2
        d.rounded_rectangle([x0, y, x1, y + block_h],
                            radius=block_h // 3, fill=_rgb(colour(pitch)) + (255,))

    # the arrow: audio becomes notes. Kept short and clear of both sides so it reads
    # as a transformation rather than as part of the waveform.
    ax0 = left_x1 + int(s * 0.035)
    ax1 = right_x0 - int(s * 0.025)
    ay = mid_y
    head = int(s * 0.040)
    d.line([ax0, ay, ax1 - head * 0.8, ay], fill=INK + (210,),
           width=max(2, int(s * 0.014)))
    d.polygon([(ax1, ay), (ax1 - head, ay - head * 0.60),
               (ax1 - head, ay + head * 0.60)], fill=INK + (230,))

    return img.resize((size, size), Image.LANCZOS)


def draw_banner(width: int = 1280, height: int = 360) -> Image.Image:
    from PIL import Image, ImageDraw, ImageFont
    scale = 2
    img = Image.new("RGB", (width * scale, height * scale), PAPER)
    d = ImageDraw.Draw(img)
    w, h = width * scale, height * scale

    icon_size = int(h * 0.62)
    icon = draw_logo(icon_size, transparent=True)
    img.paste(icon, (int(w * 0.07), (h - icon_size) // 2), icon)

    def font(px: int):
        for name in ("segoeuisb.ttf", "seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
            try:
                return ImageFont.truetype(name, px)
            except OSError:
                continue
        return ImageFont.load_default()

    tx = int(w * 0.07) + icon_size + int(w * 0.05)
    d.text((tx, int(h * 0.30)), "drum2midi", font=font(int(h * 0.20)), fill=INK)
    d.text((tx, int(h * 0.56)), "drum recordings to General MIDI — measured, not guessed",
           font=font(int(h * 0.085)), fill=(90, 98, 110))

    return img.resize((width, height), Image.LANCZOS)


def draw_social(width: int = 1280, height: int = 640) -> Image.Image:
    """GitHub's social preview card, shown wherever the repo link is pasted.

    Deliberately not the banner at a different aspect ratio. This is often the only
    thing someone sees before deciding whether to click, so it carries the one claim
    that distinguishes the project: a measured comparison against a commercial product,
    with the number that is statistically defensible rather than the flattering one.
    """
    from PIL import Image, ImageDraw, ImageFont
    scale = 2
    w, h = width * scale, height * scale
    img = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(img)

    def font(px: int, bold: bool = True):
        names = (("segoeuisb.ttf", "seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")
                 if bold else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"))
        for name in names:
            try:
                return ImageFont.truetype(name, px)
            except OSError:
                continue
        return ImageFont.load_default()

    icon_size = int(h * 0.30)
    icon = draw_logo(icon_size, transparent=True)
    left = int(w * 0.075)
    img.paste(icon, (left, int(h * 0.11)), icon)

    tx = left + icon_size + int(w * 0.035)
    d.text((tx, int(h * 0.145)), "drum2midi", font=font(int(h * 0.125)), fill=INK)
    d.text((tx, int(h * 0.285)), "measured, not guessed",
           font=font(int(h * 0.050), bold=False), fill=WAVE)

    d.text((left, int(h * 0.46)), "Drum recordings to General MIDI,",
           font=font(int(h * 0.062), bold=False), fill=INK)
    d.text((left, int(h * 0.545)), "using only open-source models",
           font=font(int(h * 0.062), bold=False), fill=INK)

    # the comparison, as a small scoreboard rather than a sentence
    bar_y = int(h * 0.71)
    d.line([left, bar_y - int(h * 0.035), w - left, bar_y - int(h * 0.035)],
           fill=(215, 219, 224), width=scale * 2)

    cells = [("0.882", "drum2midi", WAVE),
             ("0.820", "ReStem 2 Pro, $199", DIM)]
    x = left
    for value, label, colour in cells:
        d.text((x, bar_y), value, font=font(int(h * 0.105)), fill=colour)
        d.text((x, bar_y + int(h * 0.125)), label,
               font=font(int(h * 0.040), bold=False), fill=DIM)
        x += int(w * 0.26)

    d.text((x + int(w * 0.02), bar_y + int(h * 0.015)),
           "onset F1 on 23 hand-annotated",
           font=font(int(h * 0.040), bold=False), fill=DIM)
    d.text((x + int(w * 0.02), bar_y + int(h * 0.070)),
           "recordings, 95% CI [+0.033, +0.097]",
           font=font(int(h * 0.040), bold=False), fill=DIM)

    return img.resize((width, height), Image.LANCZOS)


def draw_small(size: int) -> Image.Image:
    """A simplified mark for 32 pixels and below.

    The full logo has a waveform, an arrow, a grid and nine note blocks. At 16 or 24
    pixels those collapse into noise, so the small sizes keep only the part that still
    reads at a glance: four coloured note blocks, in the kit's own colours.
    """
    from PIL import Image, ImageDraw
    scale = 8
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    pad = int(s * 0.08)
    d.rounded_rectangle([pad, pad, s - pad, s - pad], radius=int(s * 0.22),
                        fill=(255, 255, 255, 255), outline=INK + (255,),
                        width=max(2, int(s * 0.05)))

    # two rows of two blocks: crash, hi-hat / snare, kick
    inner = int(s * 0.22)
    gap = int(s * 0.06)
    w = (s - 2 * inner - gap) // 2
    h = int(w * 0.62)
    top = (s - 2 * h - gap) // 2
    for i, pitch in enumerate((49, 42, 38, 36)):
        x = inner + (i % 2) * (w + gap)
        y = top + (i // 2) * (h + gap)
        d.rounded_rectangle([x, y, x + w, y + h], radius=h // 3,
                            fill=_rgb(colour(pitch)) + (255,))
    return img.resize((size, size), Image.LANCZOS)


def write_ico(path: Path, sizes=(16, 24, 32, 48, 64, 128, 256)) -> None:
    """Writes an .ico with BMP entries for the small sizes and PNG for 256.

    Windows only understands PNG-compressed entries at 256x256. The taskbar asks for a
    24 or 32 pixel icon, and if those are PNG it silently falls back to the host
    executable's icon -- which is why a pythonw script keeps showing the Python logo.
    Pillow writes every entry as PNG for RGBA input and cannot mix the two, so the file
    is assembled here: correct for the shell, and a tenth of the size of all-BMP.
    """
    import io
    import struct

    entries = []
    for size in sorted(sizes):
        img = draw_small(size) if size <= 32 else draw_logo(size, transparent=True)
        buf = io.BytesIO()
        if size >= 256:
            img.save(buf, format="PNG")
        else:
            # An icon's BMP entry stores the height doubled (colour + mask) and omits
            # the 14-byte file header.
            tmp = io.BytesIO()
            img.save(tmp, format="BMP")
            raw = tmp.getvalue()[14:]
            header = bytearray(raw[:40])
            struct.pack_into("<i", header, 8, size * 2)
            mask_row = ((size + 31) // 32) * 4
            buf.write(bytes(header) + raw[40:] + b"\x00" * (mask_row * size))
        entries.append((size, buf.getvalue()))

    out = io.BytesIO()
    out.write(struct.pack("<HHH", 0, 1, len(entries)))
    offset = 6 + 16 * len(entries)
    for size, data in entries:
        dim = 0 if size >= 256 else size
        out.write(struct.pack("<BBBBHHII", dim, dim, 0, 0, 1, 32, len(data), offset))
        offset += len(data)
    for _, data in entries:
        out.write(data)
    path.write_bytes(out.getvalue())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", type=int, default=None,
                    help="write a single PNG of this size instead of the full set")
    ap.add_argument("--out", default=str(DOCS))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.size:
        path = out / f"logo_{args.size}.png"
        draw_logo(args.size).save(path)
        print(f"wrote {path}")
        return 0

    made = []
    for size in (256, 128, 64):
        path = out / f"logo_{size}.png"
        draw_logo(size).save(path)
        made.append(path)

    icon_path = out / "drum2midi.ico"
    write_ico(icon_path)
    made.append(icon_path)

    banner = out / "banner.png"
    draw_banner().save(banner)
    made.append(banner)

    social = out / "social.png"
    draw_social().save(social)
    made.append(social)

    for p in made:
        print(f"wrote {p}  ({p.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
