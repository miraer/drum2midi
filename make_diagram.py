"""Draws the architecture diagram as a PNG.

The README carries a Mermaid version, which GitHub renders natively, but that is
invisible in a plain editor, in a PDF or in a presentation. Drawing it from code keeps
the two in step and lets the instrument palette from drum_icons carry over, so the
diagram reads in the same visual language as the logo and the results table.

    python make_diagram.py
    python make_diagram.py --width 1600
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"

INK = (24, 28, 34)
MUTED = (110, 118, 130)
PAPER = (250, 251, 252)
GPU = (63, 81, 181)          # same indigo as the kick
CPU = (46, 125, 50)          # same green as the toms
LINE = (150, 158, 170)


def font(px: int, bold: bool = False):
    names = (("segoeuisb.ttf", "seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf")
             if bold else ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"))
    for name in names:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def box(d, xy, title, lines, accent=None, scale=1):
    x0, y0, x1, y1 = xy
    radius = int(14 * scale)
    d.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=(255, 255, 255),
                        outline=accent or LINE, width=int((3 if accent else 2) * scale))
    if accent:
        d.rounded_rectangle([x0, y0, x0 + int(8 * scale), y1], radius=radius,
                            fill=accent)
    tx = x0 + int(26 * scale)
    ty = y0 + int(16 * scale)
    d.text((tx, ty), title, font=font(int(21 * scale), True), fill=INK)
    ty += int(30 * scale)
    for line in lines:
        d.text((tx, ty), line, font=font(int(17 * scale)), fill=MUTED)
        ty += int(23 * scale)


def sub_box(d, xy, title, subtitle, scale=1, tint=None):
    """A small nested card, used to open up stage 3."""
    x0, y0, x1, y1 = xy
    d.rounded_rectangle([x0, y0, x1, y1], radius=int(10 * scale),
                        fill=tint or (247, 248, 250), outline=(214, 219, 227),
                        width=max(1, int(1.5 * scale)))
    d.text((x0 + int(14 * scale), y0 + int(11 * scale)), title,
           font=font(int(16 * scale), True), fill=INK)
    ty = y0 + int(33 * scale)
    for line in subtitle:
        d.text((x0 + int(14 * scale), ty), line, font=font(int(14 * scale)), fill=MUTED)
        ty += int(18 * scale)


def arrow(d, start, end, scale=1, dashed=False, label=None):
    x0, y0 = start
    x1, y1 = end
    width = int(3 * scale)
    if dashed:
        total = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        steps = max(int(total / (12 * scale)), 1)
        for i in range(steps):
            if i % 2:
                continue
            a = i / steps
            b = min((i + 1) / steps, 1.0)
            d.line([x0 + (x1 - x0) * a, y0 + (y1 - y0) * a,
                    x0 + (x1 - x0) * b, y0 + (y1 - y0) * b], fill=LINE, width=width)
    else:
        d.line([x0, y0, x1, y1], fill=LINE, width=width)

    head = int(10 * scale)
    if abs(x1 - x0) < abs(y1 - y0):          # mostly vertical
        d.polygon([(x1, y1), (x1 - head * 0.6, y1 - head), (x1 + head * 0.6, y1 - head)],
                  fill=LINE)
    else:
        d.polygon([(x1, y1), (x1 - head, y1 - head * 0.6), (x1 - head, y1 + head * 0.6)],
                  fill=LINE)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        d.text((mx + int(10 * scale), my - int(20 * scale)), label,
               font=font(int(15 * scale)), fill=MUTED)


def draw(width: int = 1200) -> Image.Image:
    scale = width / 1200
    w, h = width, int(980 * scale)
    img = Image.new("RGB", (w, h), PAPER)
    d = ImageDraw.Draw(img)

    def S(v):
        return int(v * scale)

    d.text((S(40), S(28)), "drum2midi — how a recording becomes MIDI",
           font=font(S(30), True), fill=INK)
    d.text((S(40), S(68)), "blue runs on the GPU, green on the CPU — chosen per stage "
                           "by measurement", font=font(S(18)), fill=MUTED)

    col_l, col_r = S(60), S(640)
    box_w = S(500)

    box(d, (col_l, S(120), col_l + box_w, S(196)), "audio in",
        ["a drum stem, or a whole song"], scale=scale)

    box(d, (col_l, S(238), col_l + box_w, S(330)), "0.  extract drums  ·  htdemucs",
        ["only with --from-song", "costs ride/crash accuracy, not toms"],
        accent=GPU, scale=scale)

    # everything downstream reads this, whether it came straight in or out of htdemucs
    box(d, (col_l, S(356), col_r + box_w, S(404)), "drum audio",
        ["the original stem, or what htdemucs extracted"], scale=scale)

    box(d, (col_l, S(450), col_l + box_w, S(556)), "1.  what and when  ·  ADTOF",
        ["5 onset classes at 100 fps", "reads the drum MIXTURE, never the stems"],
        accent=CPU, scale=scale)

    box(d, (col_r, S(450), col_r + box_w, S(556)), "2.  how loud  ·  MDX23C",
        ["6 stems, ride separate from crash", "12x faster on the GPU"],
        accent=GPU, scale=scale)

    box(d, (col_l, S(612), col_r + box_w, S(800)),
        "3.  articulation and velocity  ·  this repository, no external model",
        [], scale=scale)

    # stage 3 is where the pipeline's own decisions live, so it is worth opening up
    inner_y0, inner_y1 = S(662), S(752)
    cards = [
        ("split_toms", ["which tom was hit", "pitch order right 36/36"]),
        ("split_hats", ["open / closed / pedal", "from how long it rings"]),
        ("split_ride", ["ride or crash", "beat 3 learned models"]),
        ("velocity", ["formula for kick/snare", "learned for cymbals"]),
    ]
    total_w = (col_r + box_w) - col_l - S(56)
    card_w = (total_w - S(30) * (len(cards) - 1)) // len(cards)
    cx = col_l + S(28)
    for title, lines in cards:
        sub_box(d, (cx, inner_y0, cx + card_w, inner_y1), title, lines, scale=scale)
        cx += card_w + S(30)
    d.text((col_l + S(28), S(766)),
           "velocity is a hybrid because it was measured: the handcrafted dB formula "
           "beats gradient boosting on kick 0.843 vs 0.701",
           font=font(S(14)), fill=MUTED)

    box(d, (col_l, S(834), col_r + box_w, S(894)),
        "4.  export  ·  General MIDI, PPQ 960, detected tempo", [], scale=scale)

    mid_l = col_l + box_w // 2
    mid_r = col_r + box_w // 2
    arrow(d, (mid_l, S(196)), (mid_l, S(236)), scale, label="whole song")
    arrow(d, (mid_l, S(330)), (mid_l, S(354)), scale)
    arrow(d, (mid_l, S(404)), (mid_l, S(448)), scale)
    arrow(d, (mid_r, S(404)), (mid_r, S(448)), scale)
    arrow(d, (mid_l, S(556)), (mid_l, S(610)), scale, label="onsets")
    arrow(d, (mid_r, S(556)), (mid_r, S(610)), scale, label="audio per drum")
    arrow(d, (mid_l, S(800)), (mid_l, S(832)), scale)

    # a file that is already a drum stem skips htdemucs entirely
    arrow(d, (col_l + box_w, S(158)), (col_r + box_w - S(30), S(158)), scale,
          dashed=True, label="already a drum stem")
    arrow(d, (col_r + box_w - S(30), S(158)), (col_r + box_w - S(30), S(354)), scale,
          dashed=True)

    note = ("The separated stems never reach the onset detector. Measured: ADTOF on "
            "stems scores tom F1 0.000,\nand a CNN trained for it reaches 0.623 against "
            "the pipeline's 0.882.")
    d.text((S(60), S(924)), note, font=font(S(16)), fill=MUTED)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=1200)
    args = ap.parse_args()

    DOCS.mkdir(exist_ok=True)
    out = DOCS / "architecture.png"
    draw(args.width).save(out)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
