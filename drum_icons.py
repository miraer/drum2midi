"""One palette and one set of icons for the kit, shared by the logo and the GUI.

Colour carries meaning here rather than decoration: the same drum is the same colour in
the logo, in the results table and in any future piano roll, so a glance is enough to
tell which instrument a row or a block belongs to. Hues follow the kit from low to high
-- kick deep indigo, snare red, toms green, hi-hat amber, cymbals violet and teal --
which keeps neighbouring drums distinguishable even for the common forms of colour
blindness, where red/green pairs are the risk and lightness differences still read.

Icons are drawn, not shipped as files, so they can be produced at whatever size the
display needs.

    python drum_icons.py            # write a preview sheet to docs/
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Internal pitch -> (name, colour). Kick is 35 internally and written as 36, so both
# map to the same entry.
KIT = {
    35: ("Kick", "#3f51b5"),
    36: ("Kick", "#3f51b5"),
    38: ("Snare", "#e53935"),
    40: ("Snare rim", "#ef6c00"),
    37: ("Side stick", "#ef6c00"),
    41: ("Floor tom", "#2e7d32"),
    43: ("Tom low", "#2e7d32"),
    45: ("Tom low", "#388e3c"),
    47: ("Tom mid", "#43a047"),
    48: ("Tom mid", "#43a047"),
    50: ("Tom high", "#66bb6a"),
    42: ("Hi-hat closed", "#f9a825"),
    44: ("Hi-hat pedal", "#c17900"),
    46: ("Hi-hat open", "#ffca28"),
    49: ("Crash", "#8e24aa"),
    57: ("Crash", "#8e24aa"),
    55: ("Splash", "#ab47bc"),
    51: ("Ride", "#00897b"),
    59: ("Ride", "#00897b"),
    53: ("Ride bell", "#00695c"),
    60: ("Other", "#757575"),
}

# Which drawing to use for a pitch.
SHAPE = {
    35: "kick", 36: "kick",
    38: "snare", 40: "snare", 37: "snare",
    41: "tom", 43: "tom", 45: "tom", 47: "tom", 48: "tom", 50: "tom",
    42: "hihat", 44: "hihat_pedal", 46: "hihat_open",
    49: "cymbal", 57: "cymbal", 55: "cymbal",
    51: "ride", 59: "ride", 53: "ride",
    60: "other",
}


def colour(pitch: int) -> str:
    return KIT.get(pitch, ("?", "#9e9e9e"))[1]


def name(pitch: int) -> str:
    return KIT.get(pitch, (f"note {pitch}", ""))[0]


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def draw_icon(pitch: int, size: int = 18):
    """A small PIL image of the drum, drawn at 4x and downscaled for clean edges."""
    from PIL import Image, ImageDraw

    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    col = _rgb(colour(pitch)) + (255,)
    pale = _rgb(colour(pitch)) + (70,)
    line = max(2, int(s * 0.07))
    shape = SHAPE.get(pitch, "other")
    m = int(s * 0.10)

    if shape == "kick":
        # a bass drum head seen from the front: big circle with a beater spot
        d.ellipse([m, m, s - m, s - m], outline=col, width=line, fill=pale)
        r = int(s * 0.11)
        c = s // 2
        d.ellipse([c - r, c - r, c + r, c + r], fill=col)

    elif shape == "snare":
        # shell with snare wires across the bottom
        d.ellipse([m, m, s - m, s - m], outline=col, width=line, fill=pale)
        for i in range(3):
            y = int(s * (0.42 + i * 0.13))
            d.line([int(s * 0.26), y, int(s * 0.74), y], fill=col, width=max(1, line // 2))

    elif shape == "tom":
        # deeper shell: a circle with a rim band
        d.ellipse([m, m, s - m, s - m], outline=col, width=line, fill=pale)
        inset = int(s * 0.22)
        d.ellipse([inset, inset, s - inset, s - inset], outline=col,
                  width=max(1, line // 2))

    elif shape in ("hihat", "hihat_open", "hihat_pedal"):
        # two cymbals facing each other; the gap is the articulation, so it has to be
        # the most visible thing in the drawing
        gap = {"hihat": 0.14, "hihat_open": 0.34, "hihat_pedal": 0.03}[shape]
        mid = s / 2
        top = mid - s * gap / 2
        bot = mid + s * gap / 2
        w = int(s * 0.44)
        d.line([mid, top - s * 0.16, mid, bot + s * 0.30], fill=col,
               width=max(1, line // 2))
        for y in (top, bot):
            d.line([mid - w, y, mid + w, y], fill=col, width=line)
        if shape == "hihat_pedal":
            # a foot mark, since a closed pair alone would look like the closed icon
            d.line([mid - w * 0.5, bot + s * 0.30, mid + w * 0.5, bot + s * 0.30],
                   fill=col, width=line)

    elif shape == "cymbal":
        # a crash seen edge-on: a wide shallow ellipse on a stand
        d.ellipse([int(s * 0.04), int(s * 0.26), s - int(s * 0.04), int(s * 0.56)],
                  outline=col, width=line, fill=pale)
        d.line([s // 2, int(s * 0.44), s // 2, s - m], fill=col, width=max(1, line // 2))

    elif shape == "ride":
        # like a crash but with the bell marked, which is what distinguishes it
        d.ellipse([int(s * 0.04), int(s * 0.26), s - int(s * 0.04), int(s * 0.56)],
                  outline=col, width=line, fill=pale)
        r = int(s * 0.10)
        d.ellipse([s // 2 - r, int(s * 0.31), s // 2 + r, int(s * 0.31) + 2 * r], fill=col)
        d.line([s // 2, int(s * 0.50), s // 2, s - m], fill=col, width=max(1, line // 2))

    else:
        d.rounded_rectangle([m, int(s * 0.28), s - m, int(s * 0.72)],
                            radius=int(s * 0.12), outline=col, width=line, fill=pale)

    return img.resize((size, size), Image.LANCZOS)


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from PIL import Image, ImageDraw

    order = [36, 38, 43, 47, 50, 42, 46, 44, 49, 51, 60]
    cell, icon = 84, 40
    sheet = Image.new("RGB", (cell * len(order), cell + 26), (250, 251, 252))
    d = ImageDraw.Draw(sheet)
    for i, pitch in enumerate(order):
        img = draw_icon(pitch, icon)
        sheet.paste(img, (i * cell + (cell - icon) // 2, 16), img)
        d.text((i * cell + 6, cell + 4), name(pitch)[:12], fill=(60, 66, 76))

    (ROOT / "docs").mkdir(exist_ok=True)
    out = ROOT / "docs" / "kit_icons.png"
    sheet.save(out)
    print(f"wrote {out}")
    for pitch in order:
        print(f"  {pitch:>3}  {name(pitch):<16}{colour(pitch)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
