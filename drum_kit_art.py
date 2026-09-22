"""Draws a drum kit in three-quarter view, for use as a faint background in the GUI.

A photo would drag a licence along with it, and the project already draws its logo,
its icons and its architecture diagram in code, so the kit is drawn too. The piece
doubles as a legend: every shell and cymbal carries the same colour its icon and its
row in the results table use, so the picture explains the palette without a caption.

Seen from the front and slightly above, the way a listener sees a kit on a stage. A
plan view was tried first and read as an abstract pile of circles.

Everything is drawn opaque and the whole image is faded at the end. Fading each shape
as it is drawn makes the overlaps pile up and turns the near drums muddy.

    python drum_kit_art.py            # writes docs/kit_light.png and kit_dark.png
    python drum_kit_art.py --show     # opens a window with both
"""

from __future__ import annotations

import sys
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

from drum_icons import colour  # noqa: E402

HARDWARE = (150, 158, 170)
HARDWARE_DARK = (120, 130, 145)


def _rgb(pitch: int, dark: bool) -> tuple:
    c = tuple(int(colour(pitch)[i:i + 2], 16) for i in (1, 3, 5))
    if dark:
        # the kit colours were chosen for a white page; lift them or they vanish
        return tuple(min(255, int(v + (255 - v) * 0.40)) for v in c)
    return c


def _shade(c: tuple, f: float) -> tuple:
    return tuple(max(0, min(255, int(v * f))) for v in c)


class Kit:
    """Draws onto one image. Coordinates are fractions of the canvas, radii of width."""

    def __init__(self, w: int, h: int, dark: bool, colours=None):
        self.img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img)
        self.w, self.h, self.dark = w, h, dark
        self.colours = colours
        self.hw = HARDWARE_DARK if dark else HARDWARE
        self.line = max(1, int(w * 0.0035))

    def rgb(self, pitch: int) -> tuple:
        """The drum's colour: the caller's palette as given, else the icons' own.

        A caller's palette is taken as already fitted to its background, so it is not
        lifted for dark the way the icon colours are.
        """
        if self.colours is None:
            return _rgb(pitch, self.dark)
        h = self.colours(pitch)
        return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))

    def _px(self, x: float, y: float) -> tuple:
        return x * self.w, y * self.h

    def rod(self, x0, y0, x1, y1, width: float = 1.0) -> None:
        self.d.line([x0, y0, x1, y1], fill=self.hw + (255,),
                    width=max(1, int(self.line * 2.2 * width)))

    def shell(self, pitch: int, x: float, y: float, r: float, depth: float,
              legs: bool = False) -> None:
        """A tom or snare: the top head, the shell below it, and the rims.

        `y` is the centre of the top head, `depth` the shell height as a fraction of
        the canvas width.
        """
        cx, cy = self._px(x, y)
        rx = r * self.w
        ry = rx * 0.34          # how far the top head is foreshortened
        dep = depth * self.w
        base = self.rgb(pitch)
        side = _shade(base, 0.80 if not self.dark else 0.72)
        head = _shade(base, 1.14 if not self.dark else 1.0)

        if legs:
            for lx in (-0.78, 0.78):
                self.rod(cx + rx * lx, cy + ry * 0.4,
                         cx + rx * lx * 1.25, cy + dep + ry * 2.4)
        # the shell, capped by the near edge of the bottom head
        self.d.ellipse([cx - rx, cy + dep - ry, cx + rx, cy + dep + ry],
                       fill=side + (255,))
        self.d.rectangle([cx - rx, cy, cx + rx, cy + dep], fill=side + (255,))
        # lugs down the side: what makes a cylinder read as a drum
        for f in (-0.62, -0.21, 0.21, 0.62):
            lx = cx + rx * f
            self.d.line([lx, cy + dep * 0.16, lx, cy + dep * 0.84],
                        fill=_shade(side, 0.80) + (255,), width=self.line)
        self.d.arc([cx - rx, cy + dep - ry, cx + rx, cy + dep + ry], 0, 180,
                   fill=_shade(side, 0.72) + (255,), width=self.line * 2)
        # top head
        self.d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=head + (255,),
                       outline=_shade(base, 0.68) + (255,), width=self.line * 2)
        self.d.ellipse([cx - rx * 0.86, cy - ry * 0.86, cx + rx * 0.86, cy + ry * 0.86],
                       outline=_shade(base, 0.88) + (255,), width=self.line)

    def kick(self, pitch: int, x: float, y: float, r: float) -> None:
        """The bass drum, seen almost head on, with a ported front head."""
        cx, cy = self._px(x, y)
        rx = r * self.w
        ry = rx * 0.94
        base = self.rgb(pitch)
        for f in (-0.82, 0.82):  # spurs
            self.rod(cx + rx * f, cy + ry * 0.45, cx + rx * f * 1.20, cy + ry * 1.06, 1.3)
        self.d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                       fill=_shade(base, 1.10 if not self.dark else 0.95) + (255,),
                       outline=_shade(base, 0.66) + (255,), width=self.line * 3)
        self.d.ellipse([cx - rx * 0.90, cy - ry * 0.90, cx + rx * 0.90, cy + ry * 0.90],
                       outline=_shade(base, 0.82) + (255,), width=self.line)
        # the port hole, off to one side as it usually is
        self.d.ellipse([cx + rx * 0.14, cy + ry * 0.04, cx + rx * 0.60, cy + ry * 0.50],
                       fill=_shade(base, 0.52) + (255,),
                       outline=_shade(base, 0.70) + (255,), width=self.line)

    def cymbal(self, pitch: int, x: float, y: float, r: float, tilt: float = 0.17,
               floor: float | None = 0.97) -> None:
        """A cymbal on its stand, tilted towards the player."""
        cx, cy = self._px(x, y)
        rx = r * self.w
        ry = rx * tilt
        base = self.rgb(pitch)
        if floor is not None:
            self.rod(cx, cy, cx, floor * self.h, 1.0)
        # a sliver of the underside, so the disc has thickness
        self.d.ellipse([cx - rx, cy - ry + self.line * 2.5,
                        cx + rx, cy + ry + self.line * 2.5],
                       fill=_shade(base, 0.60) + (255,))
        self.d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry],
                       fill=_shade(base, 1.12 if not self.dark else 0.98) + (255,),
                       outline=_shade(base, 0.68) + (255,), width=self.line)
        for f in (0.74, 0.50):  # lathing grooves
            self.d.ellipse([cx - rx * f, cy - ry * f, cx + rx * f, cy + ry * f],
                           outline=_shade(base, 0.80) + (255,), width=self.line)
        self.d.ellipse([cx - rx * 0.15, cy - ry * 0.60, cx + rx * 0.15, cy + ry * 0.60],
                       fill=_shade(base, 0.66) + (255,))

    def hihat(self, pitch: int, x: float, y: float, r: float, floor: float) -> None:
        """Two cymbals a little apart, which is what tells a hi-hat from a crash."""
        cx, _ = self._px(x, y)
        gap = r * 0.34 * self.w / self.h
        self.cymbal(pitch, x, y + gap, r * 0.97, tilt=0.20, floor=floor)
        self.cymbal(pitch, x, y, r, tilt=0.20, floor=None)


def render(width: int = 300, height: int | None = None, dark: bool = False,
           alpha: int = 42, scale: int = 3, colours=None) -> Image.Image:
    """The kit as an RGBA image, faded to `alpha` so it can sit behind text.

    `colours` maps a pitch to "#rrggbb". The window passes its own theme tokens, so
    the picture stays the legend for the rows and lanes drawn in those tokens.
    """
    height = height or int(width * 0.76)
    k = Kit(width * scale, height * scale, dark, colours)

    # back to front, so the near drums overlap the far ones
    k.cymbal(49, 0.152, 0.145, 0.148, tilt=0.18, floor=0.97)   # crash, player's left
    k.cymbal(51, 0.852, 0.200, 0.146, tilt=0.15, floor=0.93)   # ride, over the floor tom
    k.shell(50, 0.408, 0.325, 0.098, 0.100)                    # rack tom, high
    k.shell(47, 0.598, 0.330, 0.106, 0.112)                    # rack tom, mid
    k.shell(43, 0.842, 0.505, 0.126, 0.225, legs=True)         # floor tom
    k.hihat(42, 0.143, 0.455, 0.116, floor=0.97)               # hi-hat
    k.shell(38, 0.238, 0.600, 0.119, 0.112, legs=True)         # snare
    k.kick(36, 0.503, 0.700, 0.206)                            # kick, nearest

    img = k.img.resize((width, height), Image.LANCZOS)
    # fade the finished picture rather than each shape, so overlaps stay clean
    img.putalpha(img.getchannel("A").point(lambda v: int(v * alpha / 255)))
    return img


def _on(bg_rgb: tuple, img: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", img.size, bg_rgb + (255,))
    bg.alpha_composite(img)
    return bg


def _main() -> None:
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    for dark in (False, True):
        # saved stronger than the GUI uses it, so it reads in the README
        img = render(720, dark=dark, alpha=95)
        p = out / f"kit_{'dark' if dark else 'light'}.png"
        _on((24, 28, 34) if dark else (250, 251, 252), img).convert("RGB").save(p)
        print("saved:", p, p.stat().st_size // 1024, "KB")

    if "--show" in sys.argv[1:]:
        import tkinter as tk
        from PIL import ImageTk
        root = tk.Tk()
        root.title("drum kit art")
        keep = []
        for i, dark in enumerate((False, True)):
            photo = ImageTk.PhotoImage(
                _on((24, 28, 34) if dark else (250, 251, 252),
                    render(440, dark=dark, alpha=95)))
            keep.append(photo)
            tk.Label(root, image=photo, borderwidth=0).grid(row=0, column=i)
        root.mainloop()


if __name__ == "__main__":
    _main()
