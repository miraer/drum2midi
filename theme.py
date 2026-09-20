"""The project's own ttk theme, drawn in code from the logo's palette.

Why not a stock theme: `vista` and `xpnative` draw through native Windows APIs, so they
ignore almost every colour you set and look nothing like the logo. `clam` accepts
colours but draws flat rectangles with 1px borders.

So the widget parts are generated as PNGs with Pillow and registered as ttk image
elements, which is the only way to get rounded corners and a real focus ring out of
Tk. Images are handed to Tk as base64 rather than files: Tk 8.6 reads PNG natively, so
the theme needs nothing on disk and survives being packaged.

Falls back to plain `clam` with the same colours when Pillow is missing, so the GUI
still opens.

Run this file directly to render a swatch of every widget:

    python theme.py
"""

from __future__ import annotations

import base64
import colorsys
import io
import sys
import tkinter as tk
from tkinter import ttk

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = ImageDraw = None


# The logo's colours: make_logo.INK, WAVE, DIM, PAPER, kept in sync by test_smoke.
class Light:
    name = "drum2midi"
    base = "#fafbfc"      # window background (PAPER)
    card = "#ffffff"      # raised surfaces: group boxes, tabs, entries
    ink = "#181c22"       # primary text (INK)
    dim = "#6b7280"       # secondary text
    line = "#d7dbe0"      # borders
    accent = "#3884de"    # the waveform blue (WAVE)
    accent_dark = "#2b6fc0"
    accent_soft = "#e8f1fc"
    ok = "#2e7d32"        # the floor tom green, reused for success
    warn = "#ef6c00"      # the snare rim orange
    disabled = "#a8aeb8"
    field = "#ffffff"


class Dark:
    name = "drum2midi-dark"
    base = "#181c22"      # INK becomes the background
    card = "#21262e"
    ink = "#e8eaed"
    dim = "#9aa2ad"
    line = "#333a44"
    accent = "#4d9bf0"
    accent_dark = "#3884de"
    accent_soft = "#1d2a3a"
    ok = "#66bb6a"
    warn = "#ffa726"
    disabled = "#5c636e"
    field = "#12161b"


PALETTES = {"light": Light, "dark": Dark}

# PhotoImages are referenced only by Tcl once registered, so Python would garbage
# collect them and the widgets would render blank. Keep them alive here.
_KEEP: list[tk.PhotoImage] = []


def _png(img) -> tk.PhotoImage:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    photo = tk.PhotoImage(data=base64.b64encode(buf.getvalue()))
    _KEEP.append(photo)
    return photo


def _rounded(w: int, h: int, radius: int, fill, outline: str | None = None,
             width: int = 1, scale: int = 4):
    """A rounded rectangle, supersampled so the corners are not jagged.

    Tk cannot antialias, so it is done here by drawing large and shrinking.
    """
    img = Image.new("RGBA", (w * scale, h * scale), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    inset = (width * scale) / 2
    d.rounded_rectangle([inset, inset, w * scale - 1 - inset, h * scale - 1 - inset],
                        radius=radius * scale, fill=fill,
                        outline=outline, width=width * scale if outline else 0)
    return img.resize((w, h), Image.LANCZOS)


def _slab(name: str, style: ttk.Style, states: dict, radius: int = 6,
          size: int = 24, border: int = 8) -> None:
    """Register one image element whose middle stretches: a button, tab or field.

    `border` tells Tk which edges are fixed; anything inside is tiled, so a single
    small PNG serves a button of any width.
    """
    default = states.pop("")
    spec = []
    for state, img in states.items():
        spec.append((state, _png(img)))
    style.element_create(name, "image", _png(default), *spec,
                         border=border, sticky="nsew", padding=0)


def _dot(size: int, palette, fill: str, outline: str, mark: str | None = None,
         round_: bool = True, scale: int = 4):
    """A checkbox or radio indicator: a small square or circle, optionally ticked."""
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if round_:
        d.ellipse([scale, scale, s - scale - 1, s - scale - 1], fill=fill,
                  outline=outline, width=scale)
    else:
        d.rounded_rectangle([scale, scale, s - scale - 1, s - scale - 1],
                            radius=3 * scale, fill=fill, outline=outline, width=scale)
    if mark == "check":
        # a tick, drawn as two strokes
        d.line([s * 0.28, s * 0.52, s * 0.44, s * 0.68], fill=palette.card,
               width=int(scale * 1.8))
        d.line([s * 0.43, s * 0.68, s * 0.73, s * 0.34], fill=palette.card,
               width=int(scale * 1.8))
    elif mark == "dot":
        d.ellipse([s * 0.32, s * 0.32, s * 0.68, s * 0.68], fill=palette.card)
    return img.resize((size, size), Image.LANCZOS)


def _build_images(style: ttk.Style, p) -> None:
    """Create every image element the theme needs. Only called when Pillow is there."""
    pre = p.name

    # --- buttons -----------------------------------------------------------------
    h, r = 30, 7
    _slab(f"{pre}.button", style, {
        "": _rounded(40, h, r, p.card, p.line),
        "disabled": _rounded(40, h, r, p.base, p.line),
        "pressed": _rounded(40, h, r, p.accent_soft, p.accent),
        "active": _rounded(40, h, r, p.card, p.accent),
    }, border=10)

    # the primary action reads as a solid block of the logo's blue
    face = accent_face(p)
    down = _shade(face, 0.82)
    _slab(f"{pre}.accentbutton", style, {
        "": _rounded(40, h, r, face, down),
        "disabled": _rounded(40, h, r, p.disabled, p.disabled),
        "pressed": _rounded(40, h, r, down, down),
        "active": _rounded(40, h, r, down, down),
    }, border=10)

    # --- text fields --------------------------------------------------------------
    _slab(f"{pre}.field", style, {
        "": _rounded(40, 28, 6, p.field, p.line),
        "disabled": _rounded(40, 28, 6, p.base, p.line),
        "focus": _rounded(40, 28, 6, p.field, p.accent, width=2),
        "hover": _rounded(40, 28, 6, p.field, p.dim),
    }, border=9)

    # --- notebook tabs ------------------------------------------------------------
    tab_h = 30
    img_sel = Image.new("RGBA", (40 * 4, tab_h * 4), (0, 0, 0, 0))
    d = ImageDraw.Draw(img_sel)
    d.rounded_rectangle([0, 0, 40 * 4 - 1, tab_h * 4 + 40], radius=7 * 4,
                        fill=p.card, outline=p.line, width=4)
    # the selected tab carries a blue underline, like the logo's waveform
    d.rectangle([3 * 4, tab_h * 4 - 3 * 4, 40 * 4 - 3 * 4, tab_h * 4], fill=p.accent)
    _slab(f"{pre}.tab", style, {
        "": Image.new("RGBA", (40, tab_h), (0, 0, 0, 0)),
        "active": _rounded(40, tab_h, 7, p.accent_soft),
        "selected": img_sel.resize((40, tab_h), Image.LANCZOS),
    }, border=10)

    # --- indicators ---------------------------------------------------------------
    n = 17
    style.element_create(
        f"{pre}.checkindicator", "image",
        _png(_dot(n, p, p.field, p.line, round_=False)),
        ("disabled", "selected", _png(_dot(n, p, p.disabled, p.disabled, "check", False))),
        ("disabled", _png(_dot(n, p, p.base, p.line, round_=False))),
        ("selected", "pressed", _png(_dot(n, p, p.accent_dark, p.accent_dark, "check", False))),
        ("selected", _png(_dot(n, p, p.accent, p.accent_dark, "check", False))),
        ("active", _png(_dot(n, p, p.field, p.accent, round_=False))),
        padding=0, sticky="")
    style.element_create(
        f"{pre}.radioindicator", "image",
        _png(_dot(n, p, p.field, p.line)),
        ("disabled", "selected", _png(_dot(n, p, p.disabled, p.disabled, "dot"))),
        ("disabled", _png(_dot(n, p, p.base, p.line))),
        ("selected", "pressed", _png(_dot(n, p, p.accent_dark, p.accent_dark, "dot"))),
        ("selected", _png(_dot(n, p, p.accent, p.accent_dark, "dot"))),
        ("active", _png(_dot(n, p, p.field, p.accent))),
        padding=0, sticky="")

    # --- progress bar -------------------------------------------------------------
    _slab(f"{pre}.trough", style, {"": _rounded(30, 12, 6, p.base, p.line)}, border=7)
    _slab(f"{pre}.pbar", style, {"": _rounded(30, 12, 6, p.accent, p.accent_dark)},
          border=7)

    # --- group boxes ---------------------------------------------------------------
    # clam draws a Labelframe border by alternating lightcolor and darkcolor, which
    # reads as a dotted line. A transparent-centred image gives a clean rounded rule
    # and leaves the background to the frame, so children still match.
    _slab(f"{pre}.groupbox", style,
          {"": _rounded(40, 40, 8, (0, 0, 0, 0), p.line)}, border=10)

    # --- spinbox and combobox arrows ---------------------------------------------
    def arrow(direction: str, colour: str, scale: int = 4):
        s = 14 * scale
        img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        cx, cy, w2, h2 = s / 2, s / 2, s * 0.22, s * 0.12
        pts = {"up": [(cx - w2, cy + h2), (cx + w2, cy + h2), (cx, cy - h2)],
               "down": [(cx - w2, cy - h2), (cx + w2, cy - h2), (cx, cy + h2)]}[direction]
        d.polygon(pts, fill=colour)
        return img.resize((14, 14), Image.LANCZOS)

    for d_ in ("up", "down"):
        style.element_create(f"{pre}.{d_}arrow", "image", _png(arrow(d_, p.dim)),
                             ("disabled", _png(arrow(d_, p.disabled))),
                             ("pressed", _png(arrow(d_, p.accent))),
                             ("active", _png(arrow(d_, p.ink))),
                             padding=0, sticky="")


def _layouts(style: ttk.Style, p) -> None:
    """Point each widget at the image elements. Without this they keep clam's parts."""
    pre = p.name
    style.layout("TButton", [
        (f"{pre}.button", {"sticky": "nsew", "children": [
            ("Button.padding", {"sticky": "nsew", "children": [
                ("Button.label", {"sticky": "nsew"})]})]})])
    style.layout("Accent.TButton", [
        (f"{pre}.accentbutton", {"sticky": "nsew", "children": [
            ("Button.padding", {"sticky": "nsew", "children": [
                ("Button.label", {"sticky": "nsew"})]})]})])
    style.layout("TEntry", [
        (f"{pre}.field", {"sticky": "nsew", "children": [
            ("Entry.padding", {"sticky": "nsew", "children": [
                ("Entry.textarea", {"sticky": "nsew"})]})]})])
    style.layout("TSpinbox", [
        (f"{pre}.field", {"sticky": "nsew", "children": [
            ("null", {"side": "right", "sticky": "ns", "children": [
                (f"{pre}.uparrow", {"side": "top", "sticky": "e"}),
                (f"{pre}.downarrow", {"side": "bottom", "sticky": "e"})]}),
            ("Spinbox.padding", {"sticky": "nsew", "children": [
                ("Spinbox.textarea", {"sticky": "nsew"})]})]})])
    style.layout("TCombobox", [
        (f"{pre}.field", {"sticky": "nsew", "children": [
            (f"{pre}.downarrow", {"side": "right", "sticky": "e"}),
            ("Combobox.padding", {"sticky": "nsew", "children": [
                ("Combobox.textarea", {"sticky": "nsew"})]})]})])
    style.layout("TNotebook.Tab", [
        (f"{pre}.tab", {"sticky": "nsew", "children": [
            ("Notebook.padding", {"side": "top", "sticky": "nsew", "children": [
                ("Notebook.label", {"side": "top", "sticky": ""})]})]})])
    style.layout("TCheckbutton", [
        ("Checkbutton.padding", {"sticky": "nsew", "children": [
            (f"{pre}.checkindicator", {"side": "left", "sticky": ""}),
            ("Checkbutton.focus", {"side": "left", "sticky": "w", "children": [
                ("Checkbutton.label", {"sticky": "nsew"})]})]})])
    style.layout("TRadiobutton", [
        ("Radiobutton.padding", {"sticky": "nsew", "children": [
            (f"{pre}.radioindicator", {"side": "left", "sticky": ""}),
            ("Radiobutton.focus", {"side": "left", "sticky": "w", "children": [
                ("Radiobutton.label", {"sticky": "nsew"})]})]})])
    style.layout("Horizontal.TProgressbar", [
        (f"{pre}.trough", {"sticky": "nsew", "children": [
            (f"{pre}.pbar", {"side": "left", "sticky": "ns"})]})])
    style.layout("TLabelframe", [(f"{pre}.groupbox", {"sticky": "nsew"})])


def _colours(style: ttk.Style, p, drawn: bool = True) -> None:
    """Colours for everything, including the parts that stay as clam drawings.

    `drawn` says whether the image elements were installed. It matters for exactly one
    widget: the accent button paints white text, which is only legible on the accent
    image behind it. Without Pillow there is no image, clam draws its ordinary grey
    button, and white-on-grey made the Convert button vanish completely on a machine
    that had everything else working.
    """
    style.configure(".", background=p.base, foreground=p.ink, fieldbackground=p.field,
                    bordercolor=p.line, lightcolor=p.base, darkcolor=p.base,
                    troughcolor=p.base, focuscolor=p.accent, insertcolor=p.ink)
    style.map(".", foreground=[("disabled", p.disabled)])

    style.configure("TFrame", background=p.base)
    style.configure("TLabel", background=p.base, foreground=p.ink)
    style.configure("Dim.TLabel", foreground=p.dim)
    style.configure("Ok.TLabel", foreground=p.ok)
    style.configure("Warn.TLabel", foreground=p.warn)
    style.configure("Accent.TLabel", foreground=p.accent)
    style.configure("Title.TLabel", foreground=p.ink)

    style.configure("TButton", padding=(14, 5), anchor="center", background=p.base)
    style.map("TButton", foreground=[("disabled", p.disabled), ("active", p.accent)])
    style.configure("Accent.TButton", foreground="#ffffff")
    style.map("Accent.TButton",
              foreground=[("disabled", p.base), ("active", "#ffffff"),
                          ("pressed", "#ffffff")])
    if not drawn:
        # No image element behind it, so clam's own button face has to carry the
        # accent colour or the white label is invisible.
        face = accent_face(p)
        down = _shade(face, 0.82)
        style.configure("Accent.TButton", background=face,
                        lightcolor=face, darkcolor=face, bordercolor=down)
        style.map("Accent.TButton",
                  background=[("disabled", p.base), ("pressed", down),
                              ("active", down)],
                  foreground=[("disabled", p.disabled), ("active", "#ffffff"),
                              ("pressed", "#ffffff")])

    style.configure("TCheckbutton", background=p.base, padding=(0, 3))
    style.configure("TRadiobutton", background=p.base, padding=(0, 3))
    style.map("TCheckbutton", foreground=[("disabled", p.disabled)])
    style.map("TRadiobutton", foreground=[("disabled", p.disabled)])

    style.configure("TEntry", padding=(7, 4), foreground=p.ink,
                    fieldbackground=p.field, selectbackground=p.accent,
                    selectforeground="#ffffff")
    style.configure("TSpinbox", padding=(7, 4), arrowsize=12)
    style.configure("TCombobox", padding=(7, 4), arrowsize=12)
    style.map("TCombobox", fieldbackground=[("readonly", p.field)],
              foreground=[("disabled", p.disabled)])

    style.configure("TLabelframe", background=p.base, bordercolor=p.line,
                    lightcolor=p.line, darkcolor=p.line, relief="solid",
                    borderwidth=1, padding=10, labelmargins=(10, 0, 0, 0))
    style.configure("TLabelframe.Label", background=p.base, foreground=p.dim,
                    font=("Segoe UI Semibold", 9))

    style.configure("TNotebook", background=p.base, borderwidth=0, tabmargins=(0, 4, 0, 0))
    style.configure("TNotebook.Tab", padding=(16, 6), background=p.base, foreground=p.dim)
    style.map("TNotebook.Tab", foreground=[("selected", p.ink), ("active", p.ink)])

    style.configure("Treeview", background=p.base, fieldbackground=p.base,
                    foreground=p.ink, borderwidth=0, rowheight=24)
    style.configure("Treeview.Heading", background=p.base, foreground=p.dim,
                    relief="flat", padding=(6, 4))
    style.map("Treeview", background=[("selected", p.accent)],
              foreground=[("selected", "#ffffff")])
    style.map("Treeview.Heading", background=[("active", p.accent_soft)])

    style.configure("TSeparator", background=p.line)
    style.configure("Horizontal.TProgressbar", background=p.accent, troughcolor=p.base)
    style.configure("TScrollbar", background=p.base, troughcolor=p.base,
                    bordercolor=p.line, arrowcolor=p.dim, relief="flat")
    style.map("TScrollbar", background=[("active", p.line)])


def apply(root: tk.Misc, mode: str = "light") -> type:
    """Install and select the theme. Returns the palette so callers can match colours.

    Safe to call when Pillow is missing: the colours still apply and the accent button
    falls back to a painted background, only the rounded corners are lost.
    """
    p = PALETTES[mode]
    style = ttk.Style(root)
    if p.name not in style.theme_names():
        style.theme_create(p.name, parent="clam")
    style.theme_use(p.name)
    drawn = False
    if Image is not None:
        try:
            _build_images(style, p)
            _layouts(style, p)
            drawn = True
        except tk.TclError:
            # elements already registered by an earlier call, so they are there
            drawn = True
    _colours(style, p, drawn)

    # classic tk widgets (Text, Canvas, menus) do not read ttk styles
    root.tk_setPalette(background=p.base, foreground=p.ink,
                       selectBackground=p.accent, selectForeground="#ffffff")
    try:
        root.configure(background=p.base)
    except tk.TclError:
        pass
    return p


def text_options(p) -> dict:
    """Colours for a tk.Text or tk.Listbox so it matches the theme."""
    return {"background": p.card, "foreground": p.ink, "insertbackground": p.ink,
            "selectbackground": p.accent, "selectforeground": "#ffffff",
            "highlightthickness": 1, "highlightbackground": p.line,
            "highlightcolor": p.accent, "borderwidth": 0, "relief": "flat"}


def readable(colour: str, p) -> str:
    """Lift an instrument colour until it is legible on this theme's background.

    The kit colours were picked for white; on the dark background the kick's indigo
    and the crash's purple all but vanish. Raising lightness in HLS keeps each drum
    recognisably its own hue instead of remapping to a second hand-picked palette.
    Saturation is capped on the way: the ride's teal is fully saturated already, and
    lightening it alone turns it into neon.
    """
    if p is Light or not colour.startswith("#") or len(colour) != 7:
        return colour
    r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, max(l, 0.60), min(s, 0.60))
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def _shade(colour: str, factor: float) -> str:
    """Same hue, scaled lightness. Used to derive a pressed state from a face colour."""
    r, g, b = (int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l * factor)), s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def accent_face(p) -> str:
    """The shade the primary button is painted, so its white label stays legible.

    The dark palette's accent is a lighter blue, and white on it measures 2.89 against
    the 3.0 a UI component needs -- the Convert button was the least readable control
    in the window while also being the most important one. Derived rather than
    hard-coded so a change to either blue cannot quietly reintroduce it.
    """
    return p.accent if contrast("#ffffff", p.accent) >= 3.0 else p.accent_dark


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two "#rrggbb" colours, 1.0 (same) to 21.0.

    Exists because a test that asked only whether two colours *differed* let the
    invisible Convert button through: #ffffff on #fafbfc are different strings and
    the same colour to a human eye. 3.0 is the WCAG threshold for interface
    components and large text.
    """
    def lum(c: str) -> float:
        if not c.startswith("#") or len(c) != 7:
            raise ValueError(f"not a #rrggbb colour: {c!r}")
        out = []
        for i in (1, 3, 5):
            v = int(c[i:i + 2], 16) / 255
            out.append(v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4)
        return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]

    la, lb = lum(a), lum(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _demo() -> None:
    root = tk.Tk()
    root.title("drum2midi theme")
    mode = {"v": "light"}
    holder = ttk.Frame(root)
    holder.pack(fill="both", expand=True)

    def build() -> None:
        for w in holder.winfo_children():
            w.destroy()
        p = apply(root, mode["v"])
        f = ttk.Frame(holder, padding=16)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text="drum recordings to General MIDI",
                  font=("Segoe UI", 13, "bold")).pack(anchor="w")
        ttk.Label(f, text="measured, not guessed", style="Dim.TLabel").pack(anchor="w")
        box = ttk.Labelframe(f, text="Separator")
        box.pack(fill="x", pady=10)
        v = tk.StringVar(value="uvr")
        for val, lab in (("uvr", "MDX23C — best quality"), ("larsnet", "LarsNet — fast")):
            ttk.Radiobutton(box, text=lab, value=val, variable=v).pack(anchor="w")
        ttk.Checkbutton(box, text="fuse onsets found in stems").pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=6)
        ttk.Entry(row).pack(side="left", fill="x", expand=True)
        ttk.Spinbox(row, from_=0, to=127, width=5).pack(side="left", padx=6)
        ttk.Combobox(row, values=["1/8", "1/16"], width=8).pack(side="left")
        bar = ttk.Progressbar(f, value=64)
        bar.pack(fill="x", pady=8)
        btns = ttk.Frame(f)
        btns.pack(fill="x")
        ttk.Button(btns, text="Convert", style="Accent.TButton").pack(side="left")
        ttk.Button(btns, text="Stop").pack(side="left", padx=6)
        ttk.Button(btns, text="Disabled", state="disabled").pack(side="left")

        def flip() -> None:
            mode["v"] = "dark" if mode["v"] == "light" else "light"
            build()

        ttk.Button(btns, text="Toggle dark", command=flip).pack(side="right")
        nb = ttk.Notebook(f)
        nb.pack(fill="both", expand=True, pady=(10, 0))
        for name in ("Result", "Log", "Channels & notes"):
            t = ttk.Frame(nb, padding=8)
            ttk.Label(t, text=f"{name} goes here", style="Dim.TLabel").pack(anchor="w")
            nb.add(t, text=name)

    build()
    root.geometry("560x520")
    root.mainloop()


if __name__ == "__main__":
    _demo()
