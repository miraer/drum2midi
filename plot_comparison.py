"""Draws the ReStem comparison as a difference plot with confidence intervals.

The README presents it as a table, which is right for reading and wrong for video: the
one thing a viewer needs to see instantly is that seven of nine rows cross zero, and a
table makes that a sentence rather than a picture.

So this plots the *difference* rather than the two scores. Each row is one class, the dot
is the point estimate, the bar is the 95% interval from resampling recordings, and the
vertical line at zero is the thing most of the bars touch. Rows whose interval excludes
zero are drawn in the project's ink; the rest are grey, because grey is what they are.

Numbers come from significance.py's output rather than being retyped, so the picture
cannot drift from the table it illustrates.

    python plot_comparison.py
    python plot_comparison.py --log bench/significance_ceiling.log --out docs/comparison.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent

# from the logo, the same palette the GUI uses
INK = "#181c22"
WAVE = "#3884de"
PAPER = "#fafbfc"
GREY = "#9aa4b0"

ROW = re.compile(
    r"^\s*(?P<name>MICRO|kick|snare|toms|hi-hat|cymbals)\s+"
    r"(?P<ours>[\d.]+)\s+(?P<theirs>[\d.]+)\s+(?P<diff>[+-][\d.]+)\s+"
    r"\[\s*(?P<lo>[+-][\d.]+),\s*(?P<hi>[+-][\d.]+)\s*\]\s+(?P<verdict>\S+)")

PRETTY = {"MICRO": "Overall", "kick": "Kick", "snare": "Snare", "toms": "Toms",
          "hi-hat": "Hi-hat", "cymbals": "Cymbals"}


def read_log(path: Path) -> str:
    """Read a log whatever PowerShell wrote it as.

    `Tee-Object` on Windows PowerShell 5.1 writes UTF-16LE with a BOM, while everything
    here assumes UTF-8. Reading that as UTF-8 with errors="replace" does not raise -- it
    silently yields text in which no line matches anything, so the failure arrives as
    "no rows found" rather than as an encoding error. Sniff the BOM instead.
    """
    raw = path.read_bytes()
    for bom, enc in ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"),
                     (b"\xef\xbb\xbf", "utf-8-sig")):
        if raw.startswith(bom):
            return raw.decode(enc)
    return raw.decode("utf-8", errors="replace")


def parse(path: Path) -> list[dict]:
    rows = []
    for line in read_log(path).splitlines():
        m = ROW.match(line)
        if not m:
            continue
        d = m.groupdict()
        rows.append({"name": PRETTY.get(d["name"], d["name"]),
                     "ours": float(d["ours"]), "theirs": float(d["theirs"]),
                     "diff": float(d["diff"]), "lo": float(d["lo"]),
                     "hi": float(d["hi"]),
                     "significant": d["verdict"].lower().startswith("signif")})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path, default=ROOT / "bench" / "significance_ceiling.log")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "comparison.png")
    ap.add_argument("--width", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    if not args.log.exists():
        print(f"no such log: {args.log}\nrun significance.py first")
        return 1

    rows = parse(args.log)
    if not rows:
        print(f"no comparison rows found in {args.log}")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Overall first, then the classes in the order the README uses
    order = ["Overall", "Hi-hat", "Kick", "Snare", "Cymbals", "Toms"]
    rows.sort(key=lambda r: order.index(r["name"]) if r["name"] in order else 99)
    rows.reverse()  # matplotlib draws upwards

    fig, ax = plt.subplots(figsize=(args.width, 0.62 * len(rows) + 1.6), dpi=args.dpi)
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    # the score labels sit in a column clear of the widest bar, so nothing overlaps
    widest = max(r["hi"] for r in rows)
    label_x = widest + 0.06

    for i, r in enumerate(rows):
        colour = WAVE if r["significant"] else GREY
        ax.plot([r["lo"], r["hi"]], [i, i], color=colour, linewidth=3.4,
                solid_capstyle="round", zorder=2)
        ax.plot([r["diff"]], [i], "o", color=colour, markersize=9,
                markeredgecolor=PAPER, markeredgewidth=1.4, zorder=3)
        ax.text(label_x, i, f"{r['ours']:.3f}  vs  {r['theirs']:.3f}",
                va="center", ha="left", fontsize=9.5,
                color=INK if r["significant"] else GREY, zorder=4)

    ax.axvline(0, color=INK, linewidth=1.2, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] for r in rows], fontsize=11, color=INK)
    ax.set_xlabel("difference in onset F1, drum2midi minus ReStem 2 Pro",
                  fontsize=10, color=INK)

    n_sig = sum(1 for r in rows if r["significant"])
    won = [r["name"] for r in rows if r["significant"]][::-1]
    # Lead with what held up. The rest of the story is true and is the caption.
    ax.set_title(f"{' and '.join(won)} hold up. The rest this test set cannot resolve.",
                 fontsize=12.5, color=INK, pad=14, loc="left")

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GREY)
    ax.tick_params(axis="x", colors=GREY, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(min(r["lo"] for r in rows) - 0.05, label_x + 0.22)
    ax.set_ylim(-0.8, len(rows) - 0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.86, bottom=0.22)
    fig.text(0.13, 0.045,
             f"95% intervals from resampling the 23 recordings 4000 times."
             f"  {n_sig} of {len(rows)} exclude zero.",
             fontsize=9, color=GREY, ha="left", va="bottom")
    fig.savefig(args.out, facecolor=PAPER)
    print(f"{args.out}  ({len(rows)} rows, {n_sig} significant)")
    for r in rows[::-1]:
        mark = "*" if r["significant"] else " "
        print(f"  {mark} {r['name']:<9}{r['diff']:+.3f}  [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
