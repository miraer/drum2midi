"""Twenty-three recordings decide the headline. What do they actually represent?

The bootstrap in significance.py answers a narrower question than it looks like it does.
Resampling the 23 tracks estimates how much the number would move if you drew another 23
recordings FROM THE SAME POPULATION. It says nothing about whether that population is the
one anyone cares about. A confidence interval cannot detect a biased sample; it will sit
tightly around the wrong answer.

So this asks the other question, and it asks it with measurements rather than adjectives.

  * what the corpus is, stated plainly -- length, excerpt duration, and the fact that all
    23 come from a single production series rather than 23 unrelated sources
  * how its class balance compares with the largest reference we have measured
  * and the part that can actually falsify something: does the published result survive
    reweighting the genres? If dropping the jazz moves the ReStem delta materially, the
    headline is partly a statement about how much jazz is in MDB.

The genre grouping below is a judgement, not a field in the dataset, and it is printed so
it can be argued with.

    python corpus_representativeness.py
    python corpus_representativeness.py --ours bench/ceiling --theirs restem_midi
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if {"-h", "--help"} & set(sys.argv[1:]):
    print(__doc__)
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from compare_with_restem import FAMILIES, PRETTY, prf  # noqa: E402
from significance import collect_tracks, per_track  # noqa: E402

# Our grouping of MDB's genre labels, not the dataset's. MedleyDB gives no genre field for
# these; the names are all that is on offer, so this is a reading of them.
GENRE = {
    "BebopJazz": "jazz", "CoolJazz": "jazz", "FreeJazz": "jazz", "FunkJazz": "jazz",
    "FusionJazz": "jazz", "LatinJazz": "jazz", "ModalJazz": "jazz", "SwingJazz": "jazz",
    "80sRock": "rock/pop", "Beatles": "rock/pop", "Britpop": "rock/pop",
    "Grunge": "rock/pop", "Hendrix": "rock/pop", "Punk": "rock/pop", "Rock": "rock/pop",
    "Rockabilly": "rock/pop", "Shadows": "rock/pop", "SpeedMetal": "rock/pop",
    "Zeppelin": "rock/pop",
    "Country1": "other", "Disco": "other", "Gospel": "other", "Reggae": "other",
}

# Tom share of onsets in E-GMD, the largest drum corpus we have counted: 1,074,753 tom
# onsets of 14.3M. Measured by us while surveying it for training data, not quoted.
EGMD_TOM_SHARE = 0.0749


def genre_of(stem: str) -> str:
    return GENRE.get(stem.replace("MusicDelta_", "").replace("_Drum", ""), "unlabelled")


def micro(rows, picks) -> float:
    tp = ref = est = 0
    for name in picks:
        for f in FAMILIES:
            tp += rows[name][f]["tp"]
            ref += rows[name][f]["ref"]
            est += rows[name][f]["est"]
    return prf(tp, ref, est)[2]


def onsets(rows, picks) -> int:
    return sum(rows[n][f]["ref"] for n in picks for f in FAMILIES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours", type=Path, default=ROOT / "bench" / "ceiling")
    ap.add_argument("--theirs", type=Path, default=ROOT / "restem_midi")
    args = ap.parse_args()

    tracks = collect_tracks()
    ours = per_track(args.ours, tracks)
    theirs = per_track(args.theirs, tracks)
    names = [n for n in ours if n in theirs]
    if not names:
        print("no track scored by both; check --ours and --theirs")
        return 1

    try:
        import soundfile as sf
        durs = {w.stem: sf.info(str(w)).duration for w, _, _ in tracks if w.stem in names}
    except Exception:
        durs = {}

    print(f"corpus: {len(names)} recordings, {onsets(ours, names)} annotated onsets")
    if durs:
        d = sorted(durs.values())
        print(f"audio:  {sum(d)/60:.1f} minutes in total, median excerpt {d[len(d)//2]:.0f} s, "
              f"shortest {d[0]:.0f} s, longest {d[-1]:.0f} s")
    prefixes = {n.split("_")[0] for n in names}
    print(f"source: {len(prefixes)} distinct name prefix(es) -- {', '.join(sorted(prefixes))}")
    if len(prefixes) == 1:
        print("        every recording comes from one production series, so these are not")
        print("        23 independent sources. Drummer, room and engineer are unknown and")
        print("        are plausibly shared; nothing in the corpus lets us check.")

    print("\nclass balance, and what it is not")
    print(f"{'class':<10}{'onsets':>9}{'share':>9}")
    print("-" * 28)
    total = onsets(ours, names)
    for f in FAMILIES:
        n = sum(ours[x][f]["ref"] for x in names)
        print(f"{PRETTY.get(f, f):<10}{n:>9}{n/total:>8.2%}")
    tom_key = next((f for f in FAMILIES if PRETTY.get(f, f).startswith("tom")), "TT")
    toms = sum(ours[x][tom_key]["ref"] for x in names) / total
    print("-" * 28)
    print(f"toms are {toms:.2%} here against {EGMD_TOM_SHARE:.2%} in E-GMD "
          f"({EGMD_TOM_SHARE/toms:.1f}x more there).")
    print("That is not a small imbalance, it is a different instrument distribution, and")
    print("it is why no tom conclusion should ever have been drawn from this corpus.")

    groups: dict[str, list[str]] = {}
    for n in names:
        groups.setdefault(genre_of(n), []).append(n)

    print("\ngenre mix, grouped by hand from the track names")
    print(f"{'genre':<12}{'tracks':>8}{'onsets':>9}{'share':>8}{'ours':>8}{'ReStem':>8}{'delta':>8}")
    print("-" * 61)
    for g in sorted(groups, key=lambda k: -len(groups[k])):
        pick = groups[g]
        o, t = micro(ours, pick), micro(theirs, pick)
        print(f"{g:<12}{len(pick):>8}{onsets(ours, pick):>9}"
              f"{onsets(ours, pick)/total:>7.1%}{o:>8.3f}{t:>8.3f}{o - t:>+8.3f}")
    print("-" * 61)
    base_o, base_t = micro(ours, names), micro(theirs, names)
    print(f"{'published':<12}{len(names):>8}{total:>9}{1.0:>7.1%}"
          f"{base_o:>8.3f}{base_t:>8.3f}{base_o - base_t:>+8.3f}")

    print("\nleave one genre out -- how much of the headline is that genre?")
    print(f"{'dropped':<12}{'tracks':>8}{'ours':>8}{'ReStem':>8}{'delta':>8}{'moves by':>10}")
    print("-" * 56)
    swings = []
    for g in sorted(groups):
        pick = [n for n in names if genre_of(n) != g]
        if not pick:
            continue
        o, t = micro(ours, pick), micro(theirs, pick)
        swing = (o - t) - (base_o - base_t)
        swings.append(abs(swing))
        print(f"{g:<12}{len(pick):>8}{o:>8.3f}{t:>8.3f}{o - t:>+8.3f}{swing:>+10.3f}")

    per_genre = [micro(ours, groups[g]) - micro(theirs, groups[g]) for g in groups]
    equal = sum(per_genre) / len(per_genre)
    print("-" * 56)
    print(f"{'each genre':<12}{'weighted':>8}{'equally':>8}{'':>8}{equal:>+8.3f}"
          f"{equal - (base_o - base_t):>+10.3f}")

    print(f"""
The published delta is {base_o - base_t:+.3f} with the 95% interval significance.py gives it.
The largest swing from dropping a whole genre is {max(swings):+.3f}, and weighting every genre
equally instead of by onset count moves it by {equal - (base_o - base_t):+.3f}.

Compare those against the interval before reading anything into them. A reweighting that
moves the result by less than the interval has not shown the corpus to be unrepresentative
-- it has shown that this particular kind of unrepresentativeness does not reach the
answer. And a reweighting that moves it by more than the interval means the headline is
partly a statement about MDB's genre mix, which no bootstrap would ever have revealed.

What none of this can test, and what should be said plainly wherever the number appears:
21.8 minutes of one production house's genre demonstrations is not a sample of recorded
drumming. It is the material that exists with onset-level hand annotation, which is a
different property entirely. The second corpus in the README is there for this reason and
is the only real defence -- a claim that holds on IDMT-SMT-Drums as well is a claim that
has survived a change of everything except the code being measured.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
