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

# What else is on disk, so "23 recordings" is never mistaken for "all the real data we
# have". Counted by the commands named, not estimated. ENST figures are the set
# benchmark_enst.py actually scores (phrase, solo, minus-one, MIDI-minus-one on wet_mix);
# IDMT is split by filename prefix, which is how the dataset distinguishes its three kinds.
OTHER_CORPORA = [
    # name, recordings, minutes, onsets, tom share, acoustic?, note
    ("ENST-Drums", 210, 106.4, 45097, 0.0578, True,
     "real kits, but only 3 drummers, 3 rooms"),
    ("IDMT RealDrum", 14, None, 1289, 0.0, True,
     "a real kit in a room, no toms or cymbals"),
    ("IDMT WaveDrum+Techno", 81, None, 6638, 0.0, False,
     "samples and a drum machine"),
]


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


def class_f1(rows, picks, family) -> float:
    tp = ref = est = 0
    for name in picks:
        v = rows[name][family]
        tp, ref, est = tp + v["tp"], ref + v["ref"], est + v["est"]
    return prf(tp, ref, est)[2]


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
partly a statement about MDB's genre mix, which no bootstrap would ever have revealed.""")

    print("\nhow many recordings each class actually rests on")
    print(f"{'class':<10}{'onsets':>8}{'tracks':>8}{'top share':>11}{'eff. n':>8}"
          f"{'delta':>9}{'drop top':>10}  largest contributor")
    print("-" * 88)
    for f in FAMILIES:
        counts = sorted(((ours[n][f]["ref"], n) for n in names), reverse=True)
        tot = sum(c for c, _ in counts)
        if not tot:
            continue
        nz = [c for c, _ in counts if c]
        eff = 1 / sum((c / tot) ** 2 for c in nz)
        top_n, top_name = counts[0][0], counts[0][1]
        rest = [n for n in names if n != top_name]
        d_all = class_f1(ours, names, f) - class_f1(theirs, names, f)
        d_cut = class_f1(ours, rest, f) - class_f1(theirs, rest, f)
        print(f"{PRETTY.get(f, f):<10}{tot:>8}{len(nz):>8}{top_n/tot:>10.1%}{eff:>8.1f}"
              f"{d_all:>+9.3f}{d_cut:>+10.3f}  "
              f"{top_name.replace('MusicDelta_', '').replace('_Drum', '')}")
    print("-" * 88)
    print("""'eff. n' is the inverse Simpson index over the per-recording shares: the number of
equally-sized recordings that would carry the same concentration. A class with 1002
onsets spread over four effective recordings has the evidence of four recordings, not of
a thousand onsets, and quoting the onset count invites the opposite conclusion.

'drop top' removes the single largest contributor to that class and recomputes. Every
conclusion survives it, and the hi-hat advantage grows rather than shrinks -- so the
result is not an artefact of one recording. That is the reassuring half. The other half
is that toms are carried by 7 recordings of 23 and cymbals by 4.5 effective ones, and no
robustness check makes those rows interpretable.""")

    print("\nand this is not the only real drumming we hold")
    print(f"{'corpus':<22}{'recs':>6}{'minutes':>9}{'onsets':>9}{'toms':>8}  what it is")
    print("-" * 92)
    mins = sum(durs.values()) / 60 if durs else 0.0
    print(f"{'MDB Drums':<22}{len(names):>6}{mins:>9.1f}{total:>9}{toms:>7.2%}  "
          f"the corpus above, and the only one ReStem has been run on")
    ac_r, ac_o = len(names), total
    for nm, recs, m, ons, tom, acoustic, note in OTHER_CORPORA:
        mm = f"{m:>9.1f}" if m else f"{'—':>9}"
        print(f"{nm:<22}{recs:>6}{mm}{ons:>9}{tom:>7.2%}  {note}")
        if acoustic:
            ac_r += recs
            ac_o += ons
    print("-" * 92)
    print(f"real acoustic in total: {ac_r} recordings, {ac_o} onsets — "
          f"{ac_o/total:.1f}x the corpus the headline runs on.")
    print("""
So the honest summary of the evidence base is not "23 recordings". It is that ENST holds
5.7x the onsets of MDB on real kits with a tom share in the right range, and that the
comparison runs on MDB only because that is the one ReStem was ever pointed at.

ENST is not a cure for the problem described above, and claiming it would be would repeat
the mistake. It is 210 recordings from 3 drummers in 3 rooms: for anything that depends on
the player or the kit, the number of independent units is nearer 3 than 210, and an
interval over its recordings is too narrow in the same way. What it offers is a DIFFERENT
bias -- other players, other rooms, exercises instead of genre demonstrations, and toms
that actually occur. A result that survives both corpora has survived a change of
everything except the code.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
