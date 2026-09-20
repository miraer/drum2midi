# Roadmap

What is unsolved, what is queued, and what was tried and abandoned. Kept here rather than
in an issue tracker because most of it is measurement rather than feature work, and the
numbers are the point.

Every entry names either a figure or the measurement that would settle it. An item with
neither does not belong on this list.

---

## Unsolved, and it matters

### Passages the transcriber cannot hear at all

On one real recording, 24 seconds containing about 70 audible onsets transcribed to
nothing. Established: this is not a threshold that could be lowered — ADTOF's activations
there peak at 0.008 to 0.08 against thresholds of 0.14 to 0.32, which is noise rather
than a near miss. The material is percussive but dull, harmonic fraction 0.101 and
spectral centroid 2129 Hz against 4037 Hz in the loud sections.

**ReStem 2 Pro fails on the same passage**, emitting 73 of its 106 notes there as pitch
60, its unclassified bucket. That makes it a property of the material rather than of
either transcriber, which is a harder problem than it first looked.

The obvious remedy — fire a fallback detector only where the model is silent — was
measured and [does not
work](README.md#-a-fallback-for-the-passages-where-adtof-goes-silent). Nothing else is
currently proposed. Untried directions, none of them evaluated: a second transcriber
trained on different material, spectral pre-emphasis ahead of ADTOF, or simply detecting
the condition and telling the user rather than writing silence.

### Toms, still the weakest class

F1 **0.539** on ENST's 2617 tom onsets, which is the figure to trust, against 0.589 on
MDB's 90. Bounding the threshold recovered +0.197 and that is shipped, but the confusion
breakdown says a discrimination problem remains underneath: all 60 false toms on MDB land
within 50 ms of a real onset of another class, 53% snare and 33% kick. That is not
something a threshold reaches.

E-GMD carries 1,074,753 tom onsets under CC BY 4.0 and is the obvious training material.
The measurement that decides it is tom F1 on ENST **with the ceiling in place**, so the
bar is 0.539 rather than the 0.342 it was before — training has to beat the fixed output
stage, not the broken one.

### Pedal hi-hat

19 of 230 here; 3 of 513 for ReStem. A separate measurement suggests it may not be there
to find: across 1102 annotated hits the pedal chick and the closed hat differ by peak
0.0457 against 0.0466 and centroid 11993 against 12454, with d′ between 0.01 and 0.48.
Published as a probable limit of the signal rather than a defect, but it has not been
proven impossible.

---

## Queued

| | what would settle it |
|---|---|
| **Reproduce the ENST figures independently** | ✅ done 20 Sept. A second machine reproduced kick, snare, hi-hat and cymbals to three decimals with identical intervals; toms differed by exactly the +0.197 the ceiling buys, which identified their run as pre-ceiling rather than a disagreement. |
| **Train toms on E-GMD** | tom F1 on ENST above 0.539, with an interval, held out on a drummer not used in training |
| ~~**Synthetic training data with realistic degradation**~~ | ❌ **closed 20 Sept as measured-unnecessary.** See below. |
| **ENST as the acoustic counterweight** | it ships isolated close-mic tom stems, which is exactly what the stem-fusion stage consumes; unexplored |

### Why synthetic training data was closed without being built

The case for it was that a model needs more timbral variety than one corpus provides.
E-GMD already **is** that variety — 1059 performances rendered through 43 kits — so the
question is whether the variety it has generalises. The second machine measured it by
holding out whole kits, which E-GMD's own split never does.

It does generalise, and where it fails is not our problem:

| held-out kit | tom F1 |
|---|---|
| 60s Rock, Arena Stage, Cassette, Classic Rock, Jazz, Alternative, Acoustic Kit | 0.601 – 0.683 |
| JingleStacks, Dark Hybrid | 0.358 – 0.430 |
| **909 Simple, Ele-Drum** | **0.058, 0.015** |

Every kit at the top is an acoustic set, every kit at the bottom is a drum machine, and
`Ele-Drum` returns 0.015 across three independent held-out runs. The model transfers
across acoustic timbre for free and collapses on drum machines — **and nobody is asking
this tool to transcribe a 909.** Synthesising more of what it already handles buys
nothing, and the gap that actually matters, an electronic kit against a miked kit in a
room, is not in E-GMD at all.

One methodological result worth more than the conclusion: **the headline drop is not a
stable quantity.** Across four seeds it ranged +0.045 to +0.177 and one interval contained
zero, because the number is just how many electronic kits landed in the held-out eight.
The per-kit map is the finding; any single averaged drop would have been a sampling
artefact quoted as a result.

---

## Smaller, and specific

- **MDX23C through OpenVINO** — the one NPU case never measured. The
  [investigation](README.md) settled the question on a toy convolution and on the real
  ADTOF model, and both said no. The separator itself was never tried. Toy evidence
  misled twice inside that same investigation, so this is a gap rather than a conclusion.
- **Power draw** — never measured anywhere in this project, and it is the one axis on
  which an NPU should genuinely win.
- **ReStem's `Other` stem** — our converter does not emit pitch 60. Harmless for onset
  scoring, since it is not a scored class, but on one recording it was 22% of what ReStem
  emitted. Decide whether dropping it flatters their precision.
- **`TOM_CEILING` is the conservative end of a measured trade**, not an optimum. Ceilings
  of 0.32 to 0.40 buy more on dense material and cost MDB significantly; the exchange
  rate is measured. 0.45 assumes people mostly convert tom-sparse material, which is a
  judgement about users rather than a measurement.

---

## Tried, measured, abandoned

These live in full in the README under [Design decisions, and the ideas that
failed](README.md#design-decisions-and-the-ideas-that-failed). Summarised here so that
nobody proposes them again without new evidence:

| idea | why it died |
|---|---|
| Fallback where the model is silent | blind share does not predict per-track F1; three of the five worst tracks have **zero** blind onsets; ceiling on recall 1.27% |
| Learned onset detector on separated stems | fires everywhere |
| Replacing ADTOF with ADT_STR (2026) | MICRO 0.673 against 0.882 on the same 23 tracks |
| Inverse Drum Machine for velocity | worse, and slower |
| Tuning all five thresholds globally | overfitting — 0.848 held out against a stock 0.850 |
| Learned pedal hi-hat classifier | does not transfer |
| Learned ride/crash classifier | no better than the rule it replaced |
| Running ADTOF on isolated stems | worse than on the mix |
| Our own trigger replacing ADTOF | much worse |
| Hi-hat decay as an articulation cue | 0.83×, 95% CI [0.66, 1.38] across seven tracks — the method cannot measure it |

---

## How this list is kept

One rule, learned the hard way on 19 September 2026. Three tasks investigating the deaf
passages were closed as *measurements*, and the measurements were honest — but the defect
itself then stopped being tracked, because closing an investigation looks like finishing
something.

**Closing an investigation is not fixing a defect.** When a measurement task finishes,
either the problem is gone or it is still here under a new name.

The second rule follows from five claims withdrawn in a single day, four of them the same
error: a figure taken from one recording and never counted across the corpus. **An
observation about one recording is not a finding until something has counted the rest**,
and the enforcement is a script rather than more care.

The third rule was paid for three times in the same week, once in the published README:
a figure read out of one run and written beside a figure from another, with nothing in
either log saying which run produced it. A stale default sent `significance.py` at an
export that predated a constant; a half-width was a hardcoded number no script computed;
a tom recall belonged to a rival policy measured elsewhere. **A number is not usable
until something states what produced it**, so the scripts print the export they scored
rather than leaving it to the person reading the log.

The fourth comes from the second machine, about E-GMD: its 45 537 files are 1059
performances rendered through 43 kits, so resampling files instead of performances gives
intervals 6.6× too narrow. **The resampling unit is the thing that actually varies** —
the recording on MDB, the performance on E-GMD — and getting it wrong does not look like
an error, it looks like a stronger result.
