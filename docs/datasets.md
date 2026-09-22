# Drum datasets, and what is actually in them

Written after a survey of 18 datasets in which every download link was fetched and, where
possible, the annotations were parsed rather than quoted from the paper. Two published
descriptions turned out to be wrong, and one dataset that nobody in this corner of the
literature cites turned out to be the best source of the exact thing this project is
worst at.

Figures marked **(measured)** were computed from the annotation files. Where they say
**(measured here)**, they were recomputed in this repository by a script you can run —
E-GMD's numbers were checked that way and matched the published ones to the digit.
Everything else is cited.

---

## What we currently test on

| Set | Files | Onsets | Audio | Role here |
|---|---|---|---|---|
| MDB Drums | 23 | 7994 **(measured)** | real kits, real rooms, close-miked | main benchmark |
| IDMT-SMT-Drums | 95 | 7927 | mixed, see below | independent check |
| GMD | 1150 performances | — | **electronic kit**, human performance | velocity and articulation |

### IDMT is three datasets wearing one name

The 0.935 headline averages over three kinds of audio in very lopsided proportions
(`idmt_by_kind.py`, all figures measured):

| Subset | Files | Onsets | kick | snare | hi-hat | MICRO | What it is |
|---|---|---|---|---|---|---|---|
| RealDrum | 14 | 1289 | 0.972 | 0.816 | 0.977 | 0.938 | a real kit in a room |
| WaveDrum | 70 | 5589 | 0.973 | 0.820 | 0.968 | 0.937 | built from samples |
| TechnoDrum | 11 | 1049 | 0.990 | 0.890 | 0.902 | 0.922 | a drum machine |
| **All** | **95** | **7927** | 0.975 | 0.828 | 0.960 | **0.935** | the quoted number |

Three quarters of that evidence is synthetic, which is worth knowing. The good news is
that it does not appear to flatter us: real minus synthetic is **+0.003**, 95% CI
[−0.023, +0.028], not significant. The caveat is that RealDrum is 14 files, and IDMT
annotates only kick, snare and hi-hat — no toms, no cymbals — so this says nothing about
the classes we are weakest on.

### GMD is a real performance through an electronic kit

Not synthetic in the usual sense: a human plays, and the MIDI is captured from the pads,
so onsets and velocities are exact rather than estimated. But the *sound* comes from a
Roland module — no room, no mic bleed, no cymbal wash across the kit. Separation looks
easier than it is: MICRO 0.904 on GMD against 0.862 on the same settings elsewhere.

---

## The tom problem is a data problem

MDB Drums holds **90 tom onsets, 1.14% of the corpus (measured)**. IDMT annotates none at
all. Every tom conclusion anyone draws from these two sets rests on those 90 examples —
including the one this project published and has since retracted (see the comparison
section of the main README).

Tom coverage across the datasets that are actually obtainable:

| Dataset | Tom onsets | % of set | Tom classes | Acoustic? | Access |
|---|---|---|---|---|---|
| TMIDT `midi_drums_bal_l` | 1,101,434 | 16.4% | 3 | no, synthesised | open, **no licence stated** |
| **E-GMD** | **1,074,753 (measured here)** | 7.49% | 3 + rims | no, electronic kit | **open, CC BY 4.0** |
| ADTOF-YT | ~601,092 | ~7.2% | 1 merged | **yes, commercial music** | on request |
| StemGMD | ~ as E-GMD | ~7% | **3, as isolated stems** | no, sampled kits | open, CC BY 4.0 |
| **ENST-Drums** | **2,758** | 6.03% | **4** | **yes, real kits and rooms** | open, CC BY-NC-**ND** |
| RWC 2.0 | 10,199 | 2.61% | 6 | yes, studio recordings | **open since Feb 2026**, CC BY-NC |
| Slakh2100 | full GM range | — | 6 | no, synthesised | open, CC BY 4.0 |
| **MDB Drums** (ours) | **90 (measured here)** | **1.14%** | 4 subclasses | yes | open, CC BY-NC-SA |
| IDMT-SMT-Drums (ours) | **0** | 0% | none | partly | open, CC BY-NC-ND |
| RBMA-13 | **0** | 0% | none | yes | open — see correction |

### E-GMD, verified rather than quoted

The MIDI-only archive is 102 MB against 90 GB for the audio and carries the entire
annotation, so the claims could be checked before committing to the big download
(`fetch_egmd.py --survey`). All 45,537 files parsed, none unreadable, **14,341,145
note-ons**, and every published figure matched to the digit:

| Family | Onsets | % |
|---|---|---|
| snare | 4,469,307 | 31.16% |
| hi-hat | 3,913,861 | 27.29% |
| kick | 2,842,803 | 19.82% |
| cymbal | 1,824,834 | 12.72% |
| **tom** | **1,074,753** | **7.49%** |
| auxiliary (tambourine, clap, cowbell) | 215,587 | 1.50% |

**Those counts are renderings, not distinct drumming, and the difference is a factor of 43.**
E-GMD's 45,537 clips are 1,059 performances — keyed by `(drummer, session, id)` — each
re-recorded on exactly 43 kits, with no exceptions: 444.5 hours of audio over **10.3 hours of
distinct playing**. Parsing one rendering of each performance gives **25,524 distinct tom
onsets**, and 25,524 × 43 = 1,097,532, which is the table above to within the handful of
files the two passes count differently.

Both numbers are true and they answer different questions. 1,074,753 is how many tom onsets a
training run sees; 25,524 is how much distinct drumming exists to learn from, and the other
43/44ths are timbral augmentation across kits. For a model that has to survive a separator's
artefacts that augmentation may be exactly what is wanted — but it is not more music, and
quoting the larger figure as "how much tom data E-GMD has" overstates the corpus by two orders
of magnitude.

Toms break down as 424,582 on tom 3 head, 386,871 on tom 1, 130,591 on tom 2, plus
132,709 across the three rims. Velocity spans **4–127**, mean 66.5, 124 distinct values,
with **48.54% of onsets below 60 and 25.04% below 40** — which is the ghost-note material
MDB labels but cannot quantify, since its annotations carry no velocity.

Roland TD kits deviate from General MIDI, so those figures depend on the mapping in
`fetch_egmd.py`; it is the one published with GMD.

### ENST ships isolated tom stems, which nothing else does

The archive carries, per sequence: `kick/ snare/ hi-hat/ tom_1/ tom_2/ tom_3/
overhead_L/ overhead_R/ dry_mix/ wet_mix/ accompaniment/`. That is **752 isolated,
close-miked, genuinely acoustic tom tracks with hand-made onset labels**.

This matters for us specifically. Our stem-fusion stage consumes exactly this kind of
signal, and a previous attempt to fix toms with synthetic audio failed badly — ADT_STR,
trained on synthesis, scored 0.140 on toms. Real tom audio with exact labels sidesteps
that failure mode instead of repeating it.

---

## Velocity and ghost notes

| Dataset | Velocity | Ghost-note material |
|---|---|---|
| E-GMD | 4–127, mean 66.5, 124 distinct values **(measured here)** | **48.54% of onsets below 60, 25.04% below 40** |
| StemGMD | inherited from GMD | `StemGMD_single_hits.zip`: every piece at 10 velocities, 30→127 |
| MDB Drums | subclass labels, not velocity | ghost/buzz annotated as subclasses — the reference standard |
| RWC 2.0 | 1–127, but **score** velocities | beat-aligned score MIDI, not a transcription of the performance |

---

## Two corrections to the published record

Both were found by downloading the thing and looking inside, and both would cost someone
else days.

**RBMA-13 does not contain toms.** The public annotation zip has **three classes and zero
tom onsets (measured)**. The 8-, 18- and 47-class variants referenced in the literature
are not in the public download. Anyone planning tom work around RBMA-13 should check
before committing.

**STAR Drums is not BSD-3-Clause.** Its Zenodo record carries that tag, but it covers only
the scripts. The audio is mostly CC BY-NC-SA. A commercially-usable subset of roughly
1,742 tracks exists inside it, but the headline licence tag is misleading.

Also worth knowing: **RWC Music Database became downloadable in February 2026** and was
genuinely re-licensed to CC BY-NC 4.0, replacing the old CD-by-post and pledge-form
process. Its annotations were re-aligned with DTW and manually checked in April 2026. Two
caveats: only 21 RWC-P tracks are marked "Live drums", and the drum channel is heavy with
auxiliary percussion, so filter before use.

---

## Where this leaves us

Non-commercial licences are no longer a blocker for this project, so the ranking is by
what fixes the measured weakness rather than by licence:

1. **ENST-Drums** for real acoustic toms and ghost notes, with isolated tom stems to
   train or validate the fusion stage on.
2. **E-GMD** for sheer volume — **284x** more distinct tom onsets than MDB (25,524 against
   90), or 11,900x if the 43 per-kit renderings of each performance are counted separately,
   which is how the raw 1,074,753 arises. The larger multiplier is the one to avoid quoting:
   it counts the same drumming forty-three times. The velocity distribution the ghost-note
   work needs is real either way. Measure the electronic-to-acoustic domain gap before
   trusting it; do not assume it away.
3. **RWC 2.0** as a third independent test set of real recordings, which is the one thing
   23 tracks of MDB cannot give us.

The reason to want all three is in the main README: with 23 recordings, a single system's
confidence interval spans ±0.06, and differences smaller than about one point of F1 are
simply not resolvable.
