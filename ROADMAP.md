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
work](README.md#-a-fallback-for-the-passages-where-adtof-goes-silent). Spectral
pre-emphasis ahead of ADTOF was measured next and [fails its own
gate](README.md#-pre-emphasis-ahead-of-adtof-for-the-same-passages): two fixed tilts lift
4.2% and 25.3% [10.6, 41.5] of the blind onsets to threshold, against a pre-registered
50%, and the mallet recordings respond least. Nothing else is currently proposed.
Untried directions, none of them evaluated: a second transcriber trained on different
material, or simply detecting the condition and telling the user rather than writing
silence.

### Toms, still the weakest class

F1 **0.539** on ENST's 2617 tom onsets, which is the figure to trust, against 0.589 on
MDB's 90. Bounding the threshold recovered +0.197 and that is shipped, but the confusion
breakdown says a discrimination problem remains underneath: all 60 false toms on MDB land
within 50 ms of a real onset of another class, 53% snare and 33% kick. That is not
something a threshold reaches.

E-GMD carries 1,074,753 tom onsets under CC BY 4.0 and was the obvious training material —
though that figure is **25,524 distinct tom onsets rendered on 43 kits**, so the honest
multiplier against MDB's 90 is 284x rather than 11,900x, and the remainder is timbral
augmentation rather than more drumming. See [datasets.md](docs/datasets.md#e-gmd).
The bar was fixed in advance: tom F1 on ENST **with the ceiling in place**, so 0.539 rather
than the 0.342 it was before, over all 2,617 tom onsets, with an interval. One attempt has
been measured against it, and it lost by 0.204 [−0.274, −0.137] ([Queued](#queued)).

### Pedal hi-hat

19 correct of the 230 events we label pedal; 3 of 513 for ReStem. Those denominators are
each system's own pedal output rather than the reference, which carries 523 pedal onsets —
a reader took them for reference counts, so they are spelled out here. A separate
measurement on a different corpus suggests it may not be there
to find: across the 1102 hi-hat hits in eight Groove MIDI recordings the pedal chick and the
closed hat differ by peak
0.0457 against 0.0466 and centroid 11993 against 12454, with d′ between 0.01 and 0.48.
Published as a probable limit of the signal rather than a defect, but it has not been
proven impossible.

---

## Queued

| | what would settle it |
|---|---|
| **Reproduce the ENST figures independently** | ✅ done 20 Sept. A second machine reproduced kick, snare, hi-hat and cymbals to three decimals with identical intervals; toms differed by exactly the +0.197 the ceiling buys, which identified their run as pre-ceiling rather than a disagreement. |
| **Train toms on E-GMD** | ❌ **closed 23 Sept, measured.** The bar was fixed in advance: tom F1 on ENST above 0.539, with an interval. A detector trained on E-GMD stems (LarsNet, 3 epochs, threshold 0.270, both chosen on E-GMD) replaced the tom channel and scored **0.335 against 0.539: −0.204 [−0.274, −0.137]** paired over 210 recordings. Not one resample landed above zero. Recall rose 0.466 → 0.741, but precision fell 0.638 → 0.216 on 4.7 times as many notes. This closes one trainer, one separator and one threshold. It does not close learned tom detection. The discrimination problem it was aimed at is still there: every model measured puts its tom/snare confusion in its most confident predictions. [Write-up](README.md#-a-learned-onset-detector-on-separated-stems-for-toms-measured-properly-it-loses). |
| **Survey the alternatives to ADTOF** | ✅ **done 20 Sept.** Inventory built by reading licences and release assets, not READMEs. Upstream ADTOF publishes exactly one checkpoint, so our port is the complete set. The Inverse Drum Machine measured at MICRO 0.703 against 0.860 — Apache-2.0 against CC BY-NC-SA, which is the trade, and today it is not worth taking. Four candidates remain runnable and unrun; Separate-and-Detect emits exactly our five classes and needs a GPU. [Full table](README.md#the-rest-of-the-alternatives-surveyed). |
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

- ~~**MDX23C through OpenVINO**~~ — ✅ **measured 20 Sept, and the earlier conclusion
  does not survive.** The NPU investigation said OpenVINO loses everywhere; that was
  measured on a toy convolution and on ADTOF, which is a small recurrent net. MDX23C is a
  109 M-parameter convolutional one and behaves differently.

  The whole model **cannot** be exported to ONNX: its STFT front-end works on complex
  tensors and ONNX has no complex type. The convolutional body can, with the STFT left in
  torch where it is cheap. On a 3-second chunk of real audio, medians of five runs:

  | | time | vs torch cpu | vs torch xpu | max relative error |
  |---|---|---|---|---|
  | torch cpu | 7.77s | — | 0.20x | — |
  | **torch xpu — what the pipeline uses today** | **1.55s** | 5.00x | — | — |
  | OpenVINO CPU | 5.08s | 1.53x | 0.31x | 0.0000% |
  | **OpenVINO GPU** | **0.85s** | 9.10x | **1.82x** | 0.2355% |
  | OpenVINO NPU | 1.14s | 6.81x | 1.36x | 0.0742% |

  So there is a real gain and it is **1.8x, not 9x**: against torch CPU it looks
  enormous, but the pipeline has not used torch CPU for this stage in weeks. The NPU
  finally wins something — 1.36x over the GPU path the tool actually takes — which is the
  first positive NPU result in this project.

  **Not adopted, and the reasons are listed rather than waved at.** The GPU output differs
  from torch by 0.24% and nobody has measured what that does to onset F1; the STFT and
  iSTFT stay in torch either way, so the end-to-end gain is smaller than the table; and
  the export needs a 417 MB external-data file per model. `bench_mdx_openvino.py` runs
  the whole thing, including the numerical check, because a speedup on a graph that
  computes something else is not a speedup.
- **Power draw** — never measured anywhere in this project, and it is the one axis on
  which an NPU should genuinely win. Now more interesting than it was: the NPU is within
  1.4x of the GPU on this model, so if it draws meaningfully less it may be the better
  device even without being the faster one.
- **ReStem's `Other` stem** — this entry said our converter does not emit pitch 60. It
  does: `restem_to_midi.py` has mapped `other` → 60 since the first commit, so the claim
  was wrong about our own code. What was actually observed is that three notes did not
  line up on one comparison, and whether the `trigger_events.json` carried them at all
  can no longer be checked — ReStem overwrites its export directory on every render.
  None of the 23 benchmark recordings produce `other` events, so nothing published
  depends on it. **To close:** one render with every stem enabled, on material that
  produces `other` events, then `compare_restem_export.py`. Needs a human, because the
  quality mode cannot be set by script, and the trial expires 23 September here.
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
| Pre-emphasis ahead of ADTOF for deaf passages | failed a pre-registered gate: two fixed tilts lift 4.2% and 25.3% [10.6, 41.5] of blind onsets to threshold against 50% needed; mallets 0 and 11 of 112 |
| Learned onset detector on separated stems | toms, E-GMD-trained, paired on ENST: **−0.204 [−0.274, −0.137]**. Recall rose and precision collapsed. The older "fires everywhere" verdict was frame-exact scoring and is withdrawn. Other classes are unmeasured |
| Replacing ADTOF with ADT_STR (2026) | MICRO 0.673 against 0.882 on the same 23 tracks |
| Inverse Drum Machine for velocity | worse, and slower |
| Tuning all five thresholds globally | overfitting — 0.848 held out against a stock 0.850 |
| Lowering thresholds for soft beaters | 588 extra tom notes buy five real ones; helps mallets, harms brushes |
| A density-aware tom threshold | an *oracle* density estimator still would not know the threshold: ρ = −0.057 |
| Post-processing toms on all five channels | ranking improves to AUC 0.962 and F1 *falls*; calibration encodes the fitted corpus's class balance |
| Suppressing snare bleed in the tom channel | +0.041 on MDB, −0.019 on ENST — trades recall for precision |
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
a tom recall belonged to a rival policy measured elsewhere. It then happened a fourth
time, and worse: `--thresholds` silently disables `TOM_CEILING`, so a published table
charged a two-change difference entirely to the one change it named. **A number is not
usable until something states what produced it**, so the scripts print the export they
scored and the pipeline prints which tom policy ran.

The fourth comes from the second machine, about E-GMD: its 45 537 files are 1059
performances rendered through 43 kits, so resampling files instead of performances gives
intervals 6.6× too narrow. **The resampling unit is the thing that actually varies** —
the recording on MDB, the performance on E-GMD — and getting it wrong does not look like
an error, it looks like a stronger result.
