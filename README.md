# drum2midi

![drum2midi](docs/banner.png)

Convert a mixed drum recording into MIDI, using only open-source models.

**Measured against [ReStem 2 Pro](https://restemapp.com/restem2)** — a $199 commercial
product — on 23 hand-annotated recordings:

| | drum2midi | ReStem 2 Pro | 95% CI on the difference |
|---|---|---|---|
| **Overall onset F1** | **0.882** | 0.820 | **[+0.033, +0.097]** |
| Kick | 0.960 | 0.951 | [−0.013, +0.040] |
| Snare | 0.844 | 0.845 | [−0.025, +0.020] |
| Hi-hat | **0.892** | 0.754 | **[+0.062, +0.249]** |
| Cymbals | 0.869 | 0.755 | [−0.004, +0.373] |
| Toms | 0.605 | 0.699 | [−0.304, +0.139] |
| Ghost notes (recall) | 0.718 | 0.747 | — |
| Ride vs crash | 0.917 | 0.953 | — |
| Hi-hat articulation | 0.690 | 0.691 | — |

**Only two of those rows mean anything.** The intervals come from resampling the 23
recordings 4000 times (`significance.py`), and every row whose interval contains zero is
a difference this test set cannot resolve. The overall win is real, and it is carried
almost entirely by the hi-hat. The per-class rows for kick, snare, cymbals and **toms**
are not evidence of anything: MDB holds just **90 tom onsets, 1.14% of the set**, so the
tom interval is ±0.22 and spans both directions.

> **⚠ These are our measurements of someone else's product, and they may be wrong.**
> ReStem's developers were not involved in any of this and have not reviewed it. We are
> not neutral parties — we measured a competitor to our own work, using software we had
> to learn from the outside.
>
> The known caveat is a setting: ReStem's hi-hat articulation boundaries are
> user-adjustable, and we left them at their defaults. At the documented default split,
> every hi-hat hit in this corpus was labelled closed, which is most of the hi-hat gap.
> [The detail, and what it does and does not mean](#where-each-side-wins).
>
> The ReStem team have been written to and told this is public. If they say the product
> was used incorrectly, these numbers get rerun and corrected here, and the correction
> gets stated plainly rather than quietly edited in. That has already happened five times
> with our own claims — most recently to a caveat in this very section.

<sub>Same 23 MDB Drums tracks, same `mir_eval` code, same 50 ms MIREX tolerance, both
given the isolated drum recording. ReStem was driven through its own interface
(`restem_ui.ps1`) and its normal MIDI export was converted with `restem_to_midi.py`;
nothing was extracted from its model files. Reproduce with `compare_with_restem.py` and
`significance.py` —
[full methodology and where each side wins](#comparison-with-restem-2-pro).</sub>

On a second, independent corpus — all 95 files of IDMT-SMT-Drums, 7927 onsets — the same
settings score **0.935**, so this is not tuned to one dataset.

Every number in this README comes from a script in this repo. Negative results are
reported next to positive ones — a dozen ideas that sounded good were measured and
thrown away, including four tried this week, and each one is documented below with the
measurement that killed it.

---

## How it works

```mermaid
flowchart TD
    IN["audio in<br/><i>drum stem, or a whole song</i>"]
    EX["<b>0. extract drums</b> — htdemucs<br/><i>only with --from-song</i>"]
    STEM(["drum audio"])
    ADT["<b>1. what and when</b> — ADTOF<br/>5 onset classes, 100 fps<br/><b>reads the mixture of drums</b>"]
    SEP["<b>2. how loud</b> — MDX23C<br/>6 stems incl. separate ride/crash"]
    ART["<b>3. articulation and velocity</b><br/><i>this repository, no external model</i>"]
    T1["<b>split_toms</b><br/>which tom was hit<br/><i>pitch order right 36/36</i>"]
    T2["<b>split_hats</b><br/>open / closed / pedal<br/><i>from how long it rings</i>"]
    T3["<b>split_ride</b><br/>ride or crash<br/><i>beat 3 learned models</i>"]
    T4["<b>velocity</b><br/>dB formula for kick/snare<br/><i>learned models for cymbals</i>"]
    OUT["<b>4. export</b><br/>General MIDI · PPQ 960 · tempo"]
    MIDI(["out.mid"])

    IN -->|"whole song"| EX
    IN -->|"already a drum stem"| STEM
    EX --> STEM
    STEM --> ADT
    STEM --> SEP
    ADT -->|"onset times<br/>+ class"| ART
    SEP -->|"per-drum<br/>audio"| ART
    ART --> T1 & T2 & T3 & T4 --> OUT --> MIDI

    classDef gpu fill:#e8f0fe,stroke:#3f51b5,stroke-width:2px
    classDef cpu fill:#f1f8f2,stroke:#2e7d32,stroke-width:2px
    classDef plain fill:#fafafa,stroke:#9e9e9e
    classDef leaf fill:#f7f8fa,stroke:#d6dbe3
    class SEP,EX gpu
    class ADT cpu
    class ART,OUT plain
    class T1,T2,T3,T4 leaf
```

Blue stages run on the GPU, green on the CPU — measured per stage rather than assumed,
because convolution is 12x faster on the GPU while ADTOF's recurrent layers are 4.1x
slower there. See "Which device runs what".

<details>
<summary>The same diagram as an image (for editors that do not render Mermaid)</summary>

![architecture](docs/architecture.png)

Regenerate with `python make_diagram.py`; it is drawn from code so it cannot drift out
of date silently.
</details>

| Stage | What it produces | Model |
|---|---|---|
| 0 (optional) | whole song → drum stem | htdemucs (Demucs v4) |
| 1 | **what and when** — 5 onset classes | [ADTOF](https://github.com/MZehren/ADTOF) via [ADTOF-pytorch](https://github.com/xavriley/ADTOF-pytorch) |
| 2 | **how loud** — separated stems | MDX23C DrumSep / [LarsNet](https://github.com/polimi-ispl/larsnet) |
| 3 | articulations — tom pitches, open/closed hats, ride vs crash | stem analysis in this repo |
| 4 | tempo + General MIDI export | librosa + pretty_midi |

The two branches matter. **ADTOF looks at the drum mixture, not at the separated
stems** — that is the single most counter-intuitive design decision here, and the
separator's output never reaches the onset detector. It was measured three ways: running
ADTOF on stems scores tom F1 0.000, a handwritten amplitude trigger did no better, and a
purpose-trained CNN reached 0.623 against the pipeline's 0.882. The context of the other
drums turns out to be worth more than isolation.

With `--from-song`, htdemucs runs first and **everything downstream uses its output** —
both the transcriber and the separator see the extracted drum audio, never the original
mix.

The division of labour is not a guess either. **The separator does not improve note
detection at all** (MICRO F1 0.861 with it, 0.862 without). Its entire value is in
velocity and articulation.

### Which note numbers come out

General MIDI allows several numbers for the same drum, so this is a real choice, and a
DAW or sampler has to recognise it. It was settled by counting what hardware emits:
`gm_note_survey.py` over 1150 Groove MIDI performances recorded from a Roland TD-11
(445494 notes).

| drum | we write | what the hardware uses |
|---|---|---|
| kick | **36** | 36 in **100%** of kicks, 35 in none |
| snare | 38 | 38 (82%), 40 (18%) |
| hi-hat | 42 / 44 / 46 closed / pedal / open | 44 (60%), 42 (36%), 46 (4%) |
| toms | 43 / 47 / 50 low / mid / high | 48 (42%), 43 (36%), 45 (13%) |
| crash | 49 | 55 (39%), 57 (31%), 49 (12%) |
| ride | 51 | 51 (85%) |

This corrected an earlier mistake: the kick used to be written as 35, which General MIDI
calls "Acoustic Bass Drum" while 36 is "Bass Drum 1". No performance in the survey used
35, and ReStem 2 Pro writes 36 as well. `--notes 35=35` restores the old numbering.

Toms and crash are left alone: those percentages reflect one Roland's pad assignment,
and 49 ("Crash Cymbal 1") is the conventional choice in software instruments.

Hi-hat articulation is written as **separate notes** (42/44/46) rather than as CC4 pedal
position. Any sampler understands note 46 as an open hi-hat; only some understand CC4.

`note_names.py` prints the full table, including the letter name each DAW gives it —
note 36 is called C1 in Cubase, Logic, Ableton and Reaper, C2 in scientific notation and
C3 in FL Studio, which is why the number is what gets written:

```
 MIDI  Cubase/Logic  scientific  FL Studio  drum            when
   36            C1          C2         C3  Kick            always
   38            D1          D2         D3  Snare           always
   42           F#1         F#2        F#3  Hi-hat closed   default hi-hat
   46           A#1         A#2        A#3  Hi-hat open     when the stem rings out
   44           G#1         G#2        G#3  Hi-hat pedal    --pedal
   43            G1          G2         G3  Tom low         2 or 3 toms resolved
   47            B1          B2         B3  Tom mid         single-tom choice
   50            D2          D3         D4  Tom high        3 toms resolved
   49           C#2         C#3        C#4  Crash           default cymbal
   51           D#2         D#3        D#4  Ride            --separator uvr
```

Everything goes to MIDI channel 10 (index 9). Remap with `--notes 36=35,49=57`, or move
a drum to its own channel with `--channels 36=3`.

---

## Install

```powershell
git clone https://github.com/miraer/drum2midi.git
cd drum2midi
python -m venv .venv
.\.venv\Scripts\python.exe setup_env.py
```

That installs the Python packages, clones ADTOF-pytorch (it is not on PyPI) and then
verifies the result. The MDX23C drum separator downloads itself on the first conversion
(~400 MB).

`models/velocity.pkl` and `models/pedal.pkl` ship with the repository (2.3 MB). They are
scikit-learn pickles trained on the Groove MIDI Dataset by `train_models.py`; retraining
them would mean downloading 4.8 GB of data first. They are loaded by default and the run
says so when they are missing. `--no-learned` ignores them, and because unpickling
executes code, do not point the pipeline at a `models/` directory you did not build or
fetch from this repository.

Optional extras:

```powershell
.\.venv\Scripts\python.exe setup_env.py --with-larsnet   # the fast separator + weights
.\.venv\Scripts\python.exe setup_env.py --with-render    # FluidSynth + GM soundfont
.\.venv\Scripts\python.exe setup_env.py --with-intel-gpu # Intel GPU: separation 12x faster
.\.venv\Scripts\python.exe setup_env.py --check          # verify, install nothing
```

`--check` reports which device each stage will use. If an Intel GPU is present but torch
cannot see it, the installer says so instead of silently running on the CPU.

<details>
<summary>Manual installation</summary>

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

git clone --depth 1 https://github.com/xavriley/ADTOF-pytorch.git
.\.venv\Scripts\python.exe -m pip install -e ./ADTOF-pytorch

# optional: LarsNet weights, only for --separator larsnet
git clone --depth 1 https://github.com/polimi-ispl/larsnet.git
cd larsnet
..\.venv\Scripts\python.exe -m gdown "1U8-5924B1ii1cjv9p0MTPzayb00P4qoL" -O m.zip
Expand-Archive m.zip -DestinationPath . -Force; del m.zip; cd ..
```

</details>

<details>
<summary>Benchmark datasets (only needed to reproduce the measurements)</summary>

```powershell
# MDB Drums — 110 MB instead of 1.3 GB thanks to sparse checkout
git clone --filter=blob:none --no-checkout https://github.com/CarlSouthall/MDBDrums.git mdbdrums
cd mdbdrums
git sparse-checkout init --cone
git sparse-checkout set "MDB Drums/annotations" "MDB Drums/audio/drum_only"
git checkout master; cd ..

# IDMT-SMT-Drums — second, independent test set
curl.exe -sSL -o idmt.zip "https://zenodo.org/records/7544164/files/IDMT-SMT-DRUMS-V2.zip?download=1"

# Groove MIDI — velocity and articulation ground truth
curl.exe -sSL -o groove.zip "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip"
```

The GMD archive contains a macOS artefact named `Icon\r`, an invalid Windows filename
that makes `Expand-Archive` fail silently — extract with a filter that skips it.

</details>

---

## Usage

```powershell
$py = ".\.venv\Scripts\python.exe"

& $py drum2midi.py "drums.wav" -o out.mid              # drum stem in, MIDI out
& $py drum2midi.py "song.mp3"  -o out.mid --from-song  # whole song: extract drums first
& $py drum2midi.py "folder\"   -o "midi\"              # bulk: one MIDI per file
& $py drum2midi.py "drums.wav" --separator larsnet     # ~50x faster, lower quality
& $py drum2midi.py "drums.wav" --no-separate           # fastest, flat velocity 100
& $py drum2midi.py "drums.wav" --quantize 16 --tempo 120
```

Bulk mode reports progress and an ETA, and one bad file does not stop the run — which
matters when a folder takes hours at the default quality.

There is also a native Windows GUI — run `drum2midi.bat`.

![drum2midi GUI](docs/gui.png)

The separator radio buttons state the speed/quality trade-off, the estimated runtime
appears as soon as a file is chosen, and options that do not apply to the selected
separator are greyed out rather than silently ignored. **Folder…** picks a whole folder
instead of one file; the estimate then reads `12 files, 43.1 min of audio → roughly 7.2 h`.

There is a dark theme too, and the kit behind the results is drawn in code rather than
photographed, so it carries no licence and matches whichever theme is on:

![drum2midi GUI, dark theme](docs/gui_dark.png)

Each drum has its own colour and icon, used in the results table and in the logo alike:

![kit icons](docs/kit_icons.png)

The colours run low to high through the kit — kick indigo, snare red, toms green,
hi-hat amber, crash violet, ride teal — so neighbouring drums stay distinguishable
under the common forms of colour blindness, where red/green pairs are the risk.
They are generated by `drum_icons.py`, not stored as hand-made files.

Check the result by ear (original left, transcription right):

```powershell
& $py render_midi.py out.mid --against drums.wav -o ab.wav
```

### Tests

```powershell
& $py test_smoke.py
```

19 self-contained checks that run in under a minute on synthesised audio: MIDI validity,
General MIDI note mapping, PPQ divisibility (triplets must land on whole ticks),
quantization grids, channel and note overrides, argument validation, feature extraction,
the standalone trigger, folder batch mode, that every script answers `--help` instead of
starting a long job, and that the GUI builds and assembles a correct command line.
These catch wiring breakage, not accuracy regressions — accuracy is measured by the
benchmark scripts.

`evaluate.py` sits between the two: it synthesises a drum loop with known ground truth
and scores the whole pipeline end to end, so it verifies that the thing actually works
without downloading a single dataset.

```powershell
& $py evaluate.py
```

### Speed

MDX23C is the default because of quality. On a CPU it is slow; on a GPU it is not:

| separator | CPU | Intel Arc GPU | a 4-minute song |
|---|---|---|---|
| `uvr` (default) | ~15x slower than real time | **~1.9x slower** | 60 min → **~7.5 min** |
| `larsnet` | ~5x faster than real time | unchanged | ~50 s |
| `--no-separate` | ~3x faster than real time | unchanged | ~30 s |

### How many CPU threads

PyTorch starts one thread per physical core. That is correct on a uniform CPU and
counterproductive on Intel's hybrid laptop parts. A Core Ultra 7 165H reports 16 cores,
but only **6 are performance cores** — the other 8 (plus 2 low-power) are efficient
cores running the same work several times slower. A parallel op splits the work evenly
and then joins, so the share that lands on an E-core holds up the whole layer.

Measured with the real ADTOF model, first on a machine that was also running benchmarks,
then again on an idle one:

| threads | 1 | 2 | 4 | **6** | 8 | 12 | 16 | 22 |
|---|---|---|---|---|---|---|---|---|
| busy machine, ms | 5673 | 2099 | — | **2089** | 4097 | 4945 | 5565 | 7941 |
| idle machine, ms | 2607 | 2140 | 1748 | **1553** | 4359 | 4750 | 1967 | 2222 |

6 threads wins both times, but **the size of the win depends on what else the machine is
doing**: 2.66x over torch's default of 16 under load, and 1.27x when idle. The honest
figure to quote is the idle one, 1.27x, because that is the condition a user converting a
single file is in. The earlier 2.66x was measured while a benchmark was running and is
kept here only to show how much contention distorts this kind of measurement.

The shape of the curve is stable across both runs and is the real finding: performance
collapses between 6 and 12 threads, exactly where work starts landing on E-cores.

So `cpu_threads.py` asks Windows for the per-core `EfficiencyClass`
(`GetLogicalProcessorInformationEx`) and uses the count of top-class cores. On a uniform
CPU every core reports the same class, nothing is overridden, and torch's default stands.
Override with `DRUM2MIDI_THREADS=8`.

```powershell
& $py cpu_threads.py      # what was detected
& $py bench_threads.py    # time it yourself
```

These numbers were taken while the machine was also building a training set, so the
absolute values are noisy; the ordering was consistent across repeats.

### Which device runs what

`--device auto` (the default) does not simply prefer the GPU, because the two neural
stages have opposite hardware profiles. Measured on this machine:

| workload | CPU | Intel Arc | |
|---|---|---|---|
| separator-like Conv2d stack | 86.7 ms | 23.8 ms | GPU **3.6x faster** |
| MDX23C over a whole track | 543.5 s | 44.7 s | GPU **12.2x faster** |
| GRU over a long sequence | 225.7 ms | 1378.8 ms | GPU **6.1x slower** |
| ADTOF, the real model | 1408.4 ms | 5731.3 ms | GPU **4.1x slower** |

A recurrent network is a chain of tiny dependent kernels — each step waits for the
previous one, so launch overhead dominates and an integrated GPU loses to the CPU.
Convolution over a spectrogram is the opposite. So the default splits the work:
**separation on the GPU, transcription on the CPU.**

End to end on a 37-second file: 318 s on the CPU, **70 s** with the split. The output is
not approximately the same, it is *the same* — 126 notes, every onset matched, maximum
velocity difference 0, maximum timing difference 0.000 ms (`compare_midi.py`).

```powershell
& $py devices.py          # what this machine offers and what will be used
& $py bench_devices.py    # time each stage on each device yourself
& $py drum2midi.py drums.wav -o out.mid --device cpu   # force one device
```

Intel GPUs need the XPU build of PyTorch, which is not what `pip install torch` gives
you by default:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps torch torchvision `
    --index-url https://download.pytorch.org/whl/xpu
```

CUDA is used for both stages when present, because cuDNN has fused recurrent kernels
that XPU and MPS lack. That follows those libraries' documented behaviour rather than a
measurement here — this machine has no NVIDIA GPU, so treat it as untested.

Transcription is a once-per-song operation, and the quality gap is large
(ride/crash 0.139 → 0.917). If you need speed without a GPU, `--separator larsnet` is
still there. The GUI shows the estimated time, and which device it will use, before you
start.

---

## Accuracy

### MDB Drums — 23 real miked recordings, 7924 hand-annotated onsets

Scored with `mir_eval` at the 50 ms MIREX tolerance.

| Instrument | reference | F1 | Precision | Recall |
|---|---|---|---|---|
| Kick | 1539 | **0.960** | 0.951 | 0.970 |
| Snare | 2654 | 0.844 | 0.821 | 0.868 |
| Toms | 90 | 0.605 | 0.520 | 0.722 |
| Hi-hat | 2639 | **0.892** | 0.860 | 0.928 |
| Cymbals | 1002 | **0.869** | 0.815 | 0.930 |
| **MICRO** | 7924 | **0.882** | 0.853 | 0.914 |

Best tracks: Hendrix 1.00, Disco 0.99, Shadows 0.99. Worst: FreeJazz 0.62, Reggae 0.63 —
brushes and swing are the hardest material.

**About toms.** Only 90 of 7924 onsets in the set are toms — **1.1%**. Toms are rare in
real music, so models barely see them during training, and the measurement itself rests
on a small sample.

### IDMT-SMT-Drums — a second, independent test set

Everything above is tuned and measured on MDB Drums, which risks overfitting to one set.
IDMT-SMT-Drums is a different corpus (real, electronic and hybrid kits, 3 classes only),
all 95 mixes, 7927 annotated onsets:

| Instrument | reference | P | R | **F1** |
|---|---|---|---|---|
| Kick | 2179 | 0.991 | 0.960 | **0.975** |
| Snare | 1499 | 0.711 | 0.993 | 0.828 |
| Hi-hat | 4249 | 0.958 | 0.962 | **0.960** |
| **MICRO** | 7927 | 0.905 | 0.967 | **0.935** |

By recording type, and this is worth stating with its proportions, because they are
lopsided (`idmt_by_kind.py`):

| Subset | Files | Onsets | MICRO | What it is |
|---|---|---|---|---|
| RealDrum | 14 | 1289 | 0.938 | a real kit in a room |
| WaveDrum | 70 | 5589 | 0.937 | built from samples |
| TechnoDrum | 11 | 1049 | 0.922 | a drum machine |

**Three quarters of this corpus is synthetic**, so the obvious worry is that 0.935 is
flattered by easy material. It does not appear to be: real minus synthetic is **+0.003**,
95% CI [−0.023, +0.028] — not significant. The caveat is that RealDrum is 14 files, and
IDMT annotates only kick, snare and hi-hat, so this says nothing about toms or cymbals,
which are the classes we are actually weakest on.

An earlier version of this table reported **0.942** from the first 20 files. Running the
remaining 75 moved it to 0.935, so that subset was representative — unlike the stem
fusion experiment further down, where a small sample was badly misleading.

The score is higher than on MDB because IDMT has no toms or cymbals — the two hardest
classes. [What is actually inside every drum dataset we surveyed](docs/datasets.md),
including two published descriptions that turned out to be wrong.

**An honest caveat about the separator choice.** On this set the two separators are
almost tied: `uvr` 0.942 vs `larsnet` 0.939. The large advantage measured on MDB
(0.882 vs 0.860) comes mostly from toms, cymbals and ghost notes — none of which IDMT
annotates. So `uvr` is the right default for full kits, but if your material is just
kick/snare/hats, `larsnet` costs almost nothing and is 50x faster.

### Groove MIDI Dataset — velocity and articulation

GMD pairs performances with the exact MIDI the drum module produced, so it carries
ground-truth **velocity**, which MDB does not.

| | LarsNet | UVR MDX23C |
|---|---|---|
| MICRO F1 (onsets) | 0.904 | 0.870 |
| velocity r, kick | 0.70 | **0.81** |
| velocity r, snare | 0.81 | **0.99** |
| hi-hat articulation | 0.650 | **0.829** |
| ride vs crash | — | **0.977** |

Velocity correlation is computed **within each track** and then averaged: our velocities
are normalised per track, so pooling hits across tracks measures nothing (it drags kick
from 0.70 down to 0.53).

**Caveat:** GMD audio is Roland module output — samples with no room and **no mic bleed**.
Separation quality looks better here than it is. MICRO 0.904 on GMD vs 0.862 on the
real, miked MDB set is exactly that effect.

---

### Drum stem or whole song?

Both work — `--from-song` runs htdemucs first to pull the drums out of a full mix. MDB
Drums ships the same 23 recordings as isolated drums and as full band mixes with shared
annotations, so the cost of that extra stage can be measured directly:

| Instrument | reference | drum stem | whole song | delta |
|---|---|---|---|---|
| kick | 1539 | 0.960 | 0.923 | −0.037 |
| snare | 2654 | 0.804 | 0.776 | −0.028 |
| **toms** | 90 | 0.573 | **0.328** | **−0.245** |
| hi-hat | 2639 | 0.863 | 0.858 | −0.005 |
| cymbals | 1002 | 0.869 | 0.843 | −0.026 |
| **TOTAL** | 7924 | **0.860** | **0.834** | **−0.026** |

**Feeding a whole song costs about 0.026 F1 overall** — small, and mostly precision
(0.864 → 0.849) plus recall (0.855 → 0.819). Hi-hat barely notices (−0.005).

**Toms are the exception**: they nearly halve, 0.573 → 0.328. They are the quietest and
rarest part of the kit, so whatever htdemucs leaves behind hurts them most.

**The extractor was finally compared, and the choice turned out not to matter.** htdemucs
is what `audio-separator` offers first, and unlike every other component here it was
adopted without measurement — which was uncomfortable, because it sits on the weakest
path in the whole pipeline. The catalogue holds four models that emit a drums stem, and
their published SDR spans 1.5 dB:

| model | drums SDR |
|---|---|
| `htdemucs_ft` | **10.0** |
| `hdemucs_mmi` | 9.6 |
| `htdemucs` (default) | 9.4 |
| `htdemucs_6s` | 8.5 |

SDR is not this project's metric, so `compare_extractors.py` ran each of them over all 23
MDB full mixes, transcribed the result and scored the MIDI against the same annotations.
`score_extractors.py` reports it, and will score a run that was interrupted:

| extractor | MICRO F1 | 95% CI (bootstrap over tracks) |
|---|---|---|
| `hdemucs_mmi` | **0.8363** | [0.7764, 0.8927] |
| `htdemucs_ft` | 0.8339 | [0.7705, 0.8922] |
| `htdemucs` (default) | 0.8308 | [0.7654, 0.8920] |

**None of the gaps survive contact with statistics.** Resampling the 23 tracks 4000 times,
every pairwise difference straddles zero: htdemucs vs `hdemucs_mmi` is −0.0055 with a 95%
interval of [−0.0125, +0.0019]. The nominal ranking also disagrees with SDR, which put
`htdemucs_ft` first and `hdemucs_mmi` second — another reminder that separation quality
and transcription accuracy are not the same axis.

So the default stays `htdemucs`, now for a stated reason rather than by accident: nothing
measurably beats it. `--extractor hdemucs_mmi.yaml` selects another if you want to try.

The wider lesson is about this test set rather than about Demucs. A single extractor's
interval spans **±0.06**, so 23 recordings cannot resolve differences smaller than about
one point of F1 — and the individual claim being tested here is a tenth of that.

It also costs time: the 23 tracks took 12 minutes as stems and 111 minutes as songs,
because every file goes through the extractor first.

#### On a real commercial mix

MDB's "full mix" files are short excerpts, not finished productions, so the number above
could be optimistic. One 4½-minute song was available both as the finished stereo mix and
as the drum stem it was bounced from — the same audio, so the two runs are directly
comparable. There are no hand annotations here, so this measures **agreement with the
stem run**, not absolute accuracy:

| instrument | stem | song | recall | precision |
|---|---|---|---|---|
| kick | 270 | 257 | **0.941** | 0.988 |
| snare | 318 | 272 | 0.796 | 0.930 |
| hi-hat closed | 111 | 122 | 0.820 | 0.746 |
| **ride** | 286 | 175 | **0.598** | 0.977 |
| **crash** | 88 | 159 | 0.886 | **0.491** |
| toms (all) | 57 | 50 | 0.561 | 0.640 |
| **TOTAL** | 1154 | 1073 | 0.779 | 0.838 |

Agreement F1 **0.807**. Timing survives well (mean deviation 2.0 ms), velocity less so
(mean 7.8 of 127).

The interesting part is *where* it breaks. Kick is nearly untouched. What collapses is
the **ride/crash distinction**: ride drops to 0.598 recall while crash gains 71 notes it
should not have, and the two counts nearly cancel (374 → 334). Cymbals are broadband and
overlap with vocals and guitars, so htdemucs alters their spectrum most — and ride vs
crash is decided spectrally. Toms blur into each other the same way (low tom 0.286
recall, mid tom 0.348 precision).

**So: for kick and snare a full mix is fine. If cymbal articulation matters, feed a drum
stem.** This is the same conclusion MDB gave, but the damage is concentrated differently
— in toms there, in cymbals here.

So: if you have an isolated drum track, use it. If you only have the song, the result is
still usable, but expect tom fills to suffer.

## Comparison with ReStem 2 Pro

As far as I know there are no published accuracy figures for ReStem, so this is the first
symmetric measurement. Both systems were run over the same 23 MDB tracks and scored by
the same code (`compare_with_restem.py`).

**Read this section as what it is: one side's measurement of the other side's product.**
ReStem's developers had no part in it, have not reviewed it, and had no opportunity to
object before publication. Everything below was learned from the outside — from its
exports, its `trigger_events.json`, its interface and its published user's guide — by
people with an obvious stake in the answer. Three specific reasons to hold it loosely:

- **Articulation thresholds were left at their defaults.** ReStem's hi-hat and tom
  classification boundaries are user-adjustable, and the vendor documents adjusting them
  when articulations land on the wrong note. We adjusted nothing, which is a fair
  defaults-against-defaults comparison but is not a measurement of the model's ceiling.
- **The settings were reconstructed, not recorded.** The quality mode in use was
  established after the fact by matching exports event for event, on two tracks. The
  vendor documents Better and Best as different models; we observed identical MIDI from
  both on those two tracks, which is an observation about those tracks, not a general
  claim.
- **Anything to do with speed or memory is about this machine, not their software.** An
  Intel Arc system with no NVIDIA GPU is not the path their documentation points at, so
  no timing figure from here is quoted as a comparison.

What can be stated without qualification: nothing was extracted from their model files,
and nothing of theirs is redistributed. The ReStem team have been sent these results and
any correction they send gets applied here and labelled as a correction.

Method: ReStem 2.0.18 on trial, driven through UI automation (`restem_ui.ps1`,
`restem_batch.ps1`). No weights were extracted — only the product's normal output.

**Which quality mode, and why it does not matter.** ReStem offers Better (Offline) and
Best (Offline), the latter with an optional Bleed Reduction. The runs above used Better,
and that was not recorded at the time — the export does not carry the mode, so it had to
be established afterwards by re-transcribing tracks and matching the exports event for
event (`restem_one.ps1`, `compare_restem_runs.py`).

| | MusicDelta_Rock_Drum | MusicDelta_Beatles_Drum |
|---|---|---|
| reference toms | 0 | **30**, the most in MDB |
| Better (Offline) | 0.500 | 0.931 |
| Best (Offline) | 0.500, byte-identical | 0.931, byte-identical |
| Best + Bleed Reduction | 0.466 | 0.915 |

Better and Best produce **identical MIDI**, on a track with no toms and on the track with
the most. Only Bleed Reduction changes anything, and it loses on both: on Rock it emits 7
toms where the reference has none; on Beatles it gains 0.009 on toms and loses 0.011 on
snare. That is two tracks, chosen for contrast rather than at random, so it is evidence
that the comparison ran at or near ReStem's best — not a proof that the two modes are
identical in general.

Two caveats on the machine rather than the software. Their offline renderer wants around
5 GB for a 36-second track and swaps badly below that. And on this Intel Arc system it
computes through Vulkan; forcing it off — which was tried here on the strength of an older
note about a Vulkan crash — quadruples its memory to 20 GB and destroys its speed. Neither
affects accuracy, but both make timing figures from this machine untrustworthy.

Its engine writes `trigger_events.json`, which is the raw **TrigNet** output:

```json
{"format": "trignet-events-v1", "sr": 44100, "hop": 441,
 "stems": {"kick": [{"sample": 433, "prob": 0.928, "vel": 0.8968}],
           "hh":   [{"sample": 787, "...": "...", "cc4": -0.0475}],
           "toms": [{"sample": 181292, "tomClass": 2, "fundHz": 196.0}]}}
```

Two things are visible there: TrigNet runs at **the same 100 frames per second** as ADTOF,
and it classifies toms **by fundamental pitch** — the same approach as our `split_toms`.

### Where each side wins

Stated the way the evidence supports, rather than by reading off the bigger number:

**drum2midi is measurably better** at the hi-hat (0.892 vs 0.754, interval
[+0.062, +0.249]) and therefore overall (0.882 vs 0.820, [+0.033, +0.097]). The overall
win is essentially the hi-hat win.

> **⚠ The hi-hat result depends on a setting we left at its default.**
> ReStem classifies hi-hats by how long each hit rings, converting decay to an
> "openness" value and splitting it at documented boundaries: below 0.465 closed,
> 0.465–0.798 pedal, above 0.798 open. Every one of the 188 CC4 values in our exports
> sat below the first boundary, so everything was labelled closed and no note 46 or 44
> was ever written.
>
> **Those boundaries are user-adjustable**, through a trigger-range editor the vendor
> documents and recommends when articulations land on the wrong note. We did not touch
> them — both systems ran at their defaults, which is the comparison we intended, but it
> means this row measures ReStem's default thresholds against our material rather than
> the limit of what its model can do. A user tuning the split per song would likely
> recover open hi-hats that we recorded as closed.
>
> Two things that are *not* wrong with it, both checked against the vendor's own
> documentation after an earlier version of this section guessed otherwise: the offline
> export is the path they recommend for accuracy over their live MIDI port, and note 60
> is documented as the "Other" stem's neutral fallback, which is how we read it.

**Nothing else separates them on this test set.** Kick, snare, cymbals and toms all have
intervals that cross zero. The tom row is the starkest: the nominal 0.605 vs 0.699 looks
like a clear loss for us, but with only 90 tom onsets in the whole corpus the interval is
[−0.304, +0.139]. We do not know who is better at toms, and neither does anyone else
using MDB alone.

This correction was made after the fact. An earlier version of this README claimed
ReStem was better at toms, ghost notes and ride/crash, and us at kick — all read straight
off the point estimates. Three of those four claims do not survive a bootstrap.

A fifth claim was withdrawn the same day it was published, and it was a caveat rather
than a boast. This section briefly warned that our hi-hat figure might be measuring
ReStem's offline exporter rather than its model, since its live MIDI port was untested.
Their user's guide says the opposite in three separate places: the offline export uses a
more thorough detector than the live path and is what they recommend when transcription
accuracy matters. We had guessed at a mechanism instead of reading the manual, and the
guess happened to be flattering to us — it framed our largest win as possibly unearned,
which reads as modesty while resting on nothing. The real caveat, which the same document
supplied, is the adjustable threshold above.

Two of our reverse-engineered findings were confirmed by that document. The articulation
mapping we recovered blind from the exports — closed below ≈0.46, open above ≈0.76 —
matches the documented 0.465 and 0.798. And note 60, which we read as "detected but
unclassified", is documented as the Other stem's neutral fallback.

**Where those claims came from is worth stating.** This project was built largely by an AI
coding agent (GitHub Copilot) working under direction, and the retractions above fall into
two kinds, both worth naming.

Four were the agent reading a point estimate as though it were a result — confidently,
fluently, and wrongly. It never invented a number, and it never hedged one either. What
caught those was a standing rule that no difference is real until `significance.py` has
resampled the tracks around it.

The fifth was a different failure and a harder one to guard against: inventing a plausible
mechanism rather than looking for the documentation that already described it. The
vendor's user's guide answered the question directly, had anyone gone to find it, and the
invented explanation survived for a few hours purely because it sounded careful. A rule
about confidence intervals does nothing against that one. What worked was a person asking
whether the manual had been read.

The retractions are left in the text rather than quietly edited out, because a write-up
that shows only the surviving claims tells you nothing about how hard they were tested.

The strategies do differ, and that part is visible in the aggregate: ReStem leans towards
recall (P 0.764 / R 0.884), we sit closer to balanced (P 0.853 / R 0.914).

### Where ReStem breaks

TrigNet sometimes matches the annotation exactly — Disco: kick 118/118, snare 147/147.
But it **loses the hi-hat entirely on 3 of 23 tracks** (Hendrix, Reggae, Rock), which is
the same "separator returned an empty stem" failure we handle explicitly, and it
**over-fires on jazz**: 212 hi-hat events against 7 in the reference on FreeJazz.

### Pedal hi-hat: nobody solves it

ReStem advertises "Closed, Pedal, or Open". It does emit note 44 — the MIDI panel maps
CLOSED F#1 / PEDAL G#1 / OPEN A#1, and its rule recovered from the export is
`cc4 < 0.46` → closed, `0.46…0.76` → pedal, `≥ 0.76` → open.

But in practice:

| | articulation accuracy | closed | open | **pedal** |
|---|---|---|---|---|
| drum2midi | **0.754** | 1495/1747 | 156/238 | **19/230** |
| ReStem | 0.691 | 1591/1595 | 37/251 | **3/513** |

Both fail on pedal. Our own measurement independently agrees: across 1102 annotated
hi-hat hits the pedal chick is not separable from a closed hat in the stem (peak 0.0457
vs 0.0466, centroid 11993 vs 12454, d' between 0.01 and 0.48).

### Caveats

1. No NVIDIA GPU on the test machine — ReStem's engine crashed on Vulkan (Intel Arc,
   Error 106) and had to be forced onto D3D12. This should not affect accuracy.
2. Trial version. Per the vendor, the trial adds a quiet tone to **audio**, not MIDI.
3. MDB files are **mono**; ReStem expects stereo, which may handicap it slightly.
4. Where a threshold had to be chosen for ReStem, it was tuned in ReStem's favour.

---

## Design decisions, and the ideas that failed

This section exists because most of the tuning effort produced negative results, and
those are as useful as the positive ones.

### ✅ Choosing the separator by measurement, not by speed

The default used to be LarsNet because it is fast. Measured on MDB:

| | larsnet | **uvr (MDX23C)** |
|---|---|---|
| MICRO F1 | 0.860 | **0.882** |
| ghost notes | 0.468 | **0.718** |
| ride vs crash | 0.139 | **0.917** |

Ghost-note recall improved by half. I had previously concluded that ghost notes were a
hard limit of the ADTOF model — that was wrong; it was a dirty stem.

### ✅ Adaptive tom threshold

A fixed peak-picking threshold cannot serve toms, whose density varies wildly between
tracks. Using the **98.5th percentile of each track's own tom activations** raises
held-out tom F1 from 0.322 to 0.472 and MICRO from 0.850 to 0.858. The same trick hurts
dense classes (hi-hat 0.863 → 0.802), so it is applied to toms only.

### ✅ Tom splitting was correct but too timid

`split_toms` groups tom hits by fundamental pitch. Validated against MDB subclass labels:
**when it splits, the pitch order is right 100% of the time** (36/36). The problem was
coverage — the original gate refused to split 75% of tom pairs. Loosening it from 1.18 to
1.10 raised coverage from 25% to 34% with no loss of accuracy.

### ✅ Learned velocity — but only for two drums

| | formula | learned model |
|---|---|---|
| kick | **0.843** | 0.701 |
| snare | **0.881** | 0.815 |
| hi-hat | 0.807 | **0.817** |
| cymbals | 0.721 | **0.802** |

The handcrafted "dB below this drum's loudest hit" formula **beats** gradient boosting on
kick and snare: it is the right inductive bias, while the model latches onto absolute
levels that do not transfer between kits. `train_models.py` keeps only the models that
won.

### ✅ Splitting the work between GPU and CPU (and reversing an earlier "no")

I first concluded that the Intel GPU was not worth using, because ADTOF's recurrent layers
measured 4.5x *slower* on it and `audio-separator` had no XPU support. Both observations
were correct; the conclusion was wrong. "No support" turned out to be a four-line device
check, and the slow component was 2% of the runtime. Splitting by stage — separation on
the GPU, transcription on the CPU — takes a 37-second file from **318 s to 70 s** with
byte-identical MIDI. See "Which device runs what". The lesson is the same one as with
stem contrast: a measurement of a part is not a measurement of the whole.

### ❌ Replacing MDX23C with the Inverse Drum Machine for velocity

[IDM](https://github.com/bernardo-torres/inverse-drum-machine) (IEEE TASLP 2026) is
attractive on paper: **Apache-2.0**, 2.4 MB of weights against MDX23C's 417 MB, joint
separation and transcription, and it accepts an external transcription — exactly the
shape of this pipeline, which already knows the onsets and only needs loudness.

Scored against GMD's recorded velocities, correlated per track (`compare_idm.py`):

| | IDM | MDX23C |
|---|---|---|
| kick | 0.626 | **0.878** |
| snare | 0.639 | **0.797** |
| hi-hat | 0.714 | **0.726** |
| cymbals | 0.072 | **0.212** |
| time, 10 tracks | **52 s** | 597 s |

It is 11x faster and loses on every instrument. Worth recording that after two tracks
IDM led on hi-hat, 0.806 against 0.773; by ten tracks that had evaporated. Small samples
mislead, which this project has now demonstrated three separate times.

The author is candid about the limitation: the model is trained on StemGMD, built from
Logic Pro X samples, and warns that material far from that distribution separates badly.
MDB and GMD recordings are exactly that.

Three code-level obstacles had to be cleared first, all worth knowing if you try it:

* `separate()` applies `torch.from_numpy()` to both branches of its input check, so
  passing a tensor — the documented usage — raises immediately;
* the model's first convolution takes one channel, while their own reader returns stereo;
* `get_conditioning_vector()` declares `device: torch.device = torch.device("cuda")` as
  a default argument and the caller never overrides it, so inference fails outright on
  any machine without CUDA.

### ❌ Replacing ADTOF with ADT_STR (2026), despite its licence being better

[ADT_STR](https://github.com/pier-maker92/ADT_STR) (arXiv 2601.09520) is the first
serious candidate to appear in years: a 69M-parameter seq2seq transformer with 26 output
classes, published weights, and **CC BY-SA 4.0** — which would remove the non-commercial
restriction that ADTOF imposes on this pipeline. Its paper reports tom F1 0.77 on MDB
against our 0.605.

Run on the same 23 tracks and scored with the same code (`run_adt_str.py`,
`score_adt_str.py`):

| class | reference | ADT_STR | ours |
|---|---|---|---|
| kick | 1539 | 0.899 | **0.960** |
| snare | 2654 | 0.785 | **0.844** |
| hi-hat | 2639 | 0.620 | **0.892** |
| toms | 90 | 0.140 | **0.605** |
| cymbals | 1002 | 0.041 | **0.869** |
| **MICRO** | 7924 | **0.673** | **0.882** |

Before concluding anything, the obvious question: was it run correctly?
`verify_adt_str.py` answers it by reproducing their own published per-class numbers.
Kick comes out at 0.910 against their 0.92 and snare at 0.805 against their 0.85 — the
two classes they score highest on reproduce within 0.05. Toms (0.145 vs 0.77) and
cymbals (0.042 vs 0.52) do not. A setup error would depress every class, not two.

So the model runs correctly and simply does not transfer to this material. It was
trained on synthetic audio built from Lakh MIDI and one-shot samples, and the classes
that survive that are the ones with unambiguous transients.

Two things were needed to run it at all, both the same shape as bugs found elsewhere:

* `torchaudio` 2.11 delegates decoding to torchcodec, so audio loading was redirected to
  soundfile from outside their tree (`ADT_STR/torchaudio_shim.py`).
* their `select_inference_device()` is `if cuda / elif mps / else cpu` — the identical
  pattern to audio-separator's. On CPU the autoregressive decode burned six hours of CPU
  time for two tracks; patched onto the Intel GPU it does 23 tracks in 93 minutes.

Worth revisiting when they publish weights trained on real recordings.

### ❌ A learned onset detector on separated stems

This was the last big untested idea, and the one the whole 417 MB training set was built
for: since a stem already tells you *which* drum it is, finding *when* it was hit should
be easier than transcribing a mixture. Running ADTOF on stems failed (tom F1 0.000) and
a handwritten amplitude trigger failed, so this trained a small CNN on exactly what the
separator produces — 302 Groove MIDI performances, MDX23C stems, onset targets from the
module's own MIDI, split by performance (`build_onset_dataset.py`, `train_onset.py`).

The first numbers looked excellent:

| stem | balanced windows | **whole tracks** |
|---|---|---|
| kick | 0.968 | **0.623** |
| snare | 0.876 | **0.504** |
| toms | 0.942 | **0.581** |
| hi-hat | 0.108 | 0.321 |
| cymbals | 0.002 | 0.002 |

The left column is what training reports, and it is a mirage. Onsets are 1.5% of frames
in real audio but 25% of a balanced training sample, so precision measured on the
balanced set counts false positives against a pool thirty times too small. Evaluating
every frame of the held-out tracks costs kick 0.345 F1 and snare 0.372.

Against the pipeline's actual 0.882, even the best stem is far behind. Peak-picking
would recover something, but not that gap. **ADTOF looking at the mixture beats a
detector that was handed the isolated instrument** — the context of the other drums
turns out to be worth more than the isolation.

Cost: 3.9 hours to build the dataset (would have been 23 without the GPU) and 20 minutes
to train. Worth it to close a question that had been open since the beginning.

### ❌ A fallback for the passages where ADTOF goes silent

A real song produced a passage the pipeline left completely empty — 24 seconds, 70
audible onsets, nothing written. Not a threshold problem: ADTOF's activations there peak
at 0.008 to 0.08 against thresholds of 0.14 to 0.32, which is noise rather than a near
miss. The material is percussive (harmonic fraction 0.101, stable pitch in 4.6% of
frames) with a dull spectrum, centroid 2129 Hz against 4037 Hz in the loud sections.
ReStem does not recognise it either: 73 of the 106 notes it emits there are pitch 60,
its unclassified bucket.

That suggested a narrow fallback — emit onsets only where the model is silent. Narrow is
the operative word, because a stem-based onset detector had already been rejected above
for firing everywhere.

`blind_spots.py` measures how much of a recording falls in that hole: peak-pick the
audio's own onset envelope, then ask whether any class reacted within 60 ms. A peak
counts as **blind** when every class stays under *half* its threshold.

Across all 23 MDB tracks (`blind_spots_mdb.py`):

| | |
|---|---|
| blind onsets | 101 of 5081 detected, **1.99%** |
| tracks at exactly zero | **15 of 23** |
| concentration | 95 of the 101 sit in three tracks |
| worst track | SwingJazz 19.3%, above the 18.2% of the real song |
| correlation with per-track F1 | −0.288, 95% CI **[−0.605, +0.143]** |

**The correlation decided nothing, and should not have been asked to.** With 15 tracks
pinned at zero there is no gradient for it to read — three tracks against twenty, with
single-event noise between. The criteria for this experiment were written as though a
correlation would come out decisive; that was a mistake in the experimental design, and
reporting −0.288 as "near zero, idea dropped" would have been choosing the reading after
seeing the number.

**What decides it is where the failures actually are.** The worst-scoring tracks are not
blind:

| track | F1 | blind share |
|---|---|---|
| Reggae | 0.631 | **0.0%** (0 of 54) |
| LatinJazz | 0.668 | **0.0%** (0 of 359) |
| FreeJazz | 0.731 | 0.5% |
| SwingJazz | 0.744 | 19.3% |
| BebopJazz | 0.758 | **0.0%** (0 of 413) |

Three of the five worst tracks contain not one blind onset. The model sees those hits,
reacts to them, and still gets them wrong — wrong class, wrong time, or spurious. A
fallback that fires only into silence is **inert on exactly the recordings that need
help**. The ceiling agrees: granting that every blind peak is a real missed onset and
that a fallback recovers all of them without a single false positive, 101 of 7924
annotated onsets is a **1.27%** bound on recall.

The inversion is worth naming, because it was the argument that kept the idea alive.
"It would not fire at all on four of the six tracks measured" was offered as evidence of
safety. Measured across 23 tracks, that is inertness on 20 of them, and specifically on
the ones that score worst. The property that made it seem harmless is the property that
makes it useless.

Two smaller findings from the same measurement. The blind column was computed with the
fixed tom threshold while `transcribe()` substitutes an adaptive one, which biases the
share upward — recomputing both ways moves the mean from 1.823% to 1.703% and changes
three tracks, so the column is a slight overestimate but not an artefact. And the figures
reproduced to the digit on a second machine running **librosa 0.11.0 against 1.0.0 here**
— a major version apart in the library that computes the onset envelope, which is
stronger evidence that the measurement is a property of the audio than matched versions
would have been.

### ❌ A learned ride/crash classifier, and a tuned margin

The largest remaining gap against ReStem that is our own doing rather than a data
shortage: ride vs crash, 0.917 against 0.953. ADTOF emits one cymbal class and the
pipeline splits it with a single rule — "the ride stem peaks 1.2x above the crash stem".
MDX23C provides separate ride and crash stems, so a model should be able to do better.

Thirteen features were extracted per hit (peaks, decay times, spectral centroid,
flatness and rolloff of both stems, and their ratios) and three model families tried,
cross-validated with whole tracks held out:

| | accuracy | balanced | crash | ride |
|---|---|---|---|---|
| **heuristic, margin 1.2** | **0.917** | **0.855** | 0.77 | 0.94 |
| gradient boosting | 0.906 | 0.797 | 0.65 | 0.95 |
| decision tree | 0.895 | 0.813 | 0.70 | 0.93 |
| logistic regression | 0.800 | 0.793 | 0.78 | 0.80 |

All three lost. Tuning the margin instead looked promising — 1.0 scores 0.925 on the
full set — but choosing it inside each training fold and scoring on the held-out one
gives **0.904**, below the stock 1.2, and the folds disagreed (1.0, 1.0, 4.0, 1.0, 4.0).
The same trap as the peak-picking thresholds, caught the same way.

Why it fails is informative: the separator has already done the work. Splitting by tembre
is exactly what MDX23C was trained for, so the loudness ratio of its two stems carries
nearly all of the available signal, and the extra spectral features describe the same
thing again. With 130 crashes across 17 tracks, a model learns individual cymbals rather
than the difference between cymbal types. `train_ride.py` reproduces all of this.

### ❌ Tuning all five thresholds globally (my earlier result was biased)

I originally tuned thresholds on all 23 tracks and reported on those same 23 tracks.
Honest 2-fold cross-validation: tuned thresholds score **0.848** against ADTOF's stock
**0.850** — no gain at all, and toms got worse (0.322 → 0.265) with the per-fold choices
disagreeing (0.45 vs 0.32). Thresholds were reverted to stock except snare.

### ❌ A learned pedal hi-hat classifier does not transfer

On held-out GMD tracks it reached 0.787 balanced accuracy and looked like a win. On real
miked recordings it calls **721 of 1747 closed hats "pedal"** and drops overall hi-hat
accuracy to 0.502, below the heuristic's 0.752. Cause: it was trained on Roland modules
with no room and no bleed. `--pedal` now defaults to `heuristic`.

### ❌ Running ADTOF on isolated stems

The idea was to feed the tom stem to ADTOF, since toms fail on the mix. Result: tom F1
**0.000**. ADTOF is trained on mixtures, and on a bare stem the tom class never fires.
This is why `rescue_toms` takes timing and loudness from the stem, but **not**
classification.

### ❌ Writing our own trigger to replace ADTOF

ReStem's TrigNet suggested that a dedicated amplitude-domain trigger should work on a
clean stem. Implemented one (`trigger.py`): peak envelope, derivative of the **log**
envelope, threshold adaptive to the local level. It does see ghost notes better
(0.771 vs 0.467) but at ruinous precision (0.261), and ADTOF dominates at every operating
point. Fusing the two never beat ADTOF alone either.

### ❌ CC4 (continuous pedal position)

ReStem emits a CC4 controller ahead of each hi-hat note. GMD has this annotation (22197
messages), but a regressor on hi-hat stem features only reached **r = 0.328**, RMSE 21.25
against 22.83 for simply writing the median. Shipping a controller that weak would make a
sampler open and close the hats almost at random, so it is not shipped.

### ❌ Narrowing the peak-picker's combine window

4.9% of annotated hit pairs in GMD fall inside the 20 ms combine window, so flams should
be getting merged. Narrowing it to 7 ms changes **nothing**: 7823 notes vs 7825, F1
identical to three decimals. The limiting factor is elsewhere.

### ⚠️ Stem onset fusion helps, but unevenly

Adding onsets found in clean stems raises MICRO from 0.861 to 0.882, mostly via snare
(+0.039) and hi-hat (+0.029). But it helped 4 tracks, hurt 9 and left 10 unchanged, and
the gain correlates **−0.647** with how well the mix pass already did. Nearly all of it
comes from two jazz tracks where the mix pass was failing. It is on by default for `uvr`
because the worst case is −0.014 and the best is +0.246; `--fuse-stem-onsets off` disables it.

### ⚠️ `rescue_toms` is a LarsNet crutch

It takes timing from the tom stem and steals hits from kick and snare. With MDX23C's
clean tom stem (252x contrast) it only adds false positives: toms 0.605 → 0.398, MICRO
0.882 → 0.876. It is now enabled automatically only for LarsNet.

---

## Reproducing the measurements

Every number in this README comes from a script here. To check them yourself:

```powershell
$py = ".\.venv\Scripts\python.exe"

# 1. Onsets on real miked recordings (needs MDB Drums)
& $py benchmark_mdb.py

# 2. Independent second test set (needs IDMT-SMT-Drums), all 95 files
& $py benchmark_idmt.py

# 3. Velocity and articulation (needs Groove MIDI)
& $py benchmark_gmd.py --limit 14

# 4. Articulation detail against subclass labels
& $py validate_articulations.py

# 5. Are the thresholds overfitted? 2-fold cross-validation
& $py sweep_thresholds.py          # builds the activation cache first
& $py validate_thresholds.py

# 6. What a whole song costs versus its own drum stem, on your own material
& $py drum2midi.py "drums.wav" -o stem.mid
& $py drum2midi.py "song.mp3"  -o song.mid --from-song
& $py compare_midi.py stem.mid song.mid 30

# 7. Which device should run which stage, on your hardware
& $py bench_devices.py
```

The MDX23C stems are the expensive part. Cache them once and every later run is fast:

```powershell
& $py cache_uvr_stems.py           # 23 tracks; ~75 min on a CPU, ~8 min on a GPU
```

To reproduce the ReStem comparison you need a ReStem 2 licence or trial. The automation
scripts (`restem_ui.ps1`, `restem_enable_midi.ps1`, `restem_batch.ps1`) drive its UI,
collect `trigger_events.json` per track, and `restem_to_midi.py` converts those to MIDI.
Nothing is extracted from the product's model files — only its normal output is used.

## Troubleshooting

**The first conversion takes forever.** On a CPU, MDX23C runs about 15x slower than real
time, so a 4-minute song is roughly an hour, and the model itself downloads (~400 MB) on
first use. On an Intel or NVIDIA GPU it is about 2x slower than real time — see
"Which device runs what". The GUI shows an estimate, and the device, before you start.
Use `--separator larsnet` if you need speed without a GPU.

**`WARNING: separator returned an empty stem for toms`.** The separator produced silence
instead of an instrument. The pipeline detects this and reads velocity from the mix
instead, so the output is still usable, just without articulation detail for that drum.
This is not specific to us: a reviewer measured the same failure on the commercial
ReStem 2, and our own measurements show it loses the hi-hat entirely on 3 of 23 tracks.

**Toms come out as kick or snare.** ADTOF's tom class is its weakest — only 1.1% of
annotated onsets in MDB Drums are toms, so models barely see them. `--rescue-toms on`
recovers some of them from the tom stem, but only helps when that stem is dirty
(LarsNet). On MDX23C it adds more false positives than it finds.

**Quantization ruins the groove.** The grid is derived from the tempo, and automatic
tempo detection frequently lands on double or half the real value. Pass `--tempo`
explicitly whenever you know it.

**Triplets land off the grid in my DAW.** They should not — the writer uses PPQ 960,
which is divisible by 3. If you see this, check that your DAW is not re-quantizing on
import. `python test_smoke.py` verifies this end to end.

**`ModuleNotFoundError: No module named 'torchcodec'`.** torchaudio 2.11 removed its own
audio backends. The pipeline never loads files through torchaudio, so this only appears
if something else imports it — reinstall with `pip install -r requirements.txt`.

**`python -m audio_separator...` does nothing.** That module has no `__main__` guard. Use
the `audio-separator` console script instead.

**Everything is slow and the fan is loud.** Check `python devices.py`. If it reports no
GPU, see the install notes above — an Intel GPU needs the XPU build of torch.

**Can an integrated Intel Arc GPU help? — yes, and my first answer was wrong.**

I originally measured individual operations, found that ADTOF's GRU ran 4.5x *slower* on
the GPU, noted that `audio-separator` contains no mention of XPU, and concluded the GPU
was not worth using. Two of those three facts were right. The conclusion was not.

What the reasoning missed:

1. **"The library has no XPU support" is not the same as "the library cannot be given
   XPU support."** Device selection there is one `if cuda / elif mps / else cpu`. The
   hot path already goes through `self.torch_device`, there is a working precedent for a
   third backend (DirectML), and `device_utils` *probes* whether a device can run the
   complex STFT/iSTFT the models need, falling back on its own. Teaching it XPU is four
   lines (`xpu_separate.py`), applied as a monkey patch so the package stays upgradable.
2. **DirectML's failure was about the allocator, not the model.** Upstream had already
   fixed every code-level incompatibility in these models — state-dict loading, complex
   ops, SDPA, rotary embedding, einsum — and stopped only because torch-directml's
   allocator could not sustain chunked inference. XPU is an in-tree backend with the
   normal PyTorch allocator, so that specific blocker does not apply.
3. **One slow component does not make the whole pipeline slow.** The GRU really is
   slower on the GPU — which is why `--device auto` leaves transcription on the CPU and
   moves only separation. Measuring per stage is what turns "no" into "12x".

Measured after the patch:

| workload | CPU | Arc XPU | |
|---|---|---|---|
| Conv2d stack (separator-like) | 86.7 ms | 23.8 ms | **3.6x faster** |
| **MDX23C, whole track** | **543.5 s** | **44.7 s** | **12.2x faster** |
| GRU over a long sequence | 225.7 ms | 1378.8 ms | 6.1x slower |
| ADTOF, the real model | 1408.4 ms | 5731.3 ms | 4.1x slower |

End to end, 37 s of audio: **318 s → 70 s**, with byte-identical MIDI (126 notes, zero
velocity difference, zero timing difference — `compare_midi.py`).

Installation is still fragile: the XPU build pins exact Intel runtime versions
(`intel-sycl-rt==2026.1.0` and friends), and installing `torchaudio` or `torchvision`
from PyPI afterwards **silently replaces the XPU build with the CPU one** — which is why
`setup_env.py --with-intel-gpu` reinstalls them from the XPU index with `--no-deps`.

**Check what is actually installed:**

```powershell
python setup_env.py --check
```

## Limitations

1. **Speed.** MDX23C runs ~15x slower than real time on a CPU; ~1.9x on an Intel GPU.
2. **Toms** — F1 0.605, and measured on only 90 annotated onsets.
3. **Pedal hi-hat** — 19 of 230. Not solved by anyone, including ReStem (3 of 513).
4. **Ghost notes** — 0.718 vs ReStem's 0.747.
5. **Empty stems.** A separator can return silence instead of an instrument. The pipeline
   detects this, warns, and falls back to reading velocity from the mix.
6. **Jazz** — brushes and swing remain the hardest material.
7. **Learned velocity models** were trained on GMD's electronic kits. Transfer is verified
   for cymbals and hi-hat, and failed for pedal. Disable with `--no-learned`.

---

## Licensing — read before using commercially

The code in this repository is MIT. **The models it depends on are not.** The same table
lives in [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md), so it travels with the code.

MIT on this repository is not a claim that the assembled pipeline is free for commercial
use, and it cannot be: nobody can relicense somebody else's weights by wrapping them.
What it means is narrower and worth stating plainly:

- **MIT covers this code and the two files in `models/`.** Those two are trained by
  `train_models.py` on GMD, which is CC BY 4.0, so they carry no non-commercial terms.
- **The NC models are not in this repository.** The install steps download them from
  their authors, whose licences reach you directly and unchanged.
- **With the default models, the assembled pipeline is non-commercial**, because the
  strictest licence in the set governs the combination. ADTOF and the LarsNet weights
  are CC BY-NC.

| Component | License | Commercial use |
|---|---|---|
| ADTOF | CC BY-NC-SA 4.0 | ❌ |
| ADTOF-pytorch | **no license file** | ❌ all rights reserved by default |
| LarsNet weights | CC BY-NC 4.0 | ❌ |
| MDX23C DrumSep | not stated | ⚠️ unclear |
| MDB Drums (benchmark) | CC BY-NC-SA 4.0 | ❌ |
| IDMT-SMT-Drums (benchmark) | CC BY-NC-ND 4.0 | ❌ |
| Demucs, audio-separator, DrumSep code | MIT | ✅ |
| GMD / E-GMD / StemGMD (datasets) | CC BY 4.0 | ✅ |

A non-commercial licence in Creative Commons terms restricts **use**, not just
redistribution — running an NC model on your own server and charging users is exactly the
commercial use that is prohibited. A SaaS wrapper does not work around it.

A clean commercial path exists, because the **training data is free**: GMD, E-GMD and
StemGMD are all CC BY 4.0, and Demucs is MIT. It requires training a separator and a
transcriber from scratch.

---

## Scripts

| | |
|---|---|
| `drum2midi.py` | the pipeline |
| `drum2midi_gui.pyw` | native Windows GUI (`drum2midi.bat`) |
| `setup_env.py` | one-shot install and `--check` diagnostics |
| `devices.py` | which device runs which stage, and why |
| `note_names.py` | the note map, with each DAW's letter names |
| `drum_icons.py` | the kit palette and icons, shared by the logo and the GUI |
| `make_logo.py` | draws the logo, icon and banner from code |
| `make_diagram.py` | draws the architecture diagram from code |
| `make_exe.py`, `make_embedded.py` | give the GUI its own branded executable |
| `make_shortcut.py` | desktop and Start-menu shortcuts with an app identity |
| `check_running_icon.py`, `check_packaged.py` | diagnose a stubborn taskbar icon |
| `fix_mojibake.py` | repairs text mangled by an editor that assumed the code page |
| `cpu_threads.py` | performance-core detection for hybrid CPUs |
| `features.py` | per-hit feature extraction, shared by training and inference |
| `test_smoke.py` | fast self-contained checks, no datasets required |
| `evaluate.py` | end-to-end score on a synthesised loop — the only measurement that needs no dataset |
| **Benchmarks** | |
| `benchmark_mdb.py` | MDB Drums — onsets on real miked recordings |
| `benchmark_idmt.py` | IDMT-SMT-Drums — independent second test set |
| `benchmark_gmd.py` | Groove MIDI — velocity and articulation |
| `decide_default_separator.py` | head-to-head that picked `uvr` as the default |
| `validate_articulations.py` | ride/crash, pedal, toms, ghosts vs subclass labels |
| `validate_thresholds.py` | 2-fold cross-validation of thresholds |
| `validate_fusion.py` | stem onset fusion, per-track breakdown |
| `compare_with_restem.py` | symmetric comparison against ReStem 2 Pro |
| `compare_input_type.py` | drum stem vs whole song as input |
| `run_adt_str.py`, `score_adt_str.py` | run and score ADT_STR (2026) on our benchmark |
| `fetch_adt_str.py` | download one ADT_STR checkpoint variant |
| `verify_adt_str.py` | check a third-party model reproduces its own published numbers |
| `compare_idm.py` | Inverse Drum Machine vs MDX23C for the velocity stage |
| **Analysis** | |
| `analyze_ghosts.py` | recall by hit strength; is it fixable by threshold |
| `analyze_ghost_fusion.py` | do stems recover quiet hits |
| `analyze_pedal.py` | feature separability for pedal hi-hat |
| `analyze_restem_cc4.py` | how ReStem's cc4 maps to articulation |
| `check_hihat_decay.py` | do "open" hi-hats really ring longer than "closed" ones |
| `which_stem.py` | which stem an unexplained pitch in another tool's MIDI follows |
| `probe_soundfont.py` | which General MIDI drum notes a sampler actually plays |
| `sweep_thresholds.py`, `sweep_velocity.py`, `sweep_combine.py` | parameter sweeps |
| `experiment_stem_adt.py`, `experiment_stem_onsets.py`, `experiment_trigger.py` | rejected approaches |
| **Training** | |
| `build_dataset.py`, `build_onset_dataset.py` | feature extraction from GMD |
| `train_models.py`, `train_cc4.py` | model training with held-out evaluation |
| `train_ride.py`, `train_onset.py` | two ideas that were measured and rejected |
| `bench_training.py` | is CPU training feasible (yes: 3.8 h for an onset detector) |
| **Utilities** | |
| `keep_awake.py` | stops Modern Standby from pausing an overnight run |
| `bench_devices.py` | time each stage on each available device |
| `bench_threads.py` | find the best CPU thread count for this machine |
| `xpu_separate.py` | teaches audio-separator to use an Intel GPU, and proves it matches |
| `compare_midi.py` | note-by-note diff of two MIDI files (`--by-instrument` across tools) |
| `midi_pitches.py` | pitch histogram, side by side |
| `explain_pitch.py` | what an unknown pitch in another tool's file corresponds to |
| `gm_note_survey.py` | which General MIDI notes real drum modules emit |
| `cache_uvr_stems.py` | pre-compute MDX23C stems (expensive, done once) |
| `render_midi.py` | render MIDI to audio, plus stereo A/B against the original |
| `inspect_midi.py`, `inspect_structure.py` | MIDI internals |
| `compare_separators.py`, `compare_uvr.py` | separator quality probes |

---

## Windows notes

* `torchaudio` 2.11 dropped its own backends and requires `torchcodec`, so the pipeline
  loads audio with `librosa` and hands LarsNet a tensor directly.
* `larsnet/larsnet.py` prints an `α` character, which crashes on a cp1252 console.
  Scripts set UTF-8 themselves.
* `dl.fbaipublicfiles.com` may be blocked, in which case Demucs cannot fetch htdemucs on
  its own; `audio-separator` mirrors it.
* `audio_separator.utils.cli` has no `if __name__ == "__main__"` guard, so running it via
  `python -m` silently does nothing. Use the `audio-separator` console script.
* The GMD archive contains a macOS artefact named `Icon\r`, which is an invalid Windows
  filename and makes `Expand-Archive` fail silently. Extract with a filter.

### The taskbar icon, and why it took five attempts

Worth writing down, because it applies to any Python GUI on Windows and because four
plausible fixes were wrong.

The window icon, the window *class* icon, an explicit Application User Model ID, a
Start-menu shortcut carrying that ID, a correctly formatted `.ico` and a cleared icon
cache were all applied and individually verified — and the taskbar kept showing the
Python logo. `check_running_icon.py` finally asked the only question that mattered:

```
owning executable = ...WindowsApps\PythonSoftwareFoundation.Python.3.13...\pythonw3.13.exe
```

A virtual environment's `pythonw.exe` is a thin launcher that hands off to the real
interpreter. With a Microsoft Store install that interpreter lives under `WindowsApps`,
and Windows binds the taskbar button to *that* process, which cannot be rebranded.

`make_embedded.py` solves it without reinstalling anything: it downloads the embeddable
distribution (10 MB, an ordinary folder rather than a packaged app), copies `tkinter`
and the tcl/tk DLLs from the existing install, points `._pth` at the venv's
`site-packages`, and registers `.venv/Library/bin` through a `sitecustomize.py` so
torch's Intel runtime is still found. Our icon is written into its `pythonw.exe` copy
with the Win32 resource API — no compiler, no packer.

The lesson is the one this project keeps relearning: measure the thing itself. Four
rounds of plausible reasoning were beaten by one diagnostic that reported which process
actually owned the window.

Related: `make_exe.py` (icon resources), `make_shortcut.py` (shortcuts with an app ID),
`check_packaged.py` (is the interpreter a Store app), `check_running_icon.py`.

## Credits

- **ADTOF** — M. Zehren, M. Alunno, P. Bientinesi. PyTorch port by [@xavriley](https://github.com/xavriley/ADTOF-pytorch).
- **LarsNet / StemGMD** — A. I. Mezza, R. Giampiccolo, A. Bernardini, A. Sarti, *Toward deep drum source separation*, Pattern Recognition Letters 183 (2024).
- **MDX23C DrumSep** — model by aufr33 and jarredou, distributed via [audio-separator](https://github.com/nomadkaraoke/python-audio-separator).
- **MDB Drums** — C. Southall et al., derived from MedleyDB.
- **Groove MIDI Dataset** — J. Gillick et al., Magenta.
- **IDMT-SMT-Drums** — Fraunhofer IDMT.
