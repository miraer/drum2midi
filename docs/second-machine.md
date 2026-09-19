# Running part of the work on a second machine

Written for a machine with plenty of CPU and memory but no GPU and no ReStem. That
combination decides what is worth moving and what is not.

## What actually has to travel

Almost nothing. The heavy folders here are all reproducible:

| | Size | How it gets there |
|---|---|---|
| code and our two models | 2 MB | `git clone` |
| model weights (MDX23C, ADTOF) | 2 GB | `setup_env.py` downloads them |
| MDB Drums | 330 MB | `git clone`, see README Install |
| E-GMD MIDI | 102 MB | `fetch_egmd.py` |
| virtual environment | 3.4 GB | rebuilt by `setup_env.py` |
| **ReStem exports** | **202 MB** | **cannot be reproduced — trial is one machine, seven days** |
| bench cache | 1.1 GB | re-runnable, but it is hours of compute |

Only the last two are worth copying, and the ReStem exports are only needed for work
that compares against ReStem — which is not the work this machine should be doing.

## What to send there, and why

The split follows a measurement already in the README: transcription is faster on the
CPU than on the GPU here, because ADTOF is recurrent — 1408 ms against 5731 ms, 4.1x.
Separation is the opposite, 12.2x faster on the GPU. So a GPU-less machine is not a
weak machine for this project; it is a machine that should not be separating.

**Good fit — nothing but ADTOF and arithmetic:**

- `blind_spots_mdb.py` — 23 tracks, about 20 minutes, needs MDB and `bench/note36`
- `blind_spots.py` on any audio
- `significance.py`, `score_extractors.py` — pure arithmetic over existing MIDI
- training on E-GMD, where 64 GB of memory is the useful part

**Keep here:**

- anything with `--from-song` or `--separator uvr`, which is separation and wants the GPU
- everything involving ReStem: the trial is tied to this machine
- the GUI

## Setting it up

```powershell
git clone https://github.com/<owner>/drum2midi.git
cd drum2midi
python setup_env.py                 # venv, dependencies, MDX23C and ADTOF weights
python setup_env.py --check         # confirm before trusting it
```

Then the data for whichever job it is running. For the blind-spot work:

```powershell
# MDB Drums, 330 MB — the annotations and the drum-only recordings
git clone --filter=blob:none --no-checkout https://github.com/CarlSouthall/MDBDrums.git mdbdrums
cd mdbdrums; git checkout; cd ..
```

For E-GMD:

```powershell
python fetch_egmd.py --survey       # 102 MB of MIDI, and prints what is in it
python fetch_egmd.py --audio        # 90 GB, only if training
```

`bench/note36` holds the 23 transcriptions that `blind_spots_mdb.py` scores against.
They are small; copy them rather than recomputing, because recomputing them needs the
separator and therefore the GPU:

```powershell
# from this machine
Compress-Archive bench\note36 note36.zip     # a few hundred KB
```

## Getting the results back

Every script writes a CSV or a log rather than only printing:

```powershell
python blind_spots_mdb.py            # writes bench\blind_mdb.csv
```

Copy that file back here and the analysis continues where it left off. Nothing in the
comparison depends on which machine produced the numbers, provided the thresholds and
the model weights match — `setup_env.py --check` prints both, and they should be
compared before trusting a result from elsewhere.

## Two things to verify before trusting a remote number

1. **Same thresholds.** `DEFAULT_THRESHOLDS` in `drum2midi.py` is tuned per class. A
   result computed with different values is not comparable.
2. **Same ADTOF weights.** `setup_env.py --check` reports the checkpoint; if the two
   machines resolved different files, the numbers are measuring different models.

Both are cheap to check and expensive to discover afterwards.
