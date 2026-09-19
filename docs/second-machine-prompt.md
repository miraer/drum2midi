# Kick-off prompt for the second machine

Copy everything below the line into a fresh Copilot session on the other machine. It is
written to be self-contained: that session has none of this one's context.

---

You are helping with **drum2midi**, an open-source pipeline that turns a mixed drum
recording into MIDI. The repository is at `https://github.com/<owner>/drum2midi` (private
— use an account with access).

**The rule that governs this project:** every claim is a measurement, and negative
results are reported as readily as positive ones. Several conclusions have already been
retracted here because a confidence interval showed they were noise. If a result is
disappointing, say so plainly; do not soften it and do not go looking for a kinder
framing.

**Why this machine.** It has a strong CPU and 64 GB of RAM but no GPU. That is not a
limitation for the job below: we measured that ADTOF transcription runs 4.1x *faster* on
the CPU than on a GPU here, because it is recurrent. Separation is the opposite and stays
on the other machine. So do not run anything with `--from-song` or `--separator uvr`.

## Setup

```powershell
git clone https://github.com/<owner>/drum2midi.git
cd drum2midi
python setup_env.py
python setup_env.py --check
```

Then MDB Drums, about 330 MB:

```powershell
git clone --filter=blob:none --no-checkout https://github.com/CarlSouthall/MDBDrums.git mdbdrums
cd mdbdrums; git checkout; cd ..
```

You will also be given a small zip of `bench/note36` — 23 MIDI files, a few hundred KB.
Unpack it to `bench/note36`. Do not try to regenerate them: that needs the separator and
therefore a GPU.

**Before trusting any number, confirm two things match the other machine:**
`DEFAULT_THRESHOLDS` in `drum2midi.py`, and the ADTOF checkpoint reported by
`setup_env.py --check`. Different values mean you are measuring a different system.

## The job

Run `python blind_spots_mdb.py`. About 20 minutes. It writes `bench/blind_mdb.csv`.

**What it measures.** A peak in the audio's own onset envelope counts as *blind* when
every ADTOF class stays under half its threshold near that peak — that is, the model did
not merely fall short, it did not react at all. The script reports the blind share per
track next to that track's F1, and correlates the two.

**Why we care.** On one real song 18.2% of audible onsets are blind, in passages covering
17.2% of the track, and 45 of them are louder than the median onset, so they are not faint
hits. On MDB the picture so far is uneven: 80sRock, Beatles, BebopJazz and Britpop are at
0.0%, Country1 at 9.2%, CoolJazz at 9.6%. We want the full 23.

**How to read the result — decided in advance, so the reading is not chosen after seeing
the number:**

- A clearly negative correlation means blind onsets travel with poor scores. That would
  justify prototyping a fallback which fires only where the model is silent.
- A correlation near zero means blind onsets are not what costs us accuracy. The fallback
  idea is then dropped and written up as a measured negative result.

Do not report a difference as real without an interval. `significance.py` in the repo
bootstraps over tracks and shows the pattern to follow: resample tracks, not onsets,
because the recording is the unit that varies.

**Context that raises the bar:** a stem-based onset detector was already tried here and
rejected — it scored 0.623 against 0.882, because it fired everywhere. The only reason
this idea is not dead on arrival is that it would never fire at all on four of the six
tracks measured so far.

## If there is time afterwards

```powershell
python fetch_egmd.py --survey
```

102 MB of MIDI, no audio. It parses all 45,537 files and prints the class distribution
and velocity histogram. We have already verified those figures on the other machine —
1,074,753 tom onsets at 7.49%, velocity 4-127 with 48.54% below 60 — so this is a check
that the two machines agree, not new information. Report any discrepancy.

Do **not** start the 90 GB audio download without asking first.

## What to send back

1. `bench/blind_mdb.csv`
2. The correlation figure and your reading of it against the two criteria above
3. Anything that contradicts the partial results quoted here — that matters more than
   confirmation

Keep commits out of it unless asked; this is a measurement run, not a code change. If
you do commit anything, run `python check_privacy.py` first: it rejects machine paths,
account names, private drafts and credentials, and a test enforces it.
