# Contributing

The project has one rule that matters more than any style guide:

> **Every claim is a measurement. Negative results are reported next to positive ones.**

That is not a slogan. Roughly a dozen ideas in this repository were tried, measured and
thrown away, and each one is written up with the number that killed it. Several published
conclusions were later retracted here because a confidence interval showed they were
noise. If your change makes the project better, show the measurement. If it turns out not
to, that finding is just as welcome — say so and it goes in the README.

## Before you claim an improvement

A MICRO F1 quoted to three decimals over 23 recordings reads as though it were precise.
It is not. Use the tools that already exist:

```powershell
& $py significance.py        # bootstraps the ReStem comparison over tracks
& $py score_extractors.py    # same idea for the separator choice
```

Both resample **tracks**, not onsets, because the recording is the unit that varies. A
difference whose 95% interval contains zero has not been demonstrated. On this test set
that threshold is around one point of F1 — smaller differences are simply not resolvable
with 23 recordings, and saying so is more useful than picking the bigger number.

Watch for small classes. MDB Drums contains **90 tom onsets, 1.14% of the corpus**, and
IDMT-SMT-Drums contains none. Any per-class tom figure from these sets, in either
direction, is close to meaningless. See [docs/datasets.md](docs/datasets.md).

Watch for balanced samples too. An earlier experiment reported kick F1 0.968 on a
balanced training sample and 0.623 on real tracks, because onsets are 1.5% of frames in
real audio and 25% in the sample. Score on whole recordings, and score them the way the
pipeline is scored: peak-picked, matched within 50 ms. That same 0.623 was frame-exact,
and on another model frame-exact scoring gave 0.227 where the pipeline's scorer gave 0.578.

Watch for populations with the same name. Every ENST figure here is over **210
recordings and 2617 tom onsets**. The other 108 recordings are isolated hits, and a script
that walks the annotations without filtering them out scores 318 recordings. That turns
the shipped tom F1 of 0.539 into 0.521, and it has happened three times. Take the
recordings from `benchmark_enst.recordings()`. If a script picks them another way, say how
in a comment starting `# ENST population:`, because the smoke test checks for one or the
other.

## Running the tests

```powershell
& $py test_smoke.py
```

28 tests, a few minutes, no network. They are deliberately cheap so there is no excuse to
skip them. Several exist because of specific past failures:

- `every script answers --help` — seven scripts once started multi-hour jobs when asked
  what they did.
- `pipeline finds its own modules without the script directory` — the packaged launcher
  could not import half the project, and `--help` still worked, so nothing caught it.
- `kick is written as general midi 36` — a survey of 445,494 notes from Roland hardware
  found 36 used for every kick and 35 for none.

If you fix a bug, add the test that would have caught it. That is how most of these
arrived.

## Reproducing the published numbers

Everything in the README comes from a script in this repository. The datasets are
downloaded by the user and are not redistributed here — see the Install section for
which ones and where from, and `THIRD-PARTY-NOTICES.md` for what their licences permit.
Non-commercial terms apply to several of the models; the code is MIT, but that does not
extend to somebody else's weights.

## Style

Nothing unusual: follow the surrounding code. Comments explain *why*, not *what* — the
interesting comments in this codebase record the measurement or the failure that led to a
decision, so a future reader does not repeat it.

Commit messages are prose, not bullet lists, and they explain the reasoning rather than
restating the diff.
