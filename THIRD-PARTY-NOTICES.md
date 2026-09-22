# Third-party notices

The MIT licence in `LICENSE` covers the code in this repository **and the two model
files in `models/`**. It does **not** cover the models and datasets the pipeline
downloads at install time, and several of those are non-commercial.

## Why MIT is not a contradiction

None of the non-commercial weights or datasets are redistributed here. The install
steps fetch them from their original sources, so their licences reach you directly from
the people who wrote them, unchanged. This repository is a wrapper: MIT applies to the
wrapper, each dependency keeps its own terms, and the strictest terms in the set govern
what you may do with the combination.

The practical consequence is the part worth reading twice: **with the default models,
the assembled pipeline is non-commercial**, because a CC BY-NC licence restricts *use*,
not only redistribution. Running an NC model on your own server and charging for access
is exactly the prohibited use, and a SaaS wrapper does not work around it. MIT on this
code does not, and cannot, grant you rights to somebody else's weights.

## The two models shipped here are clean

`models/velocity.pkl` and `models/pedal.pkl` are trained by `train_models.py` on the
**Groove MIDI Dataset**, which is CC BY 4.0 and permits commercial use. They are our own
work, derived from freely licensed data, and are covered by this repository's MIT
licence.

> Groove MIDI Dataset, Jon Gillick, Adam Roberts, Jesse Engel, Douglas Eck and David
> Bamman, "Learning to Groove with Inverse Sequence Transformations" (ICML 2019).
> Licensed under CC BY 4.0. https://magenta.tensorflow.org/datasets/groove

Two other trainers in this repository, `train_ride.py` and `train_onset.py`, do read
non-commercial datasets. Both experiments were rejected on their measured results and
neither ships a model file, so nothing derived from those datasets is distributed here.

## Dependencies and their terms

| Component | Licence | Effect |
|---|---|---|
| ADTOF | CC BY-NC-SA 4.0 | non-commercial |
| ADTOF-pytorch | no licence file | all rights reserved by default |
| LarsNet weights | CC BY-NC 4.0 | non-commercial |
| MDX23C DrumSep | not stated | unclear |
| MDB Drums | CC BY-NC-SA 4.0 | non-commercial, benchmark only |
| IDMT-SMT-Drums | CC BY-NC-ND 4.0 | non-commercial, benchmark only |
| Demucs, audio-separator, DrumSep code | MIT | commercial use allowed |
| PySide6 (the window only) | LGPL-3.0 / GPL-2.0 / GPL-3.0 | commercial use allowed, with conditions — see below |
| GMD / E-GMD / StemGMD | CC BY 4.0 | commercial use allowed, attribution required |

PySide6 is the one copyleft dependency in the list, and it arrived with the window
redesign; the previous toolkit was tkinter, which ships with Python under the PSF licence.
Qt for Python offers no permissive option — PyPI states `LGPL-3.0-only OR GPL-2.0-only OR
GPL-3.0-only` — so the terms are worth stating rather than leaving to be discovered.

Installed with pip and imported at runtime, this is ordinary dynamic linking and does not
affect the licence of this code: `drum2midi.py` and the command line do not import it at
all, and the MIT terms above are unchanged. The condition attaches to **redistribution of a
bundle**. Anyone shipping `runtime/` from `make_embedded.py`, or any other package with Qt
inside it, has to keep the LGPL parts replaceable by the recipient and carry the licence
text. Using the window, or building it from this repository, carries no such obligation.

A commercial path exists and does not require relicensing anything: the training data is
already free. GMD, E-GMD and StemGMD are CC BY 4.0 and Demucs is MIT, so a separator and
a transcriber trained from scratch on those would carry no NC terms. That is work, not a
licensing trick.

See the Licensing section of `README.md` for the same table in context.

