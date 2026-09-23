#!/usr/bin/env python
"""
drum2midi - free drum-audio -> MIDI pipeline.

Stage 1  ADTOF-pytorch transcribes the mixed drum track  -> WHAT and WHEN (5 classes)
Stage 2  LarsNet demixes the same track into 5 stems     -> HOW LOUD (velocity)
Stage 3  Per-stem analysis refines articulations         -> tom pitches, open/closed hats
Stage 4  Tempo estimation + General MIDI export

ADTOF alone emits a fixed velocity of 100 and a single tom/hi-hat class.
The LarsNet stems are what make the result musically usable.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
# Python normally puts the script's own directory on sys.path, but not always: the
# embedded runtime built by make_embedded.py defines sys.path entirely through its
# ._pth file, and `python -P` suppresses it too. Either way the sibling modules
# imported further down -- devices, cpu_threads, note_names -- would not be found.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LARSNET_DIR = ROOT / "larsnet"
DRUMSEP_DIR = ROOT / "drumsep"
DRUMSEP_SIG = "49469ca8"
UVR_DIR = ROOT / "uvr"
MODELS_DIR = ROOT / "models"
UVR_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
UVR_TAG = "MDX23C-DrumSep-aufr33-jarredou"
HTDEMUCS_MODEL = "htdemucs.yaml"

# tqdm writes "  42%|####      | 21/50 [...]" to stderr; only the number is needed
_TQDM_PCT = re.compile(r"(\d{1,3})%\|")
SR = 44100

# ADTOF 5-class output -> General MIDI percussion notes
ADTOF_KICK, ADTOF_SNARE, ADTOF_TOM, ADTOF_HAT, ADTOF_CYM = 35, 38, 47, 42, 49

# Which LarsNet stem carries each ADTOF class. The two taxonomies line up 1:1,
# which is precisely why this combination works.
CLASS_TO_STEM = {
    ADTOF_KICK: "kick",
    ADTOF_SNARE: "snare",
    ADTOF_TOM: "toms",
    ADTOF_HAT: "hihat",
    ADTOF_CYM: "cymbals",
}

GM_NAMES = {
    35: "Kick", 36: "Kick", 38: "Snare", 40: "Snare rim",
    41: "Floor tom", 43: "Floor tom hi", 45: "Low tom", 47: "Mid tom",
    48: "Hi-mid tom", 50: "High tom",
    42: "Hi-hat closed", 44: "Hi-hat pedal", 46: "Hi-hat open",
    49: "Crash", 51: "Ride",
}

# 2-fold cross-validated on MDB Drums (validate_thresholds.py). Tuning every class
# globally turned out to be overfitting - on held-out tracks it scored 0.848 against
# ADTOF's stock 0.850. Only snare survives as a genuine global improvement. Toms are
# handled separately below.
DEFAULT_THRESHOLDS = [0.22, 0.14, 0.32, 0.22, 0.30]

# Peak activation per class from the last transcribe() call, so a run that detects
# nothing can say whether the threshold or the model is the limit.
_LAST_PEAKS: List[float] = []

# Toms are sparse and their density swings wildly between tracks, so no fixed value
# fits. Taking a high percentile of each track's own tom activations lifts held-out
# tom F1 from 0.322 to 0.472, and MICRO from 0.850 to 0.858.
TOM_PERCENTILE = 98.5
TOM_FLOOR = 0.25

# That percentile is taken over frames, which quietly makes it a cap on how many toms a
# recording is allowed to contain: at 100 fps the top 1.5% of frames is about 90 frames
# a minute, and a tom occupies two or three of them. On tom-sparse material that is
# exactly the phantom suppression it was built for. On tom-dense material it forbids
# detections the model is making perfectly well -- a percentile must drift high exactly
# when the class it thresholds is being played a lot.
#
# The floor protected sparse material from a threshold drifting too low. Nothing
# protected dense material from one drifting too high. The ceiling does.
#
# Measured on ENST-Drums, 210 recordings, paired bootstrap over recordings:
#   ENST tom F1  0.342 -> 0.539   [+0.098, +0.286]   significant
#   MDB  tom F1  0.605 -> 0.589   [-0.036, +0.000]   not significant
#   tom recall on ENST            0.227 -> 0.466
# This line read 0.227 -> 0.634 until it was checked: 0.634 is the recall of dropping
# the adaptive threshold entirely, a different policy measured in a different run, and
# quoting it here credited the ceiling with recall it does not deliver.
# and, because sixteen candidates scored on the recordings that chose them is how a
# benchmark gets overfitted, the value was re-chosen on ENST drummers 1-2 alone and
# applied to drummer 3 unseen: +0.194 [+0.061, +0.307] against +0.197 in sample.
#
# 0.45 is the conservative end of the trade, not the optimum. Ceilings of 0.32 to 0.40
# buy more on dense material (+0.248 to +0.217) and cost MDB significantly. Ordinary
# drum tracks are tom-sparse and are what most people convert, so the sparse corpus gets
# the benefit of the doubt. That is a judgement about users, not a measurement.
TOM_CEILING = 0.45

# How many dB below a drum's loudest hit maps to the lowest velocity. Swept against
# GMD's real module velocities (sweep_velocity.py, 2371 annotated hits): a kick keeps
# a narrow dynamic range inside a groove, while cymbals span soft ride taps to loud
# crashes. The curve is fairly flat, so these are gentle preferences, not cliffs.
DYN_RANGE = {"kick": 18.0, "snare": 36.0, "toms": 36.0, "hihat": 36.0, "cymbals": 48.0}

# Tom pitch buckets used when --split-toms finds 1, 2 or 3 distinct drums
TOM_LADDER = {1: [47], 2: [43, 47], 3: [43, 47, 50]}


@contextlib.contextmanager
def _in_dir(path: Path):
    """LarsNet resolves its checkpoint paths relative to the process CWD."""
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# Stage 1 - transcription
# --------------------------------------------------------------------------

_ADTOF_CACHE: Dict[str, object] = {}


def _adtof_model(device: str):
    """Build the ADTOF model once; stem fusion needs it several times per run."""
    if device in _ADTOF_CACHE:
        return _ADTOF_CACHE[device]
    from adtof_pytorch import (calculate_n_bins, create_frame_rnn_model,
                               get_default_weights_path, load_pytorch_weights)
    model = create_frame_rnn_model(calculate_n_bins())
    model.eval()
    weights = get_default_weights_path()
    if weights and Path(weights).exists():
        model = load_pytorch_weights(model, weights, strict=False)
    model.to(device)
    _ADTOF_CACHE[device] = model
    return model


def transcribe(audio_path: Path, device: str, thresholds: Optional[Sequence[float]],
               fps: int = 100, adaptive_toms: bool = True) -> Dict[int, List[float]]:
    from adtof_pytorch import LABELS_5, PeakPicker, load_audio_for_model
    import torch

    model = _adtof_model(device)
    x = load_audio_for_model(str(audio_path)).to(device)
    with torch.no_grad():
        activations = model(x).cpu().numpy()

    thr = list(thresholds) if thresholds else list(DEFAULT_THRESHOLDS)
    if thresholds is None and adaptive_toms:
        tom_col = activations[0][:, 2]
        thr[2] = min(max(float(np.percentile(tom_col, TOM_PERCENTILE)), TOM_FLOOR),
                     TOM_CEILING)

    picker = PeakPicker(thresholds=thr, fps=fps)
    # Recorded so a failure can say whether lowering the threshold could possibly help.
    # On quiet mallet material the model produces almost no activation at all, and the
    # old advice to lower --thresholds sent people down a road that ends at 0.05 with
    # still nothing picked.
    global _LAST_PEAKS
    _LAST_PEAKS = [float(activations[0][:, i].max())
                   for i in range(activations[0].shape[1])]
    return picker.pick(activations, labels=LABELS_5, label_offset=0)[0]


def fuse_stem_onsets(onsets: Dict[int, List[float]], stems: Dict[str, np.ndarray],
                     device: str, thresholds: Optional[Sequence[float]],
                     gap: float = 0.05) -> int:
    """Add onsets that a clean stem reveals but the mix-based pass missed.

    On an isolated stem the instrument is already known, so ADTOF is used purely as an
    onset detector: take the maximum over its five outputs and peak-pick that.

    Only worth it when the separation is clean. Over all 23 MDB tracks with MDX23C
    stems, MICRO F1 goes 0.861 -> 0.882: recall climbs 0.855 -> 0.914 while precision
    slips 0.866 -> 0.853. With LarsNet stems the same fusion drops MICRO below the
    mix-only baseline, which is why it is tied to --separator uvr.

    The gain is heavily concentrated: it helped 4 tracks, hurt 9 and left 10 unchanged,
    and correlates -0.647 with how well the mix pass already did. Mean gain is +0.029 on
    the weaker half of the set against -0.002 on the stronger half. Almost all of it
    comes from two jazz tracks (BebopJazz +0.246, ModalJazz +0.098). Losses are tiny,
    never worse than -0.014, so it stays on by default - but on material the mix pass
    already handles well it only costs a little precision.

    Snare (+0.039) and hi-hat (+0.029) account for the improvement; kick, toms and
    cymbals are unchanged.
    """
    from adtof_pytorch.post_processing import NotePeakPickingProcessor
    from adtof_pytorch import load_audio_for_model
    import soundfile as sf
    import torch

    model = _adtof_model(device)
    thr = list(thresholds) if thresholds else list(DEFAULT_THRESHOLDS)
    added = 0
    tmp = Path(tempfile.mkdtemp(prefix="stemfuse_"))
    try:
        for cls, stem_name in CLASS_TO_STEM.items():
            stem = stems.get(stem_name)
            if stem is None or cls not in onsets:
                continue
            path = tmp / f"{stem_name}.wav"
            sf.write(str(path), stem.T, SR)
            with torch.no_grad():
                act = model(load_audio_for_model(str(path)).to(device)).cpu().numpy()[0]
            idx = [35, 38, 47, 42, 49].index(cls)
            proc = NotePeakPickingProcessor(threshold=float(thr[idx]), pre_avg=0.1,
                                            post_avg=0.01, pre_max=0.02, post_max=0.01,
                                            combine=0.02, fps=100)
            found = [t for t, _ in proc.process(act.max(axis=1))]
            base = np.array(onsets[cls])
            for t in found:
                if not base.size or np.min(np.abs(base - t)) > gap:
                    onsets[cls].append(float(t))
                    added += 1
            onsets[cls].sort()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return added


# --------------------------------------------------------------------------
# Stage 2 - separation
# --------------------------------------------------------------------------

def separate(audio: np.ndarray, device: str, wiener: Optional[float],
             chunk_seconds: float) -> Dict[str, np.ndarray]:
    """Run LarsNet over the mix. Long files are processed in overlap-added chunks
    so peak RAM stays bounded regardless of track length."""
    import torch

    sys.path.insert(0, str(LARSNET_DIR))
    with _in_dir(LARSNET_DIR):
        from larsnet import LarsNet

        net = LarsNet(
            wiener_filter=wiener is not None,
            wiener_exponent=wiener if wiener is not None else 1.0,
            device=device,
            config="config.yaml",
        )

        # LarsNet folds time into 512-frame / ~11.9 s blocks internally. Snapping the
        # chunk length to a whole number of blocks keeps our seams on its own seams.
        block = 512 * 1024
        n_blocks = max(1, int(round(chunk_seconds * SR / block)))
        chunk = n_blocks * block
        overlap = min(block // 8, chunk // 4)
        hop = chunk - overlap

        total = audio.shape[-1]
        stems = {s: np.zeros((2, total), dtype=np.float64) for s in net.stems}
        weight = np.zeros(total, dtype=np.float64)

        starts = list(range(0, max(1, total), hop))
        for idx, start in enumerate(starts):
            end = min(start + chunk, total)
            if end - start <= 0:
                break
            seg = torch.from_numpy(audio[:, start:end].astype(np.float32))

            with torch.no_grad():
                out = net.separate_wiener(seg) if wiener is not None else net.separate(seg)

            n = end - start
            win = np.ones(n, dtype=np.float64)
            fade = min(overlap, n // 2)
            if fade > 0:
                ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
                if start > 0:
                    win[:fade] = ramp
                if end < total:
                    win[-fade:] = ramp[::-1]

            for stem, wav in out.items():
                arr = wav.cpu().numpy()
                if arr.ndim == 1:
                    arr = np.stack([arr, arr])
                arr = arr[:, :n] if arr.shape[-1] >= n else np.pad(arr, ((0, 0), (0, n - arr.shape[-1])))
                stems[stem][:, start:end] += arr * win
            weight[start:end] += win

            log(f"    chunk {idx + 1}/{len(starts)}  [{start / SR:7.1f}s - {end / SR:7.1f}s]")
            if end >= total:
                break

    sys.path.remove(str(LARSNET_DIR))
    weight[weight < 1e-8] = 1.0
    return {s: (v / weight).astype(np.float32) for s, v in stems.items()}


def xpu_available() -> bool:
    """Intel GPU (Arc / Xe). Measured at 12x faster than CPU for MDX23C."""
    try:
        import torch
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def _separator_command(src: Path, model: str, outdir: Path, device: str) -> List[str]:
    """The command that runs audio-separator on `device`.

    Separate from _audio_separator so the device actually reaching the separator can
    be asserted without running a separation, which is the longest stage here.

    audio-separator chooses its own device and offers no way to say otherwise -- its
    --help lists DirectML and fp16 but nothing for the CPU -- so both non-default
    devices need the CLI wrapped in a patch rather than a flag.
    """
    # audio_separator.utils.cli has no __main__ guard, so `-m` silently does nothing.
    # Prefer the console script and fall back to importing main() explicitly.
    exe = Path(sys.executable).parent / ("audio-separator.exe" if os.name == "nt"
                                         else "audio-separator")
    use_xpu = device != "cpu" and xpu_available()
    if device == "cpu":
        # `--device cpu` used to stop at this function: the flag picked the transcriber's
        # device and was then dropped, the console script ran, and it logged "CUDA is
        # available in Torch, setting Torch device to CUDA" and used the GPU. Measured
        # on a 4070 SUPER, stage 2 took 26.6s under --device cpu and 26.9s under
        # --device auto, while stages 1 and 3 slowed by 4.1x and 3.7x as they should.
        # The longest stage of the pipeline ignored the flag entirely.
        head = [sys.executable, "-c",
                f"import sys; sys.path.insert(0, r'{ROOT}'); "
                "from xpu_separate import force_cpu_audio_separator; "
                "force_cpu_audio_separator(); "
                "from audio_separator.utils.cli import main; sys.exit(main())"]
    elif use_xpu:
        # audio-separator only knows cuda and mps, so its own entry point would run on
        # the CPU. Patch the device choice, then hand over to the unmodified CLI.
        head = [sys.executable, "-c",
                f"import sys; sys.path.insert(0, r'{ROOT}'); "
                "from xpu_separate import patch_audio_separator; patch_audio_separator(); "
                "from audio_separator.utils.cli import main; sys.exit(main())"]
    elif exe.exists():
        head = [str(exe)]
    else:
        head = [sys.executable, "-c",
                "import sys; from audio_separator.utils.cli import main; sys.exit(main())"]
    return head + [str(src), "-m", model, "--output_dir", str(outdir),
                   "--model_file_dir", str(UVR_DIR / "models")]


def _audio_separator(src: Path, model: str, outdir: Path, device: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = _separator_command(src, model, outdir, device)

    # Separation is by far the longest stage, so its progress is forwarded rather than
    # swallowed: audio-separator writes a tqdm bar to stderr, and the percentage from it
    # is what lets the GUI move during the several minutes this takes.
    tail: List[str] = []
    last_pct = -1
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", bufsize=1)
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            match = _TQDM_PCT.search(line)
            if match:
                pct = int(match.group(1))
                if pct != last_pct and pct % 5 == 0:
                    last_pct = pct
                    log(f"      separating: {pct}%")
                continue
            tail.append(line)
            del tail[:-15]
    finally:
        proc.wait()

    produced = list(outdir.iterdir()) if outdir.exists() else []
    if proc.returncode != 0 or not produced:
        raise RuntimeError(tail[-1][:200] if tail else "audio-separator produced no output")
    return outdir


# Python modules each separator imports, as (module, what it is, pip name).
SEPARATOR_MODULES = {
    "uvr": [("audio_separator", "the audio-separator package", "audio-separator")],
    "larsnet": [("yaml", "PyYAML", "pyyaml"),
                ("torchaudio", "torchaudio", "torchaudio")],
    "hybrid": [("yaml", "PyYAML", "pyyaml"),
               ("torchaudio", "torchaudio", "torchaudio")],
}

# Programs a separator runs that are not Python packages at all. audio-separator
# calls `ffmpeg -version` before it does anything else, so a missing ffmpeg stops the
# default separator dead -- and nothing here imports it, which is exactly why it was
# absent from every dependency list this project had.
SEPARATOR_BINARIES = {
    "uvr": [("ffmpeg", "FFmpeg")],
}


def separator_missing(args) -> Optional[tuple]:
    """What the chosen separator needs and has not got, checked before any work.

    Separation is step 2 of 4, so a missing dependency used to surface only after
    ADTOF had transcribed the whole file -- minutes of work thrown away to reach an
    error that was knowable at the start.

    That kept happening after this function existed, because it only looked for
    Python modules. audio-separator shells out to ffmpeg on startup, and with no
    ffmpeg installed the failure arrived as the child process's own last traceback
    line, relayed verbatim: "FileNotFoundError: [WinError 2] The system cannot find
    the file specified". It names neither ffmpeg nor a remedy, and it came after a
    full transcription of a four-and-a-half minute recording.

    Returns (what is needed, why it is not usable, how to fix, extra detail or None).
    """
    import importlib.util
    import shutil

    from devices import pip_here
    # setup_env owns the per-platform ffmpeg instructions and imports nothing outside
    # the standard library, so there is one copy of that advice rather than two that
    # can drift. The dependency runs this way round because setup_env has to work
    # before anything in requirements.txt is installed.
    from setup_env import ffmpeg_advice

    if getattr(args, "no_separate", False):
        return None
    chosen = getattr(args, "separator", "")

    for module, what, package in SEPARATOR_MODULES.get(chosen, ()):
        try:
            found = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            return (what, f"`import {module}` fails in this interpreter",
                    f"{pip_here()} install {package}", sys.executable)

    for program, what in SEPARATOR_BINARIES.get(chosen, ()):
        if shutil.which(program) is None:
            return (what, f"`{program}` is not on PATH", ffmpeg_advice(), None)
    return None


def separate_uvr(src: Path, device: str, length: int) -> Dict[str, np.ndarray]:
    """UVR MDX23C drum separator (aufr33/jarredou). Six stems - it is the only
    option here that splits ride from crash. Much slower than LarsNet on CPU.

    MDX23C runs about 10x slower than real time, so a cache built by
    cache_uvr_stems.py is reused when the track is already there."""
    import librosa

    cached = ROOT / "bench" / "uvr_stems" / src.stem
    if cached.is_dir():
        stems: Dict[str, np.ndarray] = {}
        for name in ("kick", "snare", "toms", "hihat", "cymbals", "ride"):
            f = cached / f"{name}.flac"
            if f.exists():
                y, _ = librosa.load(str(f), sr=SR, mono=False)
                stems[name] = _fit(np.asarray(y), length)
        if len(stems) >= 5:
            return stems

    tmp = Path(tempfile.mkdtemp(prefix="uvr_"))
    try:
        _audio_separator(src, UVR_MODEL, tmp, device)
        stems = {}
        for raw, name in (("kick", "kick"), ("snare", "snare"),
                          ("toms", "toms"), ("hh", "hihat"),
                          ("crash", "cymbals"), ("ride", "ride")):
            matches = list(tmp.glob(f"*({raw})*"))
            if matches:
                y, _ = librosa.load(str(matches[0]), sr=SR, mono=False)
                stems[name] = _fit(np.asarray(y), length)
        return stems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def diffq_available() -> bool:
    """Whether the Demucs models, and so --from-song, can load."""
    import importlib.util
    return importlib.util.find_spec("diffq") is not None


def extract_drums(src: Path, device: str, dest: Path, model: str = None) -> Path:
    """Pull a drum stem out of a full mix, so the pipeline can take whole songs.

    Four models in audio-separator emit a drums stem and their published SDR differs
    by 1.5 dB, so the choice is exposed -- compare_extractors.py scores them end to end
    rather than by SDR.
    """
    chosen = model or HTDEMUCS_MODEL
    # The Demucs models (named *.yaml) load through audio-separator's Demucs code,
    # which imports diffq -- optional on macOS, where it often cannot be built. Say so
    # here rather than let the import fail deep inside the separator.
    if chosen.endswith(".yaml") and not diffq_available():
        raise RuntimeError(
            f"{chosen} needs the diffq package, which is not installed. On macOS it "
            "has to be compiled: run `xcode-select --install`, then "
            f"`{sys.executable} -m pip install diffq`")
    tmp = Path(tempfile.mkdtemp(prefix="extract_"))
    try:
        _audio_separator(src, chosen, tmp, device)
        hits = [p for p in tmp.iterdir() if "drums" in p.name.lower()]
        if not hits:
            raise RuntimeError(f"{chosen} produced no drums stem")
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hits[0], dest)
        return dest
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def split_ride(events: List[tuple], stems: Dict[str, np.ndarray],
               margin: float = 1.2) -> int:
    """ADTOF has a single cymbal class. When a separator gives independent ride and
    crash stems, decide per hit which one actually rang."""
    crash, ride = stems.get("cymbals"), stems.get("ride")
    if crash is None or ride is None:
        return 0
    c_sig, r_sig = _mono(crash), _mono(ride)
    moved = 0
    for i, (t, pitch, vel) in enumerate(events):
        if pitch != ADTOF_CYM:
            continue
        if peak_at(r_sig, t) > peak_at(c_sig, t) * margin:
            events[i] = (t, 51, vel)
            moved += 1
    return moved


def load_learned(enabled: bool = True) -> Dict[str, Optional[dict]]:
    """Optional models trained on GMD by train_models.py.

    Velocity models exist only for the drums where they beat the handcrafted formula
    on held-out tracks (cymbals clearly, hi-hat marginally); kick and snare are still
    better served by the formula. The pedal classifier reaches 0.787 balanced accuracy
    where handcrafted features managed only 0.633.
    """
    out: Dict[str, Optional[dict]] = {"velocity": None, "pedal": None}
    if not enabled:
        return out
    import pickle
    for key, name in (("velocity", "velocity.pkl"), ("pedal", "pedal.pkl")):
        path = MODELS_DIR / name
        if path.exists():
            try:
                with path.open("rb") as fh:
                    out[key] = pickle.load(fh)
            except Exception as exc:
                log(f"      WARNING: could not load {name}: {exc}")
    if not any(out.values()):
        # silently falling back would make the run look like it used the models
        log("      no trained models in models/ - using handcrafted formulas "
            "(see README: Install)")
    return out


def _fit(stem: np.ndarray, length: int) -> np.ndarray:
    if stem.ndim == 1:
        stem = np.stack([stem, stem])
    if stem.shape[-1] >= length:
        return stem[:, :length]
    return np.pad(stem, ((0, 0), (0, length - stem.shape[-1])))


def separate_drumsep(src: Path, device: str, length: int) -> Dict[str, np.ndarray]:
    """DrumSep (Hybrid Demucs fine-tune, inagoy/drumsep). Four stems, and note that
    hi-hat is NOT separated: 'platillos' carries hats and cymbals together."""
    import librosa

    repo = DRUMSEP_DIR / "model"
    if not (repo / f"{DRUMSEP_SIG}.th").exists():
        raise RuntimeError(f"DrumSep model not found at {repo / (DRUMSEP_SIG + '.th')}")

    tmp = Path(tempfile.mkdtemp(prefix="drumsep_"))
    try:
        subprocess.run(
            [sys.executable, "-m", "demucs", "--repo", str(repo), "-n", DRUMSEP_SIG,
             "-d", device, "-o", str(tmp), str(src)],
            check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        outdir = tmp / DRUMSEP_SIG / src.stem
        stems: Dict[str, np.ndarray] = {}
        for raw, name in (("bombo", "kick"), ("redoblante", "snare"), ("toms", "toms")):
            f = outdir / f"{raw}.wav"
            if f.exists():
                y, _ = librosa.load(str(f), sr=SR, mono=False)
                stems[name] = _fit(np.asarray(y), length)
        cym = outdir / "platillos.wav"
        if cym.exists():
            y = _fit(np.asarray(librosa.load(str(cym), sr=SR, mono=False)[0]), length)
            stems["hihat"] = y
            stems["cymbals"] = y
        return stems
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------
# Stage 3 - velocity and articulation
# --------------------------------------------------------------------------

def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0) if x.ndim > 1 else x


def peak_at(sig: np.ndarray, t: float, pre: float = 0.005, post: float = 0.035) -> float:
    a = max(0, int((t - pre) * SR))
    b = min(len(sig), int((t + post) * SR))
    return float(np.max(np.abs(sig[a:b]))) if b > a else 0.0


def stem_is_usable(stem: np.ndarray, mix: np.ndarray, floor_db: float = -55.0) -> bool:
    """A demixer can return an essentially empty stem while the transcriber still
    reports hits for that instrument. Measuring velocity off such a stem would just
    digitise its noise floor, so detect the case and let the caller fall back."""
    s = float(np.sqrt(np.mean(_mono(stem) ** 2)))
    m = float(np.sqrt(np.mean(_mono(mix) ** 2)))
    if m <= 1e-9:
        return False
    return 20.0 * np.log10(max(s, 1e-12) / m) > floor_db


def velocities(onsets: List[float], stem: np.ndarray, dyn_range: float,
               vmin: int, vmax: int) -> List[int]:
    """Map each hit's peak level to a MIDI velocity, normalised against the
    loudest hit of that same instrument so dynamics stay relative."""
    if not onsets:
        return []
    sig = _mono(stem)
    peaks = np.array([peak_at(sig, t) for t in onsets])
    loudest = peaks.max()
    if loudest <= 1e-6:
        return [100] * len(onsets)
    db = 20.0 * np.log10(np.maximum(peaks, 1e-9) / loudest)
    norm = np.clip((db + dyn_range) / dyn_range, 0.0, 1.0)
    return [int(np.clip(round(vmin + (vmax - vmin) * v), 1, 127)) for v in norm]


def _fundamental(sig: np.ndarray, t: float, fmin: float = 55.0, fmax: float = 400.0) -> float:
    a, b = int(t * SR), min(len(sig), int((t + 0.12) * SR))
    if b - a < 512:
        return 0.0
    seg = sig[a:b] * np.hanning(b - a)
    spec = np.abs(np.fft.rfft(seg, n=8192))
    freqs = np.fft.rfftfreq(8192, 1.0 / SR)
    band = (freqs >= fmin) & (freqs <= fmax)
    if not band.any() or spec[band].max() <= 0:
        return 0.0
    return float(freqs[band][np.argmax(spec[band])])


def _kmeans_1d(values: np.ndarray, k: int, iters: int = 40) -> np.ndarray:
    centres = np.percentile(values, np.linspace(10, 90, k))
    labels = np.zeros(len(values), dtype=int)
    for _ in range(iters):
        labels = np.argmin(np.abs(values[:, None] - centres[None, :]), axis=1)
        moved = False
        for j in range(k):
            if (labels == j).any():
                nc = values[labels == j].mean()
                if abs(nc - centres[j]) > 1e-6:
                    centres[j], moved = nc, True
        if not moved:
            break
    return np.argsort(np.argsort(centres))[labels]


def split_toms(onsets: List[float], stem: np.ndarray, max_toms: int,
               min_ratio: float = 1.10) -> List[int]:
    """Cluster tom hits by fundamental pitch into up to `max_toms` drums.

    `min_ratio` is how far apart neighbouring tom fundamentals must be before the
    split is trusted. Validated against the MDB subclass labels (LFT/HFT/MHT): when
    the split fires, the pitch order came out right 27 times out of 27, so accuracy
    was never the problem - coverage was. The original 1.18 gate refused to split
    75% of tom pairs.
    """
    if not onsets:
        return []
    sig = _mono(stem)
    f0 = np.array([_fundamental(sig, t) for t in onsets])
    usable = f0 > 0
    if usable.sum() < 2:
        return [ADTOF_TOM] * len(onsets)

    best_k = 1
    for k in range(2, min(max_toms, int(usable.sum())) + 1):
        lab = _kmeans_1d(f0[usable], k)
        groups = [f0[usable][lab == j] for j in range(k)]
        if any(len(g) < 2 for g in groups):
            continue
        centres = sorted(g.mean() for g in groups)
        # require a clear pitch gap between neighbouring toms
        if all(centres[i + 1] / max(centres[i], 1e-6) > min_ratio
               for i in range(len(centres) - 1)):
            best_k = k

    ladder = TOM_LADDER[best_k]
    if best_k == 1:
        return [ADTOF_TOM] * len(onsets)

    lab = _kmeans_1d(f0[usable], best_k)
    out, j = [], 0
    for ok in usable:
        if ok:
            out.append(ladder[lab[j]])
            j += 1
        else:
            out.append(ladder[len(ladder) // 2])
    return out


def _centroid(sig: np.ndarray, t: float, w: float = 0.04, fmin: float = 200.0) -> float:
    """Spectral centroid of a hit's attack. A hi-hat struck with a stick has a bright
    transient; a foot chick has none, so its centroid sits much lower."""
    a, b = int(t * SR), min(len(sig), int((t + w) * SR))
    if b - a < 256:
        return 0.0
    seg = sig[a:b] * np.hanning(b - a)
    spec = np.abs(np.fft.rfft(seg, n=4096))
    freqs = np.fft.rfftfreq(4096, 1.0 / SR)
    band = freqs >= fmin
    energy = spec[band]
    total = energy.sum()
    return float((energy * freqs[band]).sum() / total) if total > 0 else 0.0


def split_hats(onsets: List[float], stem: np.ndarray, ratio_threshold: float,
               detect_pedal: bool = False, pedal_centroid: float = 0.80,
               pedal_level: float = 0.50) -> List[int]:
    """Classify hi-hat hits as closed (42), open (46) or pedal (44).

    Open is decided by how long the hit rings. Ratio alone is unreliable - a quiet
    closed hat under a ringing cymbal shows a big tail/attack ratio purely from
    bleed - so the tail must also be loud in absolute terms.

    Pedal detection is OFF by default and kept only for experiments. Measured on GMD
    (1102 annotated hat hits, analyze_pedal.py), pedal chicks are not separable from
    closed hats in the hi-hat stem: peak 0.0457 vs 0.0466, centroid 11993 vs 12454,
    d' between 0.01 and 0.48. The best achievable single-threshold balanced accuracy
    is 0.633 against a 0.5 chance baseline, so enabling this mostly adds noise.
    """
    if not onsets:
        return []
    sig = _mono(stem)
    attacks, tails, peaks, centroids = [], [], [], []
    for t in onsets:
        a = sig[int(t * SR):min(int((t + 0.05) * SR), len(sig))]
        b = sig[min(int((t + 0.08) * SR), len(sig)):min(int((t + 0.25) * SR), len(sig))]
        attacks.append(float(np.sqrt(np.mean(a ** 2))) if a.size else 0.0)
        tails.append(float(np.sqrt(np.mean(b ** 2))) if b.size else 0.0)
        peaks.append(peak_at(sig, t))
        centroids.append(_centroid(sig, t))

    attacks = np.array(attacks); tails = np.array(tails)
    peaks = np.array(peaks); centroids = np.array(centroids)
    floor = 0.15 * attacks.max() if attacks.size and attacks.max() > 0 else 0.0
    ratios = tails / (attacks + 1e-9)
    is_open = (ratios > ratio_threshold) & (tails > floor)

    notes = [46 if o else ADTOF_HAT for o in is_open]
    if not detect_pedal or (~is_open).sum() < 4:
        return notes

    rest = ~is_open
    ref_centroid = float(np.percentile(centroids[rest], 65))
    ref_peak = float(np.percentile(peaks[rest], 65))
    if ref_centroid <= 0 or ref_peak <= 0:
        return notes

    for i in range(len(notes)):
        if rest[i] and centroids[i] < pedal_centroid * ref_centroid \
                and peaks[i] < pedal_level * ref_peak:
            notes[i] = 44
    return notes


def detect_stem_onsets(stem: np.ndarray, rel_threshold: float, min_gap: float = 0.045) -> List[float]:
    """Onsets found directly in a separated stem, independent of the transcriber."""
    import librosa

    sig = _mono(stem)
    loudest = float(np.max(np.abs(sig)))
    if loudest <= 1e-6:
        return []
    try:
        candidates = librosa.onset.onset_detect(y=sig, sr=SR, units="time", backtrack=False)
    except Exception:
        return []
    kept: List[float] = []
    for t in candidates:
        t = float(t)
        if peak_at(sig, t) < rel_threshold * loudest:
            continue
        if kept and t - kept[-1] < min_gap:
            continue
        kept.append(t)
    return kept


def rescue_toms(events: List[tuple], tom_stem: np.ndarray, stems: Dict[str, np.ndarray],
                rel_threshold: float, dyn_range: float, vmin: int, vmax: int,
                window: float = 0.045, margin: float = 1.5) -> tuple:
    """ADTOF's tom class is its weakest - fills are routinely emitted as kick or
    snare. The LarsNet tom stem sees them clearly, so take timing from the stem and
    let the loudest stem at that instant decide who owns the hit.

    `margin` guards against a bleedy tom stem stealing the whole snare part: a hit is
    only reassigned when the tom stem beats the incumbent by a clear factor.
    """
    found = detect_stem_onsets(tom_stem, rel_threshold)
    if not found:
        return events, 0, 0

    tom_sig = _mono(tom_stem)
    tom_peaks = np.array([peak_at(tom_sig, t) for t in found])
    loudest = tom_peaks.max() if tom_peaks.size else 0.0
    if loudest <= 1e-6:
        return events, 0, 0

    tom_pitches = {41, 43, 45, 47, 48, 50}
    events = list(events)
    added = reassigned = 0

    for t, tom_peak in zip(found, tom_peaks):
        db = 20.0 * np.log10(max(tom_peak, 1e-9) / loudest)
        vel = int(np.clip(round(vmin + (vmax - vmin) * np.clip((db + dyn_range) / dyn_range, 0, 1)), 1, 127))

        near = [i for i, (et, ep, _) in enumerate(events)
                if abs(et - t) <= window and (ep in tom_pitches or ep in (ADTOF_KICK, ADTOF_SNARE))]
        if any(events[i][1] in tom_pitches for i in near):
            continue

        stolen = False
        for i in near:
            rival = "kick" if events[i][1] == ADTOF_KICK else "snare"
            rival_stem = stems.get(rival)
            rival_peak = peak_at(_mono(rival_stem), t) if rival_stem is not None else 0.0
            if tom_peak > rival_peak * margin:
                events[i] = (events[i][0], ADTOF_TOM, vel)
                reassigned += 1
                stolen = True
                break
        if not stolen and not near:
            events.append((t, ADTOF_TOM, vel))
            added += 1

    return events, added, reassigned


# --------------------------------------------------------------------------
# Stage 4 - MIDI
# --------------------------------------------------------------------------

def estimate_tempo(audio: np.ndarray) -> float:
    import librosa
    try:
        tempo, _ = librosa.beat.beat_track(y=_mono(audio), sr=SR)
        t = float(np.atleast_1d(tempo)[0])
        return t if 40.0 <= t <= 260.0 else 120.0
    except Exception:
        return 120.0


def quantize(t: float, tempo: float, division: int) -> float:
    step = (60.0 / tempo) * (4.0 / division)
    return round(t / step) * step


GM_GROUPS = [
    ("Kick", (35, 36)),
    ("Snare", (38, 40)),
    ("Toms", (41, 43, 45, 47, 48, 50)),
    ("Hi-hat", (42, 44, 46)),
    ("Cymbals", (49, 51)),
]
DRUM_CHANNEL = 9

# ADTOF's kick class is identified internally by 35, but that is not what a DAW expects.
# General MIDI calls 36 "Bass Drum 1" and 35 "Acoustic Bass Drum", and a survey of 1150
# Groove MIDI performances recorded from a Roland TD-11 (gm_note_survey.py, 445494 notes)
# found 36 in 100% of kicks and 35 in none. ReStem 2 Pro writes 36 as well. Remapped at
# write time so the internal class id stays stable; --notes overrides it.
OUTPUT_NOTES = {35: 36}


# Names accepted on the left of --notes and --channels, so a kit piece can be addressed
# without looking up its number. They resolve to the pitch the pipeline emits internally.
DRUM_ALIASES = {
    "kick": ADTOF_KICK, "bd": ADTOF_KICK, "bassdrum": ADTOF_KICK,
    "snare": ADTOF_SNARE, "sd": ADTOF_SNARE,
    "hihat": ADTOF_HAT, "hh": ADTOF_HAT, "hat": ADTOF_HAT,
    "hihat-closed": ADTOF_HAT, "hh-closed": ADTOF_HAT,
    "hihat-open": 46, "hh-open": 46,
    "hihat-pedal": 44, "hh-pedal": 44,
    "tom-low": 43, "tom-mid": 47, "tom-high": 50, "tom": ADTOF_TOM,
    "crash": ADTOF_CYM, "ride": 51,
}

NOTE_LETTERS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}

# Which octave number a DAW gives to MIDI note 60. Cubase, Logic, Ableton and Reaper
# all say C3; the scientific convention says C4; FL Studio says C5. The pipeline follows
# the majority so that "C1" means the kick, as it does in Cubase and in ReStem's manual.
MIDDLE_C_OCTAVE = 3


def parse_note_token(token: str) -> int:
    """Accepts a MIDI number, a drum name, or a note name such as C1 or A#1."""
    token = token.strip()
    if not token:
        raise ValueError("empty note")

    key = token.lower().replace("_", "-").replace(" ", "")
    if key in DRUM_ALIASES:
        return DRUM_ALIASES[key]

    if token.lstrip("+-").isdigit():
        value = int(token)
        if not 0 <= value <= 127:
            raise ValueError(f"note {value} outside 0..127")
        return value

    letter = key[0]
    if letter not in NOTE_LETTERS:
        raise ValueError(f"'{token}' is not a note, a number or a drum name")
    i = 1
    semitone = NOTE_LETTERS[letter]
    while i < len(key) and key[i] in "#bs":
        semitone += 1 if key[i] in "#s" else -1
        i += 1
    octave_text = key[i:]
    try:
        octave = int(octave_text)
    except ValueError:
        raise ValueError(f"'{token}' has no octave number, e.g. C1")
    value = semitone + (octave + 5 - MIDDLE_C_OCTAVE) * 12
    if not 0 <= value <= 127:
        raise ValueError(f"'{token}' is outside the MIDI range")
    return value


def parse_pairs(spec: str, lo: int, hi: int, what: str) -> Dict[int, int]:
    """Parse 'drum=value,drum=value' into a lookup, validating ranges.

    Both sides accept a number; the left also accepts a drum name ('kick'), and when
    the value is a note it may be written as a note name ('C1').
    """
    out: Dict[int, int] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"expected name=value, got '{chunk}'")
        left, right = chunk.split("=", 1)
        pitch = parse_note_token(left)
        right = right.strip()
        if hi > 15 and not right.lstrip("+-").isdigit():
            value = parse_note_token(right)      # --notes kick=C1
        else:
            try:
                value = int(right)
            except ValueError:
                raise ValueError(f"expected a {what} number, got '{right}'")
        if not lo <= value <= hi:
            raise ValueError(f"{what} {value} outside {lo}..{hi}")
        out[pitch] = value
    return out


def apply_channels(path: Path, channel_by_pitch: Dict[int, int]) -> None:
    """pretty_midi pins every drum note to channel 9 and offers no way to change it,
    so the channel bytes are rewritten afterwards. MIDI allows mixed channels inside
    one track, so nothing else has to move."""
    import mido

    mf = mido.MidiFile(str(path))
    for track in mf.tracks:
        for msg in track:
            if msg.type in ("note_on", "note_off") and msg.note in channel_by_pitch:
                msg.channel = channel_by_pitch[msg.note]
    mf.save(str(path))


def pad_to_duration(path: Path, seconds: float, tempo: float) -> bool:
    """Extends the file's end-of-track to cover the whole recording.

    By default a MIDI file stops at its last note, so a song whose drums finish before
    the final chord produces a clip shorter than the audio. Dropped into a DAW that
    clip has no visual anchor at its right-hand edge, and lining it up by eye against
    the waveform is easy to get wrong -- which is exactly how a correct transcription
    came to look like it was missing its ending.

    Making the clip exactly as long as the audio means it can be dropped at bar one and
    left alone.
    """
    import mido

    mf = mido.MidiFile(str(path))
    target = int(round(seconds * tempo / 60.0 * mf.ticks_per_beat))
    changed = False
    for track in mf.tracks:
        total = sum(msg.time for msg in track)
        if total >= target:
            continue
        for msg in reversed(track):
            if msg.type == "end_of_track":
                msg.time += target - total
                changed = True
                break
    if changed:
        mf.save(str(path))
    return changed


def write_midi(events: List[tuple], out_path: Path, tempo: float,
               note_len: float, grid: Optional[int], split_tracks: bool = False,
               note_map: Optional[Dict[int, int]] = None,
               channel_map: Optional[Dict[int, int]] = None,
               duration: Optional[float] = None) -> None:
    import pretty_midi

    note_map = note_map or {}
    channel_map = channel_map or {}

    # pretty_midi defaults to 220 ticks per quarter, and 220 = 2*2*5*11 has no factor
    # of 3, so triplets (and 1/32 notes) cannot land on whole ticks and get rounded.
    # 960 = 2^6*3*5 is the usual DAW resolution and represents all of them exactly,
    # while shrinking the tick from 2.27 ms to 0.52 ms at 120 BPM.
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo, resolution=960)

    def make_note(t, pitch, vel):
        start = quantize(t, tempo, grid) if grid else t
        return pretty_midi.Note(velocity=int(vel), pitch=int(note_map.get(pitch, pitch)),
                                start=float(start), end=float(start + note_len))

    ordered = sorted(events)
    if not split_tracks:
        kit = pretty_midi.Instrument(program=0, is_drum=True, name="drum2midi")
        kit.notes.extend(make_note(*e) for e in ordered)
        midi.instruments.append(kit)
    else:
        # One MTrk per drum, which most DAWs turn into separate tracks on import -
        # handy when each piece goes to its own sampler.
        assigned = set()
        for name, pitches in GM_GROUPS:
            group = [e for e in ordered if e[1] in pitches]
            assigned.update(pitches)
            if not group:
                continue
            inst = pretty_midi.Instrument(program=0, is_drum=True, name=name)
            inst.notes.extend(make_note(*e) for e in group)
            midi.instruments.append(inst)
        rest = [e for e in ordered if e[1] not in assigned]
        if rest:
            inst = pretty_midi.Instrument(program=0, is_drum=True, name="Other")
            inst.notes.extend(make_note(*e) for e in rest)
            midi.instruments.append(inst)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    midi.write(str(out_path))

    if channel_map:
        # Accept either number: --channels 35=3 and --channels 36=3 both mean "kick",
        # since the default note map now rewrites 35 to 36 and a user may reasonably
        # name the drum by what they typed or by what they see in the DAW.
        final = {}
        for p, ch in channel_map.items():
            final[note_map.get(p, p)] = ch
            final.setdefault(p, ch)
        apply_channels(out_path, final)

    if duration:
        pad_to_duration(out_path, duration, tempo)


# --------------------------------------------------------------------------

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".aif", ".m4a", ".ogg", ".opus"}


def collect_inputs(src: Path) -> List[Path]:
    """One file, or every audio file in a folder. Sorted so runs are reproducible."""
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted(p for p in src.iterdir()
                      if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    return []


def main() -> int:
    p = argparse.ArgumentParser(
        description="Convert a mixed drum track to MIDI (ADTOF + a drum separator).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("input", type=Path,
                   help="Drum audio file, or a folder to convert in bulk")
    p.add_argument("-o", "--out", type=Path, default=None,
                   help="Output .mid path, or output folder when the input is a folder")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "xpu", "mps"],
                   help="auto: separation on the GPU, transcription on the CPU, because "
                        "recurrent layers measured 4.1x slower on an Intel GPU. "
                        "Any explicit value is used for both stages")
    p.add_argument("--separator", default="uvr",
                   choices=["larsnet", "drumsep", "uvr", "hybrid"],
                   help="uvr (default): MDX23C, 6 stems including ride/crash. On MDB "
                        "Drums MICRO F1 0.882 vs 0.860 for larsnet, ghost notes 0.718 vs "
                        "0.468, ride/crash 0.917 vs 0.139. larsnet: 5 stems, faster than "
                        "real time, but no ride and weaker on ghost notes")
    p.add_argument("--from-song", action="store_true",
                   help="Input is a full mix: pull the drum stem out first")
    p.add_argument("--extractor", default=None, metavar="MODEL",
                   help="Model used by --from-song, e.g. htdemucs_ft.yaml. Default "
                        "htdemucs.yaml; compare_extractors.py scores the alternatives")
    p.add_argument("--no-split-ride", action="store_true",
                   help="Keep every cymbal as crash even when a ride stem is available")
    p.add_argument("--no-pad-to-audio", action="store_true",
                   help="End the MIDI at the last note instead of at the end of the "
                        "recording. The default matches the audio length so the clip "
                        "lines up in a DAW without being dragged into place")
    p.add_argument("--no-separate", action="store_true",
                   help="Skip LarsNet: much faster, but flat velocity 100 and no articulations")
    p.add_argument("--wiener", type=float, default=1.0,
                   help="Alpha-Wiener exponent; reduces bleed between stems. 0 disables")
    p.add_argument("--chunk-seconds", type=float, default=48.0, help="Separation chunk size")
    p.add_argument("--thresholds", default="",
                   help="Per-class peak-pick thresholds kick,snare,tom,hat,cymbal. "
                        "Giving these replaces the adaptive tom policy, TOM_CEILING "
                        "included, so runs using them are not comparable with the "
                        "default ones on toms")
    p.add_argument("--fixed-tom-threshold", action="store_true",
                   help="Use a fixed tom threshold instead of adapting it per track")
    p.add_argument("--fuse-stem-onsets", default="auto", choices=["auto", "on", "off"],
                   help="Add onsets found in the separated stems. auto: on for "
                        "--separator uvr, where it measured MICRO 0.692 -> 0.801; off "
                        "for LarsNet, whose stems are too bleedy and make it worse")
    p.add_argument("--dyn-range", type=float, default=None,
                   help="dB below the loudest hit that maps to the lowest velocity. "
                        "Default is per-drum (kick 18, snare/toms/hats 36, cymbals 48)")
    p.add_argument("--vel-min", type=int, default=15)
    p.add_argument("--vel-max", type=int, default=127)
    p.add_argument("--max-toms", type=int, default=3, help="Max distinct toms to detect (1 disables)")
    p.add_argument("--open-hat-ratio", type=float, default=0.30,
                   help="Tail/attack energy ratio above which a hi-hat counts as open")
    p.add_argument("--pedal", default="heuristic", choices=["auto", "off", "heuristic"],
                   help="heuristic (default): a conservative rule, 0.752 overall hi-hat "
                        "accuracy on real MDB recordings. auto: the learned model, 0.787 "
                        "on GMD electronic kits but only 0.502 on real recordings, where "
                        "it calls a third of closed hats pedal. off: only 42/46")
    p.add_argument("--no-learned", action="store_true",
                   help="Ignore models/ entirely and use the handcrafted formulas")
    p.add_argument("--pedal-centroid", type=float, default=0.80,
                   help="Heuristic pedal: max spectral centroid, relative to struck hats")
    p.add_argument("--pedal-level", type=float, default=0.50,
                   help="Heuristic pedal: max peak level, relative to struck hats")
    p.add_argument("--tom-sensitivity", type=float, default=0.25,
                   help="Tom-stem onset threshold, relative to the loudest tom hit")
    p.add_argument("--rescue-toms", default="auto", choices=["auto", "on", "off"],
                   help="Recover toms that were labelled kick or snare. auto: on for "
                        "larsnet, off for uvr — with a clean tom stem the trick adds more "
                        "false positives than it finds (0.605 -> 0.398)")
    p.add_argument("--no-rescue-toms", action="store_true",
                   help="Same as --rescue-toms off (kept for compatibility)")
    p.add_argument("--no-split-toms", action="store_true")
    p.add_argument("--no-split-hats", action="store_true")
    p.add_argument("--quantize", type=int, default=None, metavar="N",
                   help="Snap to a 1/N grid (16 = sixteenths, 12 = eighth triplets, "
                        "24 = sixteenth triplets). Off by default")
    p.add_argument("--tempo", type=float, default=None, help="Override detected tempo")
    p.add_argument("--note-len", type=float, default=0.06)
    p.add_argument("--split-tracks", action="store_true",
                   help="Write one MIDI track per drum instead of one for the whole kit; "
                        "most DAWs then import them as separate tracks")
    p.add_argument("--channels", default="",
                   help="Per-drum MIDI channel, e.g. 'kick=0,snare=1' or '36=0,38=1'. "
                        "Default is channel 9 for everything")
    p.add_argument("--notes", default="",
                   help="Remap output pitches for samplers that do not follow General "
                        "MIDI. Each side accepts a number, and the left also accepts a "
                        "drum name, so 'kick=C1', 'kick=36' and '35=36' are the same "
                        "thing. Note names use the Cubase/Logic octave (C1 = 36)")
    p.add_argument("--keep-stems", type=Path, default=None, help="Directory to write separated stems")
    args = p.parse_args()

    src = args.input.resolve()
    if not src.exists():
        log(f"ERROR: input not found: {src}")
        return 1

    batch = collect_inputs(src)
    if not batch:
        if src.is_dir():
            log(f"ERROR: no audio files in {src}")
            log(f"       looked for: {', '.join(sorted(AUDIO_EXTS))}")
        else:
            log(f"ERROR: not an audio file: {src}")
        return 1

    if len(batch) > 1:
        out_dir = (args.out or src).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"Batch: {len(batch)} files -> {out_dir}\n")
        failed = []
        started = time.time()
        for i, item in enumerate(batch, 1):
            log(f"=== [{i}/{len(batch)}] {item.name}")
            try:
                rc = convert_one(args, item, out_dir / f"{item.stem}.mid")
            except Exception as exc:
                log(f"ERROR: {type(exc).__name__}: {exc}")
                rc = 1
            if rc != 0:
                failed.append(item.name)
            if i < len(batch):
                per = (time.time() - started) / i
                log(f"    ~{per * (len(batch) - i) / 60:.0f} min left\n")
        log(f"\nBatch done: {len(batch) - len(failed)}/{len(batch)} converted")
        if failed:
            log(f"failed: {', '.join(failed)}")
            return 1
        return 0

    out_path = (args.out or src.with_suffix(".mid")).resolve()
    if out_path.is_dir():
        out_path = out_path / f"{src.stem}.mid"
    return convert_one(args, src, out_path)


def convert_one(args, src: Path, out_path: Path) -> int:
    """Converts a single file. Returns 0 on success, so batch mode can count failures."""
    thresholds = None
    if args.thresholds:
        thresholds = [float(v) for v in args.thresholds.split(",") if v.strip()]
        if len(thresholds) != 5:
            log("ERROR: --thresholds needs exactly 5 comma-separated values")
            return 1

    try:
        channel_map = parse_pairs(args.channels, 0, 15, "channel") if args.channels else {}
        note_map = dict(OUTPUT_NOTES)
        if args.notes:
            note_map.update(parse_pairs(args.notes, 0, 127, "note"))
    except ValueError as exc:
        log(f"ERROR: {exc}")
        return 1

    import librosa
    import soundfile as sf

    t0 = time.time()

    missing = separator_missing(args)
    if missing:
        what, why, how, detail = missing
        log(f"ERROR: --separator {args.separator} needs {what}, and {why}.")
        if detail:
            log(f"       {detail}")
        log(f"       {how}")
        log("       Or run with --no-separate to skip separation entirely; "
            "velocity is then flat.")
        return 1

    from devices import SEPARATOR, TRANSCRIBER, pick_devices, summary, unused_gpu
    picked = pick_devices(args.device)
    dev_sep, dev_trans = picked[SEPARATOR], picked[TRANSCRIBER]
    log(f"      {summary(args.device)}")
    idle_gpu = unused_gpu()
    if idle_gpu and dev_sep == "cpu":
        card, how = idle_gpu
        log(f"      note: {card} is installed and unused.")
        for line in how.splitlines():
            log(f"      {line.strip()}" if line.startswith(" ") else f"      {line}")
    if dev_trans == "cpu":
        from cpu_threads import apply as apply_threads
        apply_threads(verbose=True)

    if args.from_song:
        log("[0/4] Pulling the drum stem out of the full mix ...")
        try:
            src = extract_drums(src, dev_sep, ROOT / "work" / f"{src.stem}_drums.wav",
                                args.extractor)
            log(f"      drum stem -> {src}")
        except Exception as exc:
            log(f"ERROR: could not extract drums: {exc}")
            return 1

    log(f"[1/4] Transcribing {src.name} with ADTOF ...")
    # Which tom policy ran is part of the result. An explicit --thresholds vector
    # silently replaces the adaptive policy including TOM_CEILING, and a benchmark run
    # that does not say so produces figures from one policy family that read as though
    # they belonged to another.
    if thresholds is not None:
        log(f"      thresholds {list(thresholds)} given explicitly, so the adaptive "
            f"tom policy (TOM_CEILING={TOM_CEILING}) is NOT in effect")
    elif args.fixed_tom_threshold:
        log(f"      fixed tom threshold {DEFAULT_THRESHOLDS[2]}, adaptive policy off")
    else:
        log(f"      adaptive tom threshold, {TOM_PERCENTILE}th percentile bounded to "
            f"[{TOM_FLOOR}, {TOM_CEILING}]")
    onsets = transcribe(src, dev_trans, thresholds,
                        adaptive_toms=not args.fixed_tom_threshold)
    total_hits = sum(len(v) for v in onsets.values())
    if total_hits == 0:
        peak = max(_LAST_PEAKS) if _LAST_PEAKS else None
        lowest = min(thresholds) if thresholds else min(DEFAULT_THRESHOLDS)
        if peak is None:
            log("ERROR: no drum hits detected. Try lowering --thresholds.")
        elif peak < 0.05:
            # Not a threshold problem in any useful sense. A threshold below this would
            # be picking noise, and saying "lower it" invites exactly that: on soft
            # mallet material the model peaks around 0.008 and 0.05 still finds nothing.
            log(f"ERROR: no drum hits detected. The model's strongest response anywhere "
                f"in this file is {peak:.3f}, which is essentially nothing -- it does "
                f"not recognise this as drums at all, and a threshold low enough to "
                f"pick it up would be picking noise. Soft mallet or brush material is "
                f"the usual cause.")
        elif peak < lowest:
            log(f"ERROR: no drum hits detected. The model's strongest response anywhere "
                f"in this file is {peak:.3f}, below the lowest threshold in use "
                f"({lowest:.2f}), so lowering them to about {peak * 0.8:.3f} is worth "
                f"one try.")
        else:
            # Peak clears the threshold and nothing was still picked, so the limit is
            # peak-picking rather than the threshold, and telling someone to lower it
            # sends them to 0.05 and still nothing -- which is what this file does.
            log(f"ERROR: no drum hits detected. The model's strongest response is "
                f"{peak:.3f}, which already clears the threshold ({lowest:.2f}), so "
                f"lowering it will not help: nothing in this recording looks like a "
                f"drum onset to the model. Soft mallet or brush material is the usual "
                f"cause.")
        return 1
    log(f"      {total_hits} hits detected")
    # The classes are known the moment ADTOF returns, minutes before separation
    # finishes. Printing the breakdown here lets a listener judge the transcription
    # immediately instead of staring at a progress bar. Names are deliberately the
    # coarse ADTOF classes: articulation is not decided until stage 3.
    early_names = {ADTOF_KICK: "Kick", ADTOF_SNARE: "Snare", ADTOF_TOM: "Toms",
                   ADTOF_HAT: "Hi-hat", ADTOF_CYM: "Cymbals"}
    for cls, times in sorted(onsets.items(), key=lambda kv: -len(kv[1])):
        if times:
            log(f"      found {early_names.get(cls, str(cls))}: {len(times)}")

    audio, _ = librosa.load(str(src), sr=SR, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    duration = audio.shape[-1] / SR

    stems: Dict[str, np.ndarray] = {}
    if not args.no_separate:
        mode = args.separator
        log(f"[2/4] Separating {duration:.1f}s ({mode}) ...")
        wiener = None if args.wiener == 0 else args.wiener

        if mode in ("larsnet", "hybrid"):
            stems = separate(audio, dev_sep, wiener, args.chunk_seconds)

        if mode == "uvr":
            try:
                stems = separate_uvr(src, dev_sep, audio.shape[-1])
                log("      MDX23C: 6 stems incl. separate ride and crash")
            except Exception as exc:
                log(f"ERROR: UVR separator failed: {exc}")
                return 1

        if mode in ("drumsep", "hybrid"):
            try:
                ds = separate_drumsep(src, dev_sep, audio.shape[-1])
            except Exception as exc:
                if mode == "drumsep":
                    log(f"ERROR: DrumSep failed: {exc}")
                    return 1
                log(f"      WARNING: DrumSep unavailable ({exc}); keeping LarsNet stems")
                ds = {}
            if mode == "drumsep":
                stems = ds
            else:
                # Measured on both test tracks: DrumSep resolves kick and snare with
                # less bleed, while LarsNet is far cleaner on cymbals and is the only
                # one that isolates the hi-hat at all. Take the best of each.
                for name in ("kick", "snare"):
                    if name in ds:
                        stems[name] = ds[name]
                if ds:
                    log("      hybrid: kick+snare from DrumSep, rest from LarsNet")

        if args.keep_stems:
            outdir = args.keep_stems.resolve()
            outdir.mkdir(parents=True, exist_ok=True)
            for name, wav in stems.items():
                sf.write(outdir / f"{name}.wav", wav.T, SR)
            log(f"      stems written to {outdir}")
    else:
        log("[2/4] Separation skipped (--no-separate)")

    log("[3/4] Deriving velocities and articulations ...")

    fuse_mode = args.fuse_stem_onsets
    if fuse_mode == "auto":
        # only worth it with a separator clean enough to survive onset detection
        fuse_mode = "on" if args.separator == "uvr" else "off"
    if fuse_mode == "on" and stems:
        added = fuse_stem_onsets(onsets, stems, dev_trans, thresholds)
        if added:
            log(f"      stem fusion: +{added} onsets the mix pass missed")

    learned = load_learned(not args.no_learned)
    vel_models = (learned["velocity"] or {}).get("models", {})
    pedal_model = learned["pedal"] if args.pedal == "auto" else None
    if vel_models or pedal_model:
        bits = []
        if vel_models:
            bits.append("velocity for " + "/".join(sorted(vel_models)))
        if pedal_model:
            bits.append("pedal hi-hat")
        log(f"      learned models in use: {', '.join(bits)}")

    events: List[tuple] = []
    degraded: List[str] = []
    for cls, times in onsets.items():
        if not times:
            continue
        stem = stems.get(CLASS_TO_STEM[cls])

        if stem is not None and not stem_is_usable(stem, audio):
            degraded.append(CLASS_TO_STEM[cls])
            stem = None

        if stem is None:
            # No separation, or the demixer returned an empty stem: take the level
            # from the mix instead of inventing dynamics out of a noise floor.
            src_sig = audio if stems else None
            if src_sig is None:
                events.extend((t, cls, 100) for t in times)
            else:
                fallback_dyn = args.dyn_range if args.dyn_range is not None \
                    else DYN_RANGE.get(CLASS_TO_STEM[cls], 36.0)
                events.extend(zip(times, [cls] * len(times),
                                  velocities(times, src_sig, fallback_dyn,
                                             args.vel_min, args.vel_max)))
            continue

        stem_name = CLASS_TO_STEM[cls]
        needs_features = stem_name in vel_models or (cls == ADTOF_HAT and pedal_model)
        feats = None
        if needs_features:
            try:
                from features import features_for
                feats = features_for(_mono(stem), times, _mono(audio))
            except Exception as exc:
                log(f"      WARNING: feature extraction failed ({exc}); using formula")
                feats = None

        if stem_name in vel_models and feats is not None and len(feats):
            raw = vel_models[stem_name].predict(feats)
            vels = [int(np.clip(round(float(v)), 1, 127)) for v in raw]
        else:
            dyn = args.dyn_range if args.dyn_range is not None \
                else DYN_RANGE.get(stem_name, 36.0)
            vels = velocities(times, stem, dyn, args.vel_min, args.vel_max)

        if cls == ADTOF_HAT and not args.no_split_hats:
            pitches = split_hats(times, stem, args.open_hat_ratio,
                                 detect_pedal=(args.pedal == "heuristic"),
                                 pedal_centroid=args.pedal_centroid,
                                 pedal_level=args.pedal_level)
            if pedal_model is not None and feats is not None and len(feats):
                prob = pedal_model["model"].predict_proba(feats)[:, 1]
                for i, p in enumerate(prob):
                    if pitches[i] == ADTOF_HAT and p >= pedal_model["threshold"]:
                        pitches[i] = 44
        else:
            # Toms keep the generic pitch here; they are split after the rescue pass,
            # once the full set of tom hits is known.
            pitches = [cls] * len(times)

        events.extend(zip(times, pitches, vels))

    if degraded:
        log(f"      WARNING: LarsNet returned an empty stem for {', '.join(sorted(set(degraded)))}; "
            f"velocity taken from the mix and articulation splitting skipped there.")

    tom_stem = stems.get("toms")
    tom_ok = tom_stem is not None and stem_is_usable(tom_stem, audio)

    # Rescue was designed for LarsNet, whose tom stem is often empty: it takes timing
    # from the stem and steals hits from kick and snare. On MDX23C's clean stems (tom
    # contrast 252x) it only adds false positives: toms 0.605 -> 0.398, MICRO 0.882 -> 0.876.
    rescue_default = args.separator not in ("uvr",)
    do_rescue = rescue_default if args.rescue_toms == "auto" else (args.rescue_toms == "on")
    if args.no_rescue_toms:
        do_rescue = False

    if tom_ok and do_rescue:
        events, added, reclaimed = rescue_toms(
            events, tom_stem, stems, args.tom_sensitivity,
            args.dyn_range if args.dyn_range is not None else DYN_RANGE["toms"],
            args.vel_min, args.vel_max)
        if added or reclaimed:
            log(f"      tom rescue: +{added} added, {reclaimed} reclaimed from kick/snare")

    if tom_ok and not args.no_split_toms and args.max_toms > 1:
        idx = [i for i, (_, pitch, _) in enumerate(events) if pitch == ADTOF_TOM]
        if idx:
            split = split_toms([events[i][0] for i in idx], tom_stem, args.max_toms)
            for i, pitch in zip(idx, split):
                events[i] = (events[i][0], pitch, events[i][2])

    if "ride" in stems and not args.no_split_ride:
        moved = split_ride(events, stems)
        if moved:
            log(f"      ride/crash: {moved} hits reassigned to ride")

    tempo = args.tempo or estimate_tempo(audio)
    log(f"[4/4] Writing MIDI at {tempo:.1f} BPM"
        + (f", quantized to 1/{args.quantize}" if args.quantize else ", unquantized"))
    # The file is made as long as the recording unless asked otherwise, so the clip
    # lines up with the audio in a DAW instead of stopping at the last drum hit.
    duration = None if args.no_pad_to_audio else audio.shape[-1] / SR
    write_midi(events, out_path, tempo, args.note_len, args.quantize, args.split_tracks,
               note_map=note_map, channel_map=channel_map, duration=duration)
    user_notes = {k: v for k, v in note_map.items() if OUTPUT_NOTES.get(k) != v}
    if channel_map or user_notes:
        bits = []
        if channel_map:
            bits.append(f"channels: {len(channel_map)}")
        if user_notes:
            bits.append(f"notes: {len(user_notes)}")
        log(f"      overrides - {', '.join(bits)}")

    log(f"\nDone in {time.time() - t0:.1f}s -> {out_path}")
    log(f"{'instrument':<16}{'hits':>6}{'vel min':>9}{'vel max':>9}")
    log("-" * 40)
    by_pitch: Dict[int, List[int]] = {}
    for _, pitch, vel in events:
        # report the note that was actually written, not the internal class id
        by_pitch.setdefault(note_map.get(pitch, pitch), []).append(vel)
    for pitch in sorted(by_pitch):
        v = by_pitch[pitch]
        log(f"{GM_NAMES.get(pitch, str(pitch)):<16}{len(v):>6}{min(v):>9}{max(v):>9}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
