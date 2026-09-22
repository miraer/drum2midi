"""Smoke tests for drum2midi.

Fast, self-contained checks that the pipeline is wired up correctly. No datasets and no
model downloads are required beyond what the pipeline already needs, and the audio is
synthesised on the fly, so this runs in seconds.

This is not an accuracy benchmark - accuracy is measured by benchmark_mdb.py and friends.
These tests only catch the kind of breakage that makes everything silently wrong: a
mangled argument, a broken MIDI writer, a channel override that never lands.

    python test_smoke.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
SR = 44100
PY = sys.executable

_tmp: Path | None = None
_fixture: Path | None = None
PASSED: list[str] = []
FAILED: list[tuple[str, str]] = []


def test(fn):
    """Registers a test; each one reports pass/fail without aborting the run."""
    def wrapper():
        name = fn.__name__.replace("test_", "").replace("_", " ")
        try:
            fn()
            PASSED.append(name)
            print(f"  PASS  {name}")
        except AssertionError as exc:
            FAILED.append((name, str(exc)))
            print(f"  FAIL  {name}: {exc}")
        except Exception:
            FAILED.append((name, traceback.format_exc(limit=2)))
            print(f"  ERROR {name}")
            print("        " + traceback.format_exc(limit=2).strip().replace("\n", "\n        "))
    wrapper.is_test = True
    wrapper.__name__ = fn.__name__
    return wrapper


def make_fixture() -> Path:
    """A short synthetic groove: four kicks, two snares, eight hats, one crash."""
    global _fixture
    if _fixture is not None:
        return _fixture
    import soundfile as sf

    rng = np.random.default_rng(0)
    beat = 0.5
    total = int(4 * beat * 2 * SR)
    mix = np.zeros(total)

    def place(sig, t):
        i = int(t * SR)
        n = min(len(sig), total - i)
        if n > 0:
            mix[i:i + n] += sig[:n]

    def env(n, tau):
        return np.exp(-np.arange(n) / (tau * SR))

    for bar in range(2):
        t0 = bar * 4 * beat
        for off in (0.0, 2 * beat):
            n = int(0.3 * SR)
            f = 50 + 60 * np.exp(-np.arange(n) / SR / 0.02)
            place(np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.08), t0 + off)
        for off in (beat, 3 * beat):
            n = int(0.2 * SR)
            place(rng.normal(0, 1, n) * env(n, 0.05) * 0.8, t0 + off)
        for i in range(8):
            n = int(0.08 * SR)
            hat = rng.normal(0, 1, n)
            spec = np.fft.rfft(hat)
            spec[np.fft.rfftfreq(n, 1 / SR) < 6000] *= 0.05
            place(np.fft.irfft(spec, n) * env(n, 0.015) * 0.5, t0 + i * 0.5 * beat)
    n = int(1.5 * SR)
    crash = rng.normal(0, 1, n)
    spec = np.fft.rfft(crash)
    spec[np.fft.rfftfreq(n, 1 / SR) < 3000] *= 0.1
    place(np.fft.irfft(spec, n) * env(n, 0.6) * 0.6, 0.0)

    mix /= np.max(np.abs(mix)) * 1.1
    path = _tmp / "fixture.wav"
    sf.write(path, np.stack([mix, mix], axis=1), SR)
    _fixture = path
    return path


def run_pipeline(*args, expect_ok=True) -> tuple[Path, str]:
    out = _tmp / f"out_{abs(hash(args)) % 10**8}.mid"
    cmd = [PY, str(ROOT / "drum2midi.py"), str(make_fixture()), "-o", str(out),
           "--device", "cpu", *args]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", cwd=str(ROOT))
    log = (res.stdout or "") + (res.stderr or "")
    if expect_ok:
        assert res.returncode == 0, f"exit {res.returncode}\n{log[-400:]}"
        assert out.exists(), f"no MIDI written\n{log[-400:]}"
    return out, log


# ---------------------------------------------------------------- tests

@test
def test_imports():
    """Every module must import cleanly - catches syntax damage from edits."""
    import importlib
    for name in ("features", "trigger", "drum2midi"):
        importlib.import_module(name)


@test
def test_basic_conversion():
    out, _ = run_pipeline("--no-separate")
    import pretty_midi
    notes = pretty_midi.PrettyMIDI(str(out)).instruments[0].notes
    assert len(notes) >= 10, f"only {len(notes)} notes from a 4-second groove"
    assert all(0 <= n.pitch <= 127 for n in notes), "pitch out of range"
    assert all(1 <= n.velocity <= 127 for n in notes), "velocity out of range"
    assert all(n.end > n.start for n in notes), "zero or negative note length"


@test
def test_notes_are_general_midi_drums():
    out, _ = run_pipeline("--no-separate")
    import pretty_midi
    inst = pretty_midi.PrettyMIDI(str(out)).instruments[0]
    assert inst.is_drum, "instrument is not flagged as a drum kit"
    known = {35, 36, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51}
    used = {n.pitch for n in inst.notes}
    assert used <= known, f"unexpected pitches: {sorted(used - known)}"


@test
def test_resolution_supports_triplets():
    """PPQ must divide by 3, or triplets silently land off-grid."""
    out, _ = run_pipeline("--no-separate")
    import mido
    ppq = mido.MidiFile(str(out)).ticks_per_beat
    assert ppq % 3 == 0, f"ppq {ppq} is not divisible by 3"
    assert ppq % 32 == 0, f"ppq {ppq} cannot represent 1/32 notes"


@test
def test_tempo_is_written():
    out, _ = run_pipeline("--no-separate", "--tempo", "137")
    import mido
    tempos = [m.tempo for tr in mido.MidiFile(str(out)).tracks
              for m in tr if m.type == "set_tempo"]
    assert tempos, "no set_tempo event"
    bpm = 60_000_000 / tempos[0]
    assert abs(bpm - 137) < 0.5, f"tempo is {bpm:.1f}, expected 137"


@test
def test_quantize_snaps_to_grid():
    out, _ = run_pipeline("--no-separate", "--tempo", "120", "--quantize", "16")
    import mido
    mf = mido.MidiFile(str(out))
    step = mf.ticks_per_beat / 4          # a sixteenth
    ticks = []
    for tr in mf.tracks:
        acc = 0
        for msg in tr:
            acc += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                ticks.append(acc)
    assert ticks, "no notes"
    off = [t % step for t in ticks]
    assert max(off) < 1e-6, f"notes off the grid, worst offset {max(off)} ticks"


@test
def test_triplet_quantize():
    out, _ = run_pipeline("--no-separate", "--tempo", "120", "--quantize", "12")
    import mido
    mf = mido.MidiFile(str(out))
    step = mf.ticks_per_beat * 4 / 12
    ticks = []
    for tr in mf.tracks:
        acc = 0
        for msg in tr:
            acc += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                ticks.append(acc)
    off = [t % step for t in ticks]
    assert max(off) < 1e-6, f"triplets off the grid, worst {max(off)}"


@test
def test_channel_override():
    """--channels names a drum; the kick is written as 36 but 35 must still address it."""
    out, _ = run_pipeline("--no-separate", "--channels", "35=3,38=4")
    import mido
    seen = {}
    for tr in mido.MidiFile(str(out)).tracks:
        for m in tr:
            if m.type == "note_on" and m.velocity > 0:
                seen.setdefault(m.note, set()).add(m.channel)
    for kick in (35, 36):
        if kick in seen:
            assert seen[kick] == {3}, f"kick ({kick}) on channels {seen[kick]}, expected 3"
    if 38 in seen:
        assert seen[38] == {4}, f"snare on channels {seen[38]}, expected 4"
    for pitch, chans in seen.items():
        if pitch not in (35, 36, 38):
            assert chans == {9}, f"note {pitch} moved off channel 9: {chans}"


@test
def test_kick_is_written_as_general_midi_36():
    """36 is "Bass Drum 1" and is what hardware and DAWs expect.

    Surveyed in gm_note_survey.py: 445494 notes from 1150 Roland TD-11 performances
    use 36 for every kick and 35 for none.
    """
    out, _ = run_pipeline("--no-separate")
    import pretty_midi
    pitches = {n.pitch for inst in pretty_midi.PrettyMIDI(str(out)).instruments
               for n in inst.notes}
    assert 35 not in pitches, "kick should be written as 36, not 35"
    assert 36 in pitches, f"no kick in output, got {sorted(pitches)}"

    # and the user must still be able to ask for the other number
    out2, _ = run_pipeline("--no-separate", "--notes", "35=35")
    pitches2 = {n.pitch for inst in pretty_midi.PrettyMIDI(str(out2)).instruments
                for n in inst.notes}
    assert 35 in pitches2, "--notes 35=35 should restore the old numbering"


@test
def test_note_override():
    out, _ = run_pipeline("--no-separate", "--notes", "38=40")
    import pretty_midi
    pitches = {n.pitch for n in pretty_midi.PrettyMIDI(str(out)).instruments[0].notes}
    assert 38 not in pitches, "note 38 should have been remapped away"


@test
def test_split_tracks():
    out, _ = run_pipeline("--no-separate", "--split-tracks")
    import mido
    mf = mido.MidiFile(str(out))
    named = [tr.name for tr in mf.tracks if tr.name]
    assert len(mf.tracks) >= 3, f"only {len(mf.tracks)} tracks"
    assert any(n in ("Kick", "Snare", "Hi-hat", "Toms", "Cymbals") for n in named), \
        f"no per-drum track names, got {named}"


@test
def test_bad_channel_is_rejected():
    _, log = run_pipeline("--no-separate", "--channels", "35=99", expect_ok=False)
    assert "outside" in log or "ERROR" in log, f"bad channel accepted:\n{log[-300:]}"


@test
def test_bad_tempo_is_rejected():
    out, log = run_pipeline("--no-separate", "--thresholds", "0.1,0.2", expect_ok=False)
    assert "ERROR" in log, "malformed --thresholds was accepted"


@test
def test_feature_extraction_shape():
    from features import FEATURE_NAMES, features_for
    sig = np.random.default_rng(1).normal(0, 0.1, SR * 2).astype(np.float32)
    sig[SR // 2] = 1.0
    feats = features_for(sig, [0.5, 1.0], sig)
    assert feats.shape == (2, len(FEATURE_NAMES)), f"shape {feats.shape}"
    assert np.isfinite(feats).all(), "non-finite values in features"


@test
def test_trigger_finds_hits():
    from trigger import trigger
    rng = np.random.default_rng(2)
    sig = np.zeros(SR * 2, dtype=np.float32)
    times = [0.2, 0.7, 1.2, 1.7]
    for t in times:
        i = int(t * SR)
        n = int(0.05 * SR)
        sig[i:i + n] += (rng.normal(0, 1, n) *
                         np.exp(-np.arange(n) / (0.01 * SR))).astype(np.float32)
    found = trigger(sig, sensitivity=2.0)
    assert len(found) >= 3, f"found {len(found)} of 4 clear hits"
    for t in times[:3]:
        assert min(abs(found - t)) < 0.05, f"missed the hit at {t}s"


def load_gui():
    """Imports drum2midi_gui.pyw as a module.

    The loader has to be named explicitly: ".pyw" is only a recognised source suffix
    on Windows, so on Linux spec_from_file_location returns None and the import fails
    with a bare AttributeError about 'NoneType' having no 'loader'.

    The module is registered before it runs because dataclasses look their own module
    up in sys.modules while the class is being built.
    """
    import importlib.util
    from importlib.machinery import SourceFileLoader
    if "gui" in sys.modules:
        return sys.modules["gui"]
    path = ROOT / "drum2midi_gui.pyw"
    spec = importlib.util.spec_from_loader("gui", SourceFileLoader("gui", str(path)))
    gui = importlib.util.module_from_spec(spec)
    sys.modules["gui"] = gui
    try:
        spec.loader.exec_module(gui)
    except BaseException:
        del sys.modules["gui"]
        raise
    return gui


def gui_window():
    """A real window, offscreen, that cannot touch the user's saved settings.

    Offscreen is Qt's own headless platform, so this runs on CI without a display.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    gui = load_gui()
    gui.SETTINGS = _tmp / "gui_settings.json"
    gui.make_app(["test"])
    return gui, gui.Window()


@test
def test_gui_builds():
    """The window must construct and assemble a valid command line from what it shows."""
    gui, win = gui_window()
    win._set_input(make_fixture())
    assert win.phase == "ready", f"choosing a file left the window in {win.phase}"
    assert win.c.output.endswith(".mid"), f"no default output: {win.c.output!r}"

    win.sep_cards["none"].clicked.emit("none")
    cmd = gui.build_command(win.c)
    assert "--no-separate" in cmd, f"separator flag missing: {cmd}"
    assert cmd[cmd.index("-o") + 1] == win.c.output

    # The channel cell shows 1-16, as every DAW does; the CLI takes 0-15. Picking
    # "5" in the snare row must reach the pipeline as 4, not 5.
    win.rows[38].chan.setCurrentIndex(4)
    cmd = gui.build_command(win.c)
    assert "--channels" in cmd, f"channel override not passed through: {cmd}"
    assert cmd[cmd.index("--channels") + 1] == "38=4", cmd
    win.rows[36].note.setCurrentIndex(35)
    cmd = gui.build_command(win.c)
    assert cmd[cmd.index("--notes") + 1] == "36=35", cmd

    for theme in ("light", "dark"):
        win._set_theme(theme)
        assert win.grab().width() > 0, f"{theme} theme did not render"
    win.close()


@test
def test_gui_options_follow_the_separator():
    """An option the separator cannot use is shown off and left out of the command,
    but the preference survives switching back."""
    gui = load_gui()
    c = gui.Choices(input="a.wav", output="a.mid", separator="uvr", rescue=True,
                    fuse=True)
    cmd = gui.build_command(c)
    assert "--no-rescue-toms" in cmd, "rescue toms reached MDX23C, where it hurts toms"
    assert cmd[cmd.index("--fuse-stem-onsets") + 1] == "on"

    c.separator = "larsnet"
    cmd = gui.build_command(c)
    assert "--no-rescue-toms" not in cmd, "rescue toms was dropped for LarsNet"
    assert cmd[cmd.index("--fuse-stem-onsets") + 1] == "off", "fusion is MDX23C-only"
    assert c.fuse, "the stored preference was overwritten"

    c.quant, c.triplets = "16", True
    cmd = gui.build_command(c)
    assert cmd[cmd.index("--quantize") + 1] == "24", f"1/16 triplets: {cmd}"
    c.quant = "32"
    cmd = gui.build_command(c)
    assert cmd[cmd.index("--quantize") + 1] == "32", "triplets leaked into 1/32"


@test
def test_gui_reads_the_summary_and_the_midi_back():
    """The result table and the timeline both have to survive a note override.

    The pipeline reports each row by the note it *wrote*, so with the kick moved to 35
    the summary says "Kick" for pitch 35 and nothing for 36; the window must still put
    those hits on the kick row and the kick lane.
    """
    import mido
    gui, win = gui_window()
    win._set_input(make_fixture())
    win.rows[36].note.setCurrentIndex(60)        # kick written as 60, a non-GM number
    for line in ("instrument        hits  vel min  vel max\n",
                 "----------------------------------------\n",
                 "Snare                2       80      120\n",
                 "60                   4       90      127\n",
                 "\n"):
        win._handle(line)
    assert win.results.get(36) == (4, 90, 127), f"kick row lost: {win.results}"
    assert win.results.get(38) == (2, 80, 120), f"snare row lost: {win.results}"
    assert win.rows[36].hits.text() == "4"

    midi = _tmp / "readback.mid"
    mf = mido.MidiFile(ticks_per_beat=480)
    tr = mido.MidiTrack()
    mf.tracks.append(tr)
    for note, dt in ((60, 0), (38, 480), (60, 480)):
        tr.append(mido.Message("note_on", note=note, velocity=100, channel=9, time=dt))
        tr.append(mido.Message("note_off", note=note, velocity=0, channel=9, time=10))
    mf.save(str(midi))
    hits = gui.read_hits(midi, win.c)
    lanes = sorted(lane for lane, _, _ in hits)
    assert lanes == [0, 0, 1], f"hits landed on the wrong lanes: {hits}"
    win.close()


@test
def test_gui_colours_are_legible():
    """Every colour the window draws with must be visible on what it is drawn on.

    Text needs 4.5:1. The drum colours are marks rather than text -- dots, timeline
    ticks, velocity bars -- and need 3:1. The design's light hi-hat amber was 2.7:1 on
    white, which is why that one token departs from it.
    """
    gui = load_gui()

    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]

    def ratio(a, b):
        hi, lo = sorted((lum(a), lum(b)), reverse=True)
        return (hi + 0.05) / (lo + 0.05)

    for name, t in gui.THEMES.items():
        for fg, bg, need in (("text", "panel", 4.5), ("muted", "panel", 4.5),
                             ("muted", "bg", 4.5), ("accentInk", "accent", 4.5),
                             ("accent", "panel", 3.0)):
            r = ratio(t[fg], t[bg])
            assert r >= need, f"{name}: {fg} on {bg} is {r:.2f}:1, needs {need}"
        for drum in ("kick", "snare", "tom", "hat", "crash", "ride"):
            r = ratio(t[drum], t["panel"])
            assert r >= 3.0, f"{name}: {drum} on the panel is {r:.2f}:1, needs 3"


@test
def test_batch_folder():
    """A folder in, one MIDI per file out."""
    src = _tmp / "batch_in"
    src.mkdir(exist_ok=True)
    fixture = make_fixture()
    for name in ("one.wav", "two.wav"):
        shutil.copy(fixture, src / name)
    out = _tmp / "batch_out"
    res = subprocess.run(
        [PY, str(ROOT / "drum2midi.py"), str(src), "-o", str(out),
         "--device", "cpu", "--no-separate"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    assert res.returncode == 0, f"exit {res.returncode}\n{(res.stdout + res.stderr)[-300:]}"
    produced = sorted(p.name for p in out.glob("*.mid"))
    assert produced == ["one.mid", "two.mid"], f"got {produced}"


@test
def test_empty_folder_is_a_clear_error():
    empty = _tmp / "empty_in"
    empty.mkdir(exist_ok=True)
    res = subprocess.run(
        [PY, str(ROOT / "drum2midi.py"), str(empty), "--device", "cpu", "--no-separate"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(ROOT))
    log = res.stdout + res.stderr
    assert res.returncode != 0, "empty folder should fail"
    assert "no audio files" in log, f"unhelpful message:\n{log[-300:]}"
    assert "Traceback" not in log, "crashed instead of reporting the problem"


@test
def test_device_selection_is_per_stage():
    """The GPU must not be used for the transcriber unless it is fast at recurrence.

    Measured: ADTOF is 4.1x slower on this Intel GPU than on the CPU, so a naive
    "use the accelerator everywhere" rule would make the pipeline slower.
    """
    import devices

    explicit = devices.pick_devices("cpu")
    assert explicit[devices.SEPARATOR] == "cpu", "an explicit device must be obeyed"
    assert explicit[devices.TRANSCRIBER] == "cpu"

    auto = devices.pick_devices("auto")
    assert auto[devices.SEPARATOR] in devices.available_devices()
    if auto[devices.SEPARATOR] in ("xpu", "mps"):
        assert auto[devices.TRANSCRIBER] == "cpu", (
            f"recurrent stage must stay on the CPU, got {auto[devices.TRANSCRIBER]}")
    assert devices.summary("auto")


@test
def test_device_cpu_reaches_the_separator():
    """Choosing a device has to arrive at the separator, not merely be decided.

    The test above asserts pick_devices("cpu") returns cpu for the separator, and it
    always did. The choice was then dropped on the floor: _audio_separator used it
    only to decide whether to apply the XPU patch, and the command it built for
    audio-separator carried no device at all. The library then chose for itself and
    logged "CUDA is available in Torch, setting Torch device to CUDA".

    What exposed it was a benchmark, not a test. On a 4070 SUPER stage 2 took 26.6s
    under --device cpu and 26.9s under --device auto, while stages 1 and 3 slowed by
    4.1x and 3.7x as they should. The stage a GPU helps most was the one stage the
    flag never reached, and it is 68% of the run.
    """
    import drum2midi

    cmds = {d: drum2midi._separator_command(Path("a.wav"), "m.ckpt", Path("o"), d)
            for d in ("cpu", "cuda", "auto")}
    assert any("force_cpu_audio_separator" in part for part in cmds["cpu"]), (
        f"--device cpu does not reach the separator: {cmds['cpu']}")
    for dev in ("cuda", "auto"):
        assert not any("force_cpu" in part for part in cmds[dev]), (
            f"--device {dev} pins the separator to the CPU: {cmds[dev]}")
    for dev, cmd in cmds.items():
        assert "m.ckpt" in cmd, f"--device {dev} lost the model: {cmd}"

    # And the patch has to actually pin the device, not merely be referenced.
    try:
        from audio_separator.separator.separator import Separator
    except Exception:
        return          # audio-separator not installed; the command assertions stand

    import logging

    import xpu_separate

    original = Separator.setup_torch_device
    try:
        xpu_separate.force_cpu_audio_separator()

        class _Probe:
            logger = logging.getLogger("probe")

        probe = _Probe()
        Separator.setup_torch_device(probe, {})
        assert str(probe.torch_device) == "cpu", (
            f"the patch left the separator on {probe.torch_device}")
        assert probe.onnx_execution_provider == ["CPUExecutionProvider"], (
            f"onnx provider not pinned: {probe.onnx_execution_provider}")
    finally:
        Separator.setup_torch_device = original
        if hasattr(Separator, "_cpu_patched"):
            del Separator._cpu_patched


@test
def test_drum_names_and_note_names_in_overrides():
    """--notes kick=C1 must mean the same as --notes 35=36.

    Note names follow the Cubase/Logic octave, where middle C is C3, so C1 is 36.
    Checked against ReStem 2 Pro's published map, which uses exactly these names.
    """
    sys.path.insert(0, str(ROOT))
    from drum2midi import parse_note_token, parse_pairs

    restem_map = {"C1": 36, "D1": 38, "F#1": 42, "G#1": 44, "A#1": 46, "F1": 41,
                  "A1": 45, "B1": 47, "D2": 50, "C#2": 49, "D#2": 51, "C3": 60}
    for name, number in restem_map.items():
        got = parse_note_token(name)
        assert got == number, f"{name} parsed as {got}, expected {number}"

    assert parse_note_token("kick") == 35, "drum names address the internal class id"
    assert parse_note_token("hihat-open") == 46
    assert parse_pairs("kick=C1", 0, 127, "note") == {35: 36}
    assert parse_pairs("kick=36", 0, 127, "note") == parse_pairs("35=36", 0, 127, "note")
    assert parse_pairs("snare=4", 0, 15, "channel") == {38: 4}

    for bad in ("kick=Z9", "wobble=36", "kick=C", "kick=999"):
        try:
            parse_pairs(bad, 0, 127, "note")
        except ValueError:
            continue
        raise AssertionError(f"'{bad}' should have been rejected")


@test
def test_no_unguarded_windows_calls():
    """Windows-only APIs must sit behind a platform check.

    The GUI and the pipeline are expected to run on Linux and macOS too, where
    os.startfile does not exist and "explorer" is not a command. Checked statically
    because this machine cannot execute the other platforms' paths.
    """
    import re

    risky = {
        "os.startfile": r"os\.startfile",
        "explorer": r'"explorer"',
        "ctypes.windll": r"ctypes\.windll",
        "CREATE_NO_WINDOW": r"CREATE_NO_WINDOW",
    }
    guards = ("sys.platform", "os.name", "win32", 'platform.system')
    offenders = []
    for path in (ROOT / "drum2midi.py", ROOT / "drum2midi_gui.pyw",
                 ROOT / "render_midi.py", ROOT / "setup_env.py"):
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            for name, pattern in risky.items():
                if not re.search(pattern, line):
                    continue
                # a guard may be on this line or in the dozen lines above it
                window = "\n".join(lines[max(0, i - 12):i + 1])
                if not any(g in window for g in guards):
                    offenders.append(f"{path.name}:{i+1} {name}")
    assert not offenders, "unguarded Windows-only calls: " + "; ".join(offenders)


@test
def test_shell_launcher_is_usable():
    """drum2midi.sh must be LF-terminated, or bash reports 'bad interpreter'."""
    sh = ROOT / "drum2midi.sh"
    if not sh.exists():
        return
    raw = sh.read_bytes()
    assert raw.startswith(b"#!"), "missing shebang"
    assert b"\r\n" not in raw, "CRLF line endings would break bash"

    attrs = ROOT / ".gitattributes"
    assert attrs.exists(), ".gitattributes is needed to keep LF on checkout"
    assert "*.sh text eol=lf" in attrs.read_text(encoding="utf-8")


@test
def test_cli_help_is_english():
    import re
    res = subprocess.run([PY, str(ROOT / "drum2midi.py"), "--help"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(ROOT))
    assert res.returncode == 0, "--help failed"
    cyrillic = re.findall("[\u0410-\u044f\u0401\u0451]+", res.stdout)
    assert not cyrillic, f"non-English text in --help: {cyrillic[:5]}"


@test
def test_every_script_answers_help():
    """A script that starts a 20-hour job when asked what it does is a trap.

    Checked statically: running 39 scripts would take minutes, and any script that
    still ignores --help would do its real work instead of failing fast.
    """
    import ast
    offenders = []
    for path in sorted(ROOT.glob("*.py")):
        if path.name == "test_smoke.py":
            continue
        src = path.read_text(encoding="utf-8")
        if "argparse" in src or '"--help"' in src or "'--help'" in src:
            continue
        # a library whose __main__ block only explains itself is fine
        if "print(__doc__)" in src:
            continue
        tree = ast.parse(src)
        runs = any(not isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                      ast.AsyncFunctionDef, ast.ClassDef, ast.Assign,
                                      ast.AnnAssign, ast.Expr, ast.If, ast.Try))
                   for n in tree.body)
        if runs or "__main__" in src:
            offenders.append(path.name)
    assert not offenders, f"scripts that ignore --help: {offenders}"


@test
def test_privacy_gate_does_not_flag_the_repository_itself():
    """The gate must guard only names that identify somebody.

    It derives identity from the username, the git remote's owner, and the folder above
    the checkout. That last one is the one that keeps going wrong, in both directions.

    On GitHub Actions the layout is /home/<user>/work/<repo>/<repo>, so the parent is
    the repository, and every `from drum2midi import ...` was reported as a leak -- a
    gate that failed on every CI run the project ever had and passed on every machine a
    person looked at, so nobody looked.

    Then the opposite: the parent was taken as an identity whatever it was, so a
    checkout in C:/dev/drum2midi made `dev` a guarded name and bench_training.py's
    `for dev in devices:` was reported as a leaked account. A container folder is not a
    person, and guarding one flags ordinary code.
    """
    import check_privacy

    # Placeholder usernames rather than literal home paths: the gate rejects
    # /home/<a real name> on sight, and the shape of the path is all this test needs.
    ci = check_privacy.identity_names(
        "runner", Path("/home/<user>/work/drum2midi/drum2midi"),
        owner="someone", ci=True)
    assert "drum2midi" not in [n.lower() for n in ci], (
        f"the repository's own name is treated as an identity on CI: {ci}")
    assert "someone" in ci, (
        f"the account from the remote must be guarded on CI too: {ci}")
    # `runner` is the shared CI account and an ordinary English word; this repository
    # contains the phrase "the batch runner waits" and it is not a leak.
    assert "runner" not in ci, (
        f"the CI robot account is not an identity and must not be guarded: {ci}")

    # A home directory does name its owner, and that name is guarded.
    home = check_privacy.identity_names(
        "someone", Path("/home/<user>/drum2midi"), owner="someone")
    assert "<user>" in home and "someone" in home, (
        f"a checkout in a home directory must yield both names: {home}")

    # Anything else above the checkout is a container, not a person. `dev` is the real
    # case -- this project lives in C:/dev/drum2midi -- and `projects` is the one an
    # earlier version of this test wrongly required to be guarded.
    for layout in (Path("C:/dev/drum2midi"),
                   Path("/home/<user>/projects/drum2midi")):
        names = check_privacy.identity_names("someone", layout, owner="someone")
        container = layout.parent.name.lower()
        assert container not in [n.lower() for n in names], (
            f"the container folder {container!r} is guarded as an identity: {names}")
        assert "someone" in names, (
            f"the account from the remote must still be guarded: {names}")


@test
def test_gui_table_shows_every_drum_the_pipeline_prints():
    """A drum the pipeline reports but the table cannot colour is a silent gap.

    The results table looks each row up by name; a rename in drum2midi.py would make
    rows fall back to plain text without anything failing.
    """
    gui = load_gui()
    import drum2midi
    printed = {name.lower() for name in drum2midi.GM_NAMES.values()}
    missing = sorted(printed - set(gui.LABEL_TO_PITCH))
    assert not missing, f"results table cannot colour these rows: {missing}"


@test
def test_pipeline_finds_its_own_modules_without_the_script_directory():
    """drum2midi.py must import its siblings even when sys.path lacks its folder.

    Python usually puts a script's own directory on sys.path. The embedded runtime
    built by make_embedded.py does not: its ._pth file defines sys.path outright.
    Launching through runtime\\drum2midi.exe therefore died with
    "No module named 'devices'" at the point of conversion -- far enough in that
    --help still worked and nothing caught it.

    `python -P` reproduces exactly that condition.
    """
    code = (
        "import importlib.util, sys;"
        f"spec = importlib.util.spec_from_file_location('d2m', r'{ROOT / 'drum2midi.py'}');"
        "m = importlib.util.module_from_spec(spec);"
        "spec.loader.exec_module(m);"
        # the imports that were failing, deliberately done after the module has run
        "import devices, cpu_threads;"
        "print('ok')"
    )
    res = subprocess.run([sys.executable, "-P", "-c", code],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=tempfile.gettempdir())
    assert "ok" in (res.stdout or ""), (
        "drum2midi.py does not put its own directory on sys.path:\n"
        + (res.stderr or "")[-400:])


@test
def test_embedded_runtime_runs_pth_files():
    """The embedded runtime must call site.addsitedir, or editable installs vanish.

    A directory named in ._pth goes straight onto sys.path and the .pth files inside
    it are never executed. An editable install is nothing but a .pth file, so
    adtof_pytorch -- installed with `pip install -e` -- disappeared at runtime while
    being perfectly importable in the virtual environment.
    """
    src = (ROOT / "make_embedded.py").read_text(encoding="utf-8")
    assert "addsitedir" in src, \
        "make_embedded.py must generate a sitecustomize that calls site.addsitedir"

    runtime = ROOT / "runtime" / "sitecustomize.py"
    if runtime.exists():
        body = runtime.read_text(encoding="utf-8")
        assert "addsitedir" in body, \
            "runtime/sitecustomize.py is stale; re-run make_embedded.py"


@test
def test_midi_is_as_long_as_the_audio():
    """The clip must match the recording, not stop at the last drum hit.

    A song whose drums finish before the final chord produced a MIDI file shorter than
    the audio. Dropped into a DAW, that clip has no anchor at its right-hand edge, and
    aligning it by eye against the waveform is easy to get wrong -- which is how a
    correct transcription came to look as though it were missing its ending.
    """
    import mido
    import pretty_midi
    import soundfile as sf

    out, _ = run_pipeline("--no-separate")
    duration = sf.info(str(make_fixture())).duration
    length = mido.MidiFile(str(out)).length
    assert abs(length - duration) < 0.05, \
        f"MIDI is {length:.2f}s for {duration:.2f}s of audio"

    # the notes themselves must not be moved or invented by the padding
    notes = [n.start for i in pretty_midi.PrettyMIDI(str(out)).instruments for n in i.notes]
    assert notes, "no notes written"
    assert max(notes) <= duration + 0.01, "a note lands after the end of the audio"

    # and the other end: a track whose first hit arrives late must still start at zero,
    # or a DAW anchors the clip to the first note instead of to the recording
    mf = mido.MidiFile(str(out))
    for i, track in enumerate(mf.tracks):
        assert track[0].time == 0, f"track {i} starts {track[0].time} ticks in"
    with_notes = [t for t in mf.tracks if any(m.type == "note_on" for m in t)]
    assert with_notes, "no track carries notes"
    for track in with_notes:
        first_note = next(i for i, m in enumerate(track) if m.type == "note_on")
        assert any(m.time == 0 for m in track[:first_note]), \
            "the note track has nothing at tick 0 to anchor the clip"

    # and it must be possible to turn off, in which case the file ends at the last
    # note. The fixture is short, so compare against the notes rather than against a
    # fixed margin.
    bare, _ = run_pipeline("--no-separate", "--no-pad-to-audio")
    bare_len = mido.MidiFile(str(bare)).length
    bare_notes = [n.start for i in pretty_midi.PrettyMIDI(str(bare)).instruments
                  for n in i.notes]
    assert bare_len < duration - 0.05, \
        f"--no-pad-to-audio still ran to the end of the audio: {bare_len:.2f}s"
    assert abs(bare_len - max(bare_notes)) < 0.25, \
        f"--no-pad-to-audio ended at {bare_len:.2f}s, last note {max(bare_notes):.2f}s"


@test
def test_nothing_private_is_committed():
    """No machine paths, account names, private drafts or credentials in the repo.

    A draft letter to a third party was committed once, carrying our reasoning about
    how and when to approach that company. The history had to be rewritten. This makes
    the check automatic instead of a good intention, and it covers song titles too, by
    flagging absolute media paths and non-ASCII audio filenames.
    """
    res = subprocess.run([sys.executable, str(ROOT / "check_privacy.py"), "--all"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(ROOT))
    assert res.returncode == 0, (res.stdout or "")[-900:]


@test
def test_own_clone_url_is_the_only_account_exemption():
    """The account name is allowed in this repo's clone URL, and nowhere else.

    A public repository's URL contains its owner, so `git clone` instructions cannot use
    a placeholder without handing people a URL that fails. The exemption is therefore
    keyed to the full owner/repo slug taken from the origin remote.

    The case that matters is the third one. Coordination with a second machine runs
    through a separate private repository under the same account, which must never be
    named here -- so an exemption keyed to the account rather than the slug would have
    quietly permitted exactly the leak this whole check exists to prevent.
    """
    def verdict(text: str) -> int:
        probe = _tmp / "privacy_probe.md"
        probe.write_text(text, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(ROOT / "check_privacy.py"), str(probe)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            cwd=str(ROOT)).returncode

    url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True,
                         text=True, cwd=str(ROOT)).stdout.strip()
    m = re.search(r"github\.com[/:]([\w.-]+)/([\w.-]+?)(?:\.git)?$", url)
    if not m:
        return  # no GitHub remote configured; nothing to exempt
    owner, repo = m.group(1), m.group(2)

    assert verdict(f"git clone https://github.com/{owner}/{repo}.git") == 0, \
        "the repository's own clone URL should be allowed"
    assert verdict(f"the file landed in {owner}'s folder") != 0, \
        "a bare account name must still be rejected"
    assert verdict(f"git clone https://github.com/{owner}/{repo}-coord.git") != 0, \
        "a different repository under the same account must still be rejected"


def _doctor_packages():
    """The (module, why, needed) table that setup_env.check() prints.

    Read out of the source rather than by calling check(), which probes the whole
    machine. Two tests want it: one that the prose and the flag agree, one that it
    covers everything requirements.txt installs.
    """
    import ast
    import inspect
    import textwrap

    import setup_env
    tree = ast.parse(textwrap.dedent(inspect.getsource(setup_env.check)))
    rows = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Tuple) and len(node.elts) == 3:
            try:
                mod, why, needed = (ast.literal_eval(e) for e in node.elts)
            except ValueError:
                continue
            if isinstance(mod, str) and isinstance(needed, bool):
                rows.append((mod, why, needed))
    return rows


@test
def test_a_required_package_is_not_quietly_optional():
    """setup_env.check() must fail when something required is absent.

    Requiredness used to be inferred with `optional = why != "required"`, so an
    entry whose note elaborated -- "required for the default separator" -- compared
    unequal and turned optional. Measured at the time: with audio_separator
    unimportable, --check printed "everything required is in place" and exited 0.
    This reads the table back and insists the prose and the flag agree.
    """
    rows = _doctor_packages()
    assert rows, "could not read the package table out of setup_env.check()"
    for mod, why, needed in rows:
        assert why.startswith("required") == needed, (
            f"{mod} is described as {why!r} but needed={needed}")


@test
def test_setup_check_looks_for_every_requirement():
    """Whatever requirements.txt installs, `setup_env.py --check` must look for.

    The two lists drift apart silently, and in the direction that hurts: the doctor
    never mentioned Pillow at all, so an install that had dropped it reported a
    healthy environment, and the absence surfaced much later as `No module named
    'PIL'` from a smoke test that has nothing to do with installing anything.
    """
    # requirements.txt names distributions; --check imports modules.
    import_name = {"pillow": "PIL", "pyyaml": "yaml", "scikit-learn": "sklearn",
                   "audio-separator": "audio_separator"}
    # Installed but deliberately not checked, with the reason. Empty on purpose:
    # audioread was the only candidate and it was dropped from requirements.txt
    # outright rather than exempted here. An entry in this map is a decision; an
    # omission from the doctor's table is not, which is the distinction being kept.
    unchecked = {}

    checked = {mod.lower() for mod, _, _ in _doctor_packages()}
    missing = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        dist = line
        for sep in "<>=!~[;":
            dist = dist.split(sep)[0]
        dist = dist.strip().lower()
        if dist in unchecked:
            continue
        if import_name.get(dist, dist).lower() not in checked:
            missing.append(dist)
    assert not missing, (
        "requirements.txt installs these and setup_env.py --check never looks for "
        f"them: {missing}")


@test
def test_failed_install_does_not_end_with_a_clean_bill_of_health():
    """A half-finished install must not sign off as healthy.

    setup_env.py reports that requirements.txt did not install cleanly, then runs the
    verification. If nothing pip dropped happens to be on the required list -- which
    is what happens when the casualties are optional, as tqdm and mir_eval are --
    the last thing printed used to be "everything required is in place". The exit
    code said otherwise, but nobody reads exit codes off a screen.
    """
    import contextlib
    import io

    import setup_env

    # Both stubs say "nothing is missing", so problems stays 0 and the only thing
    # that can change the verdict is the install_failed flag under test. ffmpeg is
    # stubbed too because check() counts its absence as a problem, and whether the
    # machine running the tests has it is not what this is about.
    real_have, real_which = setup_env.have, shutil.which
    setup_env.have = lambda mod: True
    shutil.which = lambda prog, *a, **kw: f"/usr/bin/{prog}"
    try:
        for failed in (False, True):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = setup_env.check(install_failed=failed)
            out = buf.getvalue()
            signed_off = "everything required is in place" in out
            assert signed_off is not failed, (
                f"install_failed={failed} printed the clean bill of health: "
                f"{signed_off}")
            assert (rc == 0) is not failed, (
                f"install_failed={failed} returned {rc}")
    finally:
        setup_env.have = real_have
        shutil.which = real_which


@test
def test_install_advice_names_an_interpreter():
    """Every instruction to install torch must name which Python it installs into.

    pytorch.org gives the CUDA command as `pip3 install torch --index-url ...`, and
    this project repeated that shape. Followed literally on Windows it installs into
    whatever pip3 PATH resolves to, which is usually the system Python and not .venv.
    The install then succeeds, says nothing is wrong, and leaves a working CUDA build
    in an interpreter the pipeline never opens -- while torch here still reports +cpu
    and the GPU sits idle. It happened: 2.14.0+cu132 in one interpreter and
    2.14.0+cpu in .venv on the same machine, with a 4070 SUPER doing nothing.

    `<interpreter> -m pip` cannot go wrong that way, so the advice has to use it.
    """
    import devices

    # The command the program prints must point at the interpreter printing it.
    assert devices.pip_here().endswith(" -m pip"), (
        f"install advice does not go through `-m pip`: {devices.pip_here()}")
    named = Path(devices.interpreter_here())
    if not named.is_absolute():
        named = ROOT / named
    assert named.resolve() == Path(sys.executable).resolve(), (
        f"advice names {named}, but this is running in {sys.executable}")

    # The same for the install lines in requirements.txt, where a reader starts.
    offers = [ln.strip() for ln in
              (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
              if ln.lstrip("# ").startswith(("NVIDIA:", "CPU:", "Intel:"))]
    assert offers, "the torch install lines have moved or been renamed"
    bare = [ln for ln in offers if "python.exe" not in ln]
    assert not bare, f"these tell the reader to use whatever pip is on PATH: {bare}"


@test
def test_separator_declares_the_programs_it_runs():
    """A separator's dependencies include the programs it shells out to, not just imports.

    separator_missing exists so that a missing dependency is reported before ADTOF
    spends minutes transcribing. It checked imports only, so audio-separator's call to
    `ffmpeg -version` was invisible to it: a machine without ffmpeg transcribed a
    four-and-a-half minute recording in full and then failed with the child process's
    own last traceback line, "FileNotFoundError: [WinError 2] The system cannot find
    the file specified", naming neither ffmpeg nor what to do about it.

    ffmpeg is in no requirements file either, because nothing here imports it, so the
    installer could not have warned about it in advance. That is what makes it worth a
    test rather than a one-line fix.
    """
    import drum2midi

    assert "ffmpeg" in [p for p, _ in drum2midi.SEPARATOR_BINARIES.get("uvr", ())], (
        "the default separator runs ffmpeg and no longer declares it: "
        f"{drum2midi.SEPARATOR_BINARIES}")

    # The installer has to be able to say so before a conversion is attempted.
    doctor = (ROOT / "setup_env.py").read_text(encoding="utf-8")
    assert 'shutil.which("ffmpeg")' in doctor, (
        "setup_env.py --check does not look for ffmpeg")

    # And the advice has to name a real way to get it, per platform. It lives in
    # setup_env because the installer needs it too, and one copy cannot drift.
    import setup_env
    advice = setup_env.ffmpeg_advice()
    assert "ffmpeg" in advice.lower(), f"unhelpful ffmpeg advice: {advice}"
    assert not hasattr(drum2midi, "ffmpeg_advice"), (
        "two copies of the ffmpeg advice: setup_env owns it")

    # The precheck must actually fire on a binary, not only on a missing import.
    class _Args:
        separator = "uvr"
        no_separate = False

    real_which = shutil.which
    try:
        shutil.which = lambda prog, *a, **kw: None if prog == "ffmpeg" else "/x"
        found = drum2midi.separator_missing(_Args())
    finally:
        shutil.which = real_which
    assert found is not None, "a missing ffmpeg is not reported before work starts"
    what, why, how, _ = found
    assert "ffmpeg" in why.lower() and "PATH" in why, (
        f"the reason does not say what is wrong: {why}")


@test
def test_advertised_separator_flags_exist():
    """Any `--separator X` this project prints or documents must be a real choice.

    The message shown when ffmpeg is missing offered `--separator none` as the way
    out. There is no such choice -- argparse answers "invalid choice: 'none' (choose
    from 'larsnet', 'drumsep', 'uvr', 'hybrid')" -- so the remedy handed to someone
    who had just hit a wall was another wall. The flag for that is --no-separate.
    """
    import re

    import drum2midi

    parser = drum2midi.build_parser() if hasattr(drum2midi, "build_parser") else None
    if parser is None:
        src = (ROOT / "drum2midi.py").read_text(encoding="utf-8")
        m = re.search(r'"--separator",[^)]*?choices=\[([^\]]*)\]', src, re.S)
        assert m, "could not find the --separator choices"
        valid = set(re.findall(r'"([^"]+)"', m.group(1)))
    else:
        valid = set(parser._option_string_actions["--separator"].choices)

    bad = []
    for name in ("drum2midi.py", "README.md"):
        for i, line in enumerate((ROOT / name).read_text(encoding="utf-8").splitlines(), 1):
            for word in re.findall(r"--separator[ =]+([a-zA-Z_][\w-]*)", line):
                if word not in valid and word not in {"X", "SEPARATOR"}:
                    bad.append(f"{name}:{i} offers --separator {word}")
    assert not bad, ("these name a separator that does not exist, valid are "
                     f"{sorted(valid)}: {bad}")


@test
def test_a_check_that_could_not_run_is_not_a_pass():
    """A path the gate cannot read must not produce a clean bill.

    A missing file was folded into the "not a text type" count and the run exited 0, so
    asking about a file that had moved -- or passing one relative to the wrong directory,
    which is how this was found -- reported that nothing was wrong with a file nobody had
    opened. The gate's whole value is that green means looked-at.
    """
    import subprocess

    gone = _tmp / "definitely_not_here.md"
    res = subprocess.run([PY, str(ROOT / "check_privacy.py"), str(gone)],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(ROOT))
    out = res.stdout + res.stderr
    assert res.returncode != 0, f"a missing path exited 0:\n{out}"
    assert "NOT FOUND" in out or "could not be read" in out, (
        f"the report does not say the file was missing:\n{out}")

    present = _tmp / "present.md"
    present.write_text("nothing sensitive here\n", encoding="utf-8")
    ok = subprocess.run([PY, str(ROOT / "check_privacy.py"), str(present)],
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", cwd=str(ROOT))
    assert ok.returncode == 0, f"a clean file did not pass:\n{ok.stdout}{ok.stderr}"

@test
def test_the_gate_answers_about_itself_under_a_hook():
    """GIT_DIR must not decide which repository the gate is asking about.

    The gate asks git who owns this project and which clone URL may therefore appear in
    its text. It asked with cwd=ROOT, which is right until something sets GIT_DIR -- and
    git exports GIT_DIR to every hook it runs. A hook in the coordination repository made
    cwd lose, so the exemption was built from that repository's slug and this project's
    own clone URL, quoted in correspondence, was reported as a leaked account name.

    Clean by hand, one hit under the hook, same file and same content. A check that is
    wrong only when it runs automatically is worse than one that is wrong always, because
    the automatic run is the one nobody is watching.
    """
    import os
    import subprocess

    ours = _tmp / "quotes_our_own_url.md"
    url = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True,
                         text=True, cwd=str(ROOT)).stdout.strip()
    if not url:
        return  # no remote configured; nothing to exempt and nothing to test
    ours.write_text(f"the repository is at {url} and that is public\n", encoding="utf-8")

    def run(env):
        return subprocess.run([PY, str(ROOT / "check_privacy.py"), str(ours)],
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", cwd=str(ROOT), env=env)

    plain = run(None)
    assert plain.returncode == 0, (
        f"our own clone URL is flagged even by hand:\n{plain.stdout}{plain.stderr}")

    hooked = dict(os.environ)
    hooked["GIT_DIR"] = str(_tmp / "some_other_repo" / ".git")
    under_hook = run(hooked)
    assert under_hook.returncode == 0, (
        "the gate flags this project's own URL when GIT_DIR points elsewhere, which is "
        f"how every hook invokes it:\n{under_hook.stdout}{under_hook.stderr}")

@test
def test_the_summary_says_why_a_file_was_skipped_and_adds_up():
    """The gate's own accounting has to be checkable, which is what it asks of everyone else.

    A commit staging two text files reported "1 read for content, 1 not a text type". The
    second was check_privacy.py, which the scanner exempts by exact name because it defines
    the patterns it looks for -- a deliberate decision reported as an accident of file
    format. It was unreproducible for ten minutes because the summary said how many were
    skipped and never which.

    So: a skipped file is named, the self-exemption is its own category, and the numbers
    reconcile. Run over the whole tree rather than an explicit list, because naming a file
    on the command line forces it to be read and the skip path never runs -- the first
    version of this test made exactly that mistake and passed while guarding nothing.
    """
    import re
    import subprocess

    res = subprocess.run([PY, str(ROOT / "check_privacy.py"), "--all"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(ROOT))
    out = res.stdout + res.stderr

    line = next((l for l in out.splitlines() if l.startswith("checked ")), "")
    assert line, f"no summary line:\n{out[:400]}"
    assert "self-exempt" in line, (
        f"the gate's own exemption is not its own category:\n{line}")
    assert "skipped as binary:" in out, (
        "skipped files are counted but not named, which is what made the original "
        f"report impossible to investigate:\n{out[:400]}")

    total = int(re.search(r"checked (\d+) file", line).group(1))
    counted = [int(n) for n, _ in re.findall(
        r"(\d+) (read for content|not a text type|self-exempt|NOT FOUND|unaccounted)",
        line)]
    assert "unaccounted" not in line, f"the categories do not reconcile:\n{line}"
    assert sum(counted) == total, (
        f"{sum(counted)} accounted for against {total} checked:\n{line}")


@test
def test_a_file_that_cannot_be_opened_is_not_a_pass():
    """Present but unreadable is the same claim as absent, and was exiting 0.

    `scan()` ends its read in `except OSError: continue`, which increments nothing, and the
    missing, self-exempt and binary categories are all derived from the path list, so none
    of them can see a read that failed. A file held open by another process -- an editor, a
    sync client, a scanner, or a file removed between listing and reading, which is exactly
    what a pre-commit hook meets -- was reported as "1 unaccounted", then "no machine paths
    found", then exit 0.

    That is 300ca50 one branch over: a path that exists and cannot be opened is as unchecked
    as one that is not there. The remainder category is what reported it, which is why it
    was kept rather than deleted as unreachable -- it was not unreachable, the tests simply
    had no unreadable input.

    A directory with a text name is the portable way to be present and unopenable.
    """
    import subprocess

    unreadable = _tmp / "held_open.md"
    unreadable.mkdir()

    res = subprocess.run([PY, str(ROOT / "check_privacy.py"), str(unreadable)],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(ROOT))
    out = res.stdout + res.stderr
    assert res.returncode != 0, (
        f"a file that could not be opened reported a clean bill:\n{out}")
    assert "could not be read" in out or "UNREADABLE" in out, (
        f"the report does not say the file was unreadable:\n{out}")


@test
def test_unreadable_is_not_reported_as_binary():
    """The same file, one label further on, still exiting 0.

    `looks_textual` returns False for two unrelated reasons: it read the file and found NUL
    bytes, or it could not read the file at all. The binary category was derived from it
    alone, so an unopenable file whose name is not in TEXT was announced as "not a text
    type" -- a claim about content, about a file nothing had opened -- and skipped. Exit 0.

    Two things hid it. The category is only consulted when the file list is not explicit,
    and naming a file on the command line forces it to be read, so the guard for the
    previous fix could not reach this branch however it was mutated: it passed with the
    bug present. And the one fixture was called .md, which is in TEXT, so it fell through
    to UNREADABLE by luck of the extension. `.dat` and an extensionless file did not. The
    files that motivated this are hooks/pre-commit, which has no extension, and the .diff
    artefacts, which are not in TEXT -- both real contents of the coordination repository.

    So this runs --all inside a throwaway repository: tracked, unreadable, not explicit.
    """
    import os
    import shutil
    import subprocess

    repo = _tmp / "unreadable_repo"
    repo.mkdir()
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1",
           "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e"}

    def git(*a):
        return subprocess.run(["git", *a], cwd=str(repo), env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")

    git("init", "-q")
    names = ["looks_like_text.md", "looks_like_data.dat", "no_extension"]
    (repo / "ordinary.md").write_text("nothing to see\n", encoding="utf-8")
    for name in names:
        (repo / name).write_text("placeholder\n", encoding="utf-8")
    git("add", "-A")
    committed = git("commit", "-qm", "fixture")
    assert (repo / ".git").exists(), f"no repository to test in: {committed.stderr}"

    # Tracked in the index, a directory on disk: present, listed by ls-files, unopenable.
    for name in names:
        (repo / name).unlink()
        (repo / name).mkdir()

    listed = git("ls-files").stdout.split()
    for name in names:
        assert name in listed, (
            f"{name} is not tracked, so --all would never reach it: {listed}")

    shutil.copy2(ROOT / "check_privacy.py", repo / "check_privacy.py")
    res = subprocess.run([PY, str(repo / "check_privacy.py"), "--all"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(repo), env=env)
    out = res.stdout + res.stderr

    assert "not a text type" not in out, (
        "an unopened file is being announced as a known file format:\n" + out[:600])
    assert "UNREADABLE" in out, (
        f"the three unreadable files are not reported as unreadable:\n{out[:600]}")
    assert res.returncode != 0, (
        f"three files nothing could open still reported a clean bill:\n{out[:600]}")


@test
def test_the_unaccounted_remainder_is_fatal():
    """The remainder printed the truth and exited 0, which is how the other four survived.

    `return 1` was guarded by `missing or unreadable`; the remainder only appended a word to
    the summary. So a file in no category at all printed `1 unaccounted` and then "no machine
    paths, personal names, private drafts or credentials found", and exited 0. Every instance
    of this bug so far printed something true and exited 0, and each was fixed only once a
    person happened to read the line. Printing is not failing.

    This is the path that survives all four earlier fixes, because `can_read()` narrows the
    window between asking and reading and cannot close it: a file readable when asked and
    locked when `scan()` reaches it is not missing, not unreadable, not binary, not exempt
    and not read. It lands in the remainder, which is precisely the category that exited 0.

    The race is made deterministic by pinning `can_read` to True in a copy of the gate --
    standing in for the window, not asserting that can_read is wrong.

    A claim that stood here and was wrong: that `read == len(paths)` cannot coexist with a
    non-empty category, because scan() increments `read` only after every `continue`. That
    holds for what scan() did and not for what main() later concluded, because main()
    re-observed each file with fresh `can_read` and `looks_textual` calls. The two can
    disagree, and the arithmetic could go negative -- see the false-red guard below, which
    is the same window in the other direction. The categories are now subtracted from the
    set scan() actually opened, so neither direction is reachable.
    """
    import os
    import shutil
    import subprocess

    repo = _tmp / "unaccounted_repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e"}

    def git(*a):
        return subprocess.run(["git", *a], cwd=str(repo), env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")

    git("init", "-q")
    (repo / "ordinary.md").write_text("nothing to see\n", encoding="utf-8")
    (repo / "vanishes.md").write_text("placeholder\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "fixture")

    source = (ROOT / "check_privacy.py").read_text(encoding="utf-8")
    marker = "def can_read(path: Path) -> bool:"
    assert marker in source, "can_read has been renamed; this guard pins it by signature"
    pinned = source.replace(marker, marker + "\n    return True  # pinned for this test", 1)
    (repo / "check_privacy.py").write_text(pinned, encoding="utf-8")

    # readable when asked, unopenable when scan() gets there
    (repo / "vanishes.md").unlink()
    (repo / "vanishes.md").mkdir()

    res = subprocess.run([PY, str(repo / "check_privacy.py"), "--all"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(repo), env=env)
    out = res.stdout + res.stderr

    assert "unaccounted" in out, (
        f"the fixture did not produce a remainder, so this guards nothing:\n{out[:600]}")
    assert "no machine paths" not in out, (
        f"a clean bill was issued over a file in no category:\n{out[:600]}")
    assert res.returncode != 0, (
        f"the remainder printed and the run still exited 0:\n{out[:600]}")
    shutil.rmtree(repo, ignore_errors=True)


@test
def test_a_file_that_was_read_is_never_called_unreadable():
    """The other direction of the same window, and the one that gets a gate bypassed.

    Every defect in this area so far was silent-green: a file nothing opened, waved through.
    Deriving the categories from the whole path list is wrong both ways, because main()
    re-observes each file rather than asking scan() what happened to it. If a lock appears
    *after* the read -- or any predicate is simply wrong about a file already opened -- then
    a file that was read lands in `unreadable`, and the gate refuses a clean commit while
    printing `checked 1 file(s)` immediately above. Two contradictory sentences about one
    file, in one run.

    That is worse than a false pass in one specific way: a gate that fails a commit nobody
    can fix is a gate people learn to skip, and the next real leak goes with it.

    I argued in the commit for the remainder that this could not happen, on the grounds that
    scan() increments `read` only after every `continue`. That reasoning was about scan()
    and the bug is in main(). The second machine built the counter-example.

    Categories are now subtracted from the set scan() reports it opened, so a file that was
    read cannot be described by any of them.
    """
    import os
    import shutil
    import subprocess

    repo = _tmp / "false_red_repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e"}

    def git(*a):
        return subprocess.run(["git", *a], cwd=str(repo), env=env, capture_output=True,
                              text=True, encoding="utf-8", errors="replace")

    git("init", "-q")
    (repo / "ordinary.md").write_text("ordinary, readable, nothing private\n",
                                      encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "fixture")

    source = (ROOT / "check_privacy.py").read_text(encoding="utf-8")
    marker = "def can_read(path: Path) -> bool:"
    assert marker in source, "can_read has been renamed; this guard pins it by signature"
    pinned = source.replace(marker, marker + "\n    return False  # pinned for this test", 1)
    (repo / "check_privacy.py").write_text(pinned, encoding="utf-8")

    res = subprocess.run([PY, str(repo / "check_privacy.py"), "--all"],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=str(repo), env=env)
    out = res.stdout + res.stderr

    assert "could not be read" not in out, (
        f"a file the scanner opened is being reported as unreadable:\n{out[:600]}")
    assert "unaccounted" not in out, (
        f"a file the scanner opened is unaccounted for:\n{out[:600]}")
    assert res.returncode == 0, (
        f"a clean readable file was refused, which is how a gate gets bypassed:\n{out[:600]}")
    shutil.rmtree(repo, ignore_errors=True)


@test
def test_a_read_that_fails_mid_scan_cannot_be_quiet():
    """The real shape of the race, with nothing pinned.

    Both other guards in this area pin `can_read` to a constant, which is honest about being
    a stand-in but means neither of them exercises the predicates as they actually behave.
    This one leaves `can_read` and `looks_textual` alone and breaks the thing that really
    breaks: the read inside `scan()`, which is where a lock taken for the duration of a read
    lands. `scan()` swallows it in `except OSError: continue`, and every defect in this area
    began there.

    Two outcomes, and the second is why this is a test and not an assertion that everything
    is fatal:

      a textual file whose read failed   -> remainder, exit 1
      a genuinely binary file            -> "not a text type", exit 0

    The second is quiet and correct. `looks_textual` returns False for it on its own merits,
    so the label describes the file rather than guessing about it, and a binary file is
    skipped by design. The property worth having is not "every unopened file is fatal" -- it
    is that a category can no longer be *wrong and quiet at the same time*.

    This was traced by the second machine as an argument about the code, and flagged by them
    as reasoning rather than measurement, on the grounds that my last such argument was
    wrong. Both branches are measured here so the property is not resting on either of us
    reading the control flow correctly.
    """
    import os
    import shutil
    import subprocess

    def run(content: bytes, name: str) -> tuple[int, str]:
        repo = _tmp / f"midscan_{name.replace('.', '_')}"
        repo.mkdir()
        env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@e",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@e",
               "D2M_FAIL_READ": name}

        def git(*a):
            return subprocess.run(["git", *a], cwd=str(repo), env=env, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace")

        git("init", "-q")
        (repo / "ordinary.md").write_text("ordinary\n", encoding="utf-8")
        (repo / name).write_bytes(content)
        git("add", "-A")
        git("commit", "-qm", "fixture")

        source = (ROOT / "check_privacy.py").read_text(encoding="utf-8")
        read_line = ('            lines = path.read_text(encoding="utf-8", '
                     'errors="replace").splitlines()')
        assert read_line in source, "the read inside scan() has moved; this guard targets it"
        injected = ('            if path.name == os.environ.get("D2M_FAIL_READ", ""):\n'
                    '                raise OSError("simulated lock during scan")\n') + read_line
        (repo / "check_privacy.py").write_text(source.replace(read_line, injected, 1),
                                               encoding="utf-8")

        res = subprocess.run([PY, str(repo / "check_privacy.py"), "--all"],
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", cwd=str(repo), env=env)
        out = res.stdout + res.stderr
        shutil.rmtree(repo, ignore_errors=True)
        return res.returncode, out

    # Deliberately ordinary content: the file is never read, so nothing in it could be
    # detected. The run has to fail because the gate cannot account for the file, not
    # because of anything it contains.
    code, out = run(b"nothing unusual in here at all\n", "textual.md")
    assert "not accounted for by any category" in out, (
        f"the run did not fail on the remainder, so this guards something else:\n{out[:600]}")
    assert "no machine paths" not in out, (
        f"a file whose read failed was waved through with honest predicates:\n{out[:600]}")
    assert code != 0, (
        f"a textual file the scanner could not read exited 0:\n{out[:600]}")

    code, out = run(b"\x00\x01binary\x00\x02", "opaque.dat")
    assert "not a text type" in out, (
        f"a genuinely binary file is no longer described as binary:\n{out[:600]}")
    assert code == 0, (
        f"a binary file, correctly labelled, should not fail the run:\n{out[:600]}")


@test
def test_egmd_disk_check_uses_the_peak_not_the_end_state():
    """The requirement that cannot fail is the one measured after the risky part is over.

    fetch_egmd.py unpacks a 90 GiB archive into 131 GiB and deletes the archive afterwards,
    so both exist at once and the high-water mark is their sum. The docstring quoted the
    post-deletion figure -- "about 135 GB" -- as the requirement. A machine with 150 GiB
    free passes that number, spends four hours downloading, and dies partway through the
    unpack, which is the failure the deletion was added to prevent and does not.

    The check is separated from the disk query so this can be decided with a number rather
    than a filesystem, and asserted against the machine that found it: 147.8 GiB free must
    be refused for the audio archive and allowed for the MIDI one.
    """
    import fetch_egmd

    ok, note = fetch_egmd.enough_space(147.8, "audio")
    assert not ok, f"147.8 GiB was accepted for a run that peaks near 221:\n{note}"
    assert "221" in note or "220" in note, (
        f"the peak is not reported, so the refusal cannot be acted on:\n{note}")

    ok, _ = fetch_egmd.enough_space(147.8, "midi")
    assert ok, "the MIDI archive is 0.1 GiB and must not be blocked"

    ok, _ = fetch_egmd.enough_space(250.0, "audio")
    assert ok, "250 GiB is comfortably above the peak and was refused"

    # The archive and its contents coexist, so keeping the archive cannot lower the peak.
    peak_kept = fetch_egmd.enough_space(200.0, "audio", keep_archive=True)[0]
    peak_deleted = fetch_egmd.enough_space(200.0, "audio", keep_archive=False)[0]
    assert peak_kept == peak_deleted, (
        "--keep-archive changed the admission decision, but it only decides what is left "
        "afterwards; the moment that runs out of disk is identical")

    archive, unpacked = fetch_egmd.FOOTPRINT["audio"]
    assert archive + unpacked > 200, (
        "the recorded footprint no longer implies the peak this guard was written for")

    # The false-red the first version of this check had, which is the second machine's
    # actual checkout: archive and unpacked set both present, 97.4 GiB free. Nothing needs
    # to be downloaded or unpacked, so demanding the full peak refuses a run that would do
    # no work at all. A guard against a silent pass is not allowed to become a noisy
    # refusal; that is how a check gets taken out of the loop.
    ok, note = fetch_egmd.enough_space(97.4, "audio", have_archive_gib=89.8,
                                       unpacked_present=True)
    assert ok, (
        f"a checkout that already holds everything was refused:\n{note}")
    assert "nothing more is needed" in note, (
        f"the reason is not stated, so the pass cannot be checked:\n{note}")

    # Archive fully downloaded, unpack interrupted: only the unpacked size is still owed.
    ok, note = fetch_egmd.enough_space(140.0, "audio", have_archive_gib=89.8,
                                       unpacked_present=False)
    assert ok, f"140 GiB free with the archive already down was refused:\n{note}"
    ok, _ = fetch_egmd.enough_space(100.0, "audio", have_archive_gib=89.8,
                                    unpacked_present=False)
    assert not ok, "100 GiB cannot hold a 131 GiB unpack and was accepted"

    # A half-finished download owes only the remainder: 44.8 still to fetch plus the
    # 131 unpack is about 176, so 200 clears it and 150 does not.
    ok, note = fetch_egmd.enough_space(200.0, "audio", have_archive_gib=45.0)
    assert ok, f"200 GiB against 45 GiB already fetched plus the unpack was refused:\n{note}"
    ok, _ = fetch_egmd.enough_space(150.0, "audio", have_archive_gib=45.0)
    assert not ok, "150 GiB cannot hold the remaining 45 plus a 131 GiB unpack"


@test
def test_egmd_redundancy_is_reported_not_left_to_prose():
    """E-GMD's published counts are renderings, and the factor is 43.

    45,537 clips are 1,059 performances -- keyed by (drummer, session, id) -- each
    re-recorded on exactly 43 kits, so 444.5 hours rest on 10.3 hours of distinct playing and
    the 1,074,753 tom onsets we publish are 25,524 counted 43 times. docs/datasets.md turned
    that into "roughly 11,900x more tom data than MDB"; the honest multiplier is 284x.

    Both figures are true and answer different questions, which is exactly why the survey has
    to print both rather than leave the distinction to a sentence someone has to remember.
    This guards the reporting, on a metadata fixture with a known answer.
    """
    import csv
    import io
    import sys

    import fetch_egmd

    folder = _tmp / "egmd_meta"
    folder.mkdir()
    rows = []
    for perf in range(4):
        for kit in range(43):
            rows.append({"drummer": "d1", "session": "s1", "id": str(perf),
                         "duration": "10.0", "kit_name": f"kit{kit}",
                         "midi_filename": f"{perf}_{kit}.mid"})
    with (folder / "e-gmd-v1.0.0.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    held, sys.stdout = sys.stdout, io.StringIO()
    try:
        fetch_egmd.redundancy(folder)
        out = sys.stdout.getvalue()
    finally:
        sys.stdout = held

    assert "distinct performances" in out, f"the distinct count is not reported:\n{out}"
    assert "4" in out and "172" in out, (
        f"the fixture's 172 clips over 4 performances are not both shown:\n{out}")
    assert "43" in out, f"the redundancy factor is not reported:\n{out}"
    assert "renderings" in out, (
        f"the summary does not say the per-family counts are renderings:\n{out}")


@test
def test_larsnet_weights_check_looks_inside_the_folder():
    """Upstream ships the weights folder empty, so its existence proved nothing.

    `install_larsnet` tested `pretrained_larsnet_models.exists()` and returned early with
    "weights already present". The LarsNet repository ships that directory containing a
    `.gitkeep` and five empty per-stem subfolders, so a `git clone` creates it before any
    weight is downloaded and the documented install -- `setup_env.py --with-larsnet` --
    printed success while fetching none of the 563 MB. `--verify` then printed a tick using
    the same test, so the check that exists to catch a broken install confirmed it.

    The obvious repair is wrong in the other direction and this guard covers both. The
    `.pth` files live in per-stem subdirectories with none at the top level, so a
    non-recursive `glob("*.pth")` never sees an installed set and re-downloads 563 MB every
    run -- a false failure rather than a false pass, which is cheaper and still incorrect.
    """
    import setup_env

    fresh = _tmp / "larsnet_fresh"
    (fresh / "larsnet" / "pretrained_larsnet_models").mkdir(parents=True)
    weights = fresh / "larsnet" / "pretrained_larsnet_models"
    (weights / ".gitkeep").write_text("", encoding="utf-8")
    for stem in ("kick", "snare", "toms", "hihat", "cymbals"):
        (weights / stem).mkdir()
    assert not setup_env.larsnet_weights_present(fresh), (
        "a clone with the folder and five empty stem subdirectories, which is what upstream "
        "ships, is being reported as an installed set")

    # and the real layout, which is nested -- not top-level, which is what a naive fix assumes
    for stem in ("kick", "snare", "toms", "hihat", "cymbals"):
        (weights / stem / f"pretrained_{stem}_unet.pth").write_bytes(b"\x00")
    assert setup_env.larsnet_weights_present(fresh), (
        "weights in per-stem subdirectories are not being found, so a correct install would "
        "be re-downloaded on every run")

    # a partial extract is not an install
    (weights / "kick" / "pretrained_kick_unet.pth").unlink()
    assert not setup_env.larsnet_weights_present(fresh), (
        "four stems of five is being accepted as a complete set")


@test
def test_restem_mode_guard_settles_before_it_refuses():
    """The guard that decides whether an overnight ReStem batch may start.

    Two measured failures, needing opposite treatment, which is why the wait is asymmetric:

    * while a render runs the mode selector leaves the accessibility tree entirely, and a
      read taken then returns nothing. Treated as "unreadable", that stops the batch. It has
      never fired only because the loop happens to check between tracks -- safety by
      accident.
    * the label lags a human moving the selector by a few seconds, so a read taken just
      after a change returns the *previous* mode. That produced one false refusal.

    Two agreeing reads do not fix the second: a stale label read twice agrees with itself.
    So a read matching what was asked for is acted on at once -- a lagging label cannot
    fabricate agreement it has not seen yet -- and anything else is given the whole window
    to become the expected value before it is reported as a disagreement. The cost of
    waiting falls on the refusal, where a wrong answer loses a night.

    Until now the only evidence any of that was tested was a sentence in the function's own
    comment; there was no test in the repository. `-DefineOnly` loads the functions without
    running the batch, so a stubbed Current-Mode can drive it with no ReStem present.
    """
    import subprocess

    harness = _tmp / "mode_harness.ps1"
    harness.write_text(r"""
. "%s" -DefineOnly
$script:calls = 0
$script:case  = ""
function Current-Mode {
    $script:calls++
    switch ($script:case) {
        "match"     { return "Best (Offline) +" }
        "lagging"   { if ($script:calls -le 2) { return "Better" } else { return "Best (Offline) +" } }
        "disagree"  { return "Better" }
        "transient" { if ($script:calls -le 2) { return $null } else { return "Best (Offline) +" } }
        "absent"    { return $null }
    }
}
foreach ($c in @("match","lagging","disagree","transient","absent")) {
    $script:case = $c
    $script:calls = 0
    $t0 = Get-Date
    $r = Settled-Mode -Want "Best (Offline) +" -Seconds 9
    $el = ((Get-Date) - $t0).TotalSeconds
    Write-Output ("CASE={0}|RESULT={1}|SECS={2}" -f $c, $r, [math]::Round($el,1))
}
""" % str(ROOT / "restem_batch_mode.ps1").replace("\\", "\\"), encoding="utf-8")

    res = subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                          "-File", str(harness)],
                         capture_output=True, text=True, encoding="utf-8",
                         errors="replace", timeout=300)
    out = res.stdout + res.stderr
    got = {}
    for line in out.splitlines():
        if line.startswith("CASE="):
            parts = dict(kv.split("=", 1) for kv in line.strip().split("|"))
            got[parts["CASE"]] = (parts["RESULT"], float(parts["SECS"]))
    assert len(got) == 5, f"harness did not report all five cases:\n{out[:800]}"

    want = "Best (Offline) +"

    # the happy path pays nothing: a matching read is acted on at once
    assert got["match"][0] == want, f"a matching read was not returned: {got['match']}"
    assert got["match"][1] < 3, (
        f"the happy path waited {got['match'][1]}s; the asymmetry is the point")

    # THE bug: a stale label read twice agrees with itself
    assert got["lagging"][0] == want, (
        f"a lagging label was reported as a disagreement: {got['lagging']} -- this refuses "
        "a batch that should have run")

    # a genuine disagreement is still a disagreement, after the whole window
    assert got["disagree"][0] == "Better", (
        f"a real mode mismatch was not reported: {got['disagree']}")
    assert got["disagree"][1] >= 6, (
        f"the refusal was decided in {got['disagree'][1]}s without waiting out the lag")

    # the selector leaving the tree mid-render is polled through, not treated as failure
    assert got["transient"][0] == want, (
        f"a transient absence stopped the batch: {got['transient']}")

    # but a selector that never appears must not be reported as agreement
    assert got["absent"][0] == "", (
        f"an unreadable selector returned a mode: {got['absent']}")

def main() -> int:
    global _tmp
    _tmp = Path(tempfile.mkdtemp(prefix="drum2midi_test_"))
    print(f"running smoke tests in {_tmp}\n")
    try:
        for name, obj in sorted(globals().items()):
            if name.startswith("test_") and getattr(obj, "is_test", False):
                obj()
    finally:
        shutil.rmtree(_tmp, ignore_errors=True)

    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
    if FAILED:
        print("\nfailures:")
        for name, why in FAILED:
            print(f"  {name}: {why.splitlines()[0] if why else ''}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
