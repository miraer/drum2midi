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
    """
    import importlib.util
    from importlib.machinery import SourceFileLoader
    path = ROOT / "drum2midi_gui.pyw"
    spec = importlib.util.spec_from_loader("gui", SourceFileLoader("gui", str(path)))
    gui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gui)
    return gui


@test
def test_gui_builds():
    """The GUI must construct and assemble a valid command line."""
    try:
        import tkinter as tk
        root = tk.Tk()
    except ImportError:
        return          # tkinter not installed (some Linux builds)
    except Exception:
        return          # no display; tk raises TclError, but be liberal here
    root.withdraw()
    gui = load_gui()
    app = gui.App(root)
    app.v_input.set(str(make_fixture()))
    app.v_output.set(str(_tmp / "gui.mid"))
    app.v_sep.set("none")
    app._sync()
    cmd = app._command()
    assert "--no-separate" in cmd, f"separator flag missing: {cmd}"
    app.adv_rows[38][1].set("5")
    assert "--channels" in " ".join(app._command()), "channel override not passed through"
    root.destroy()


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
def test_theme_palette_matches_the_logo():
    """The window and the logo must not drift apart into two different blues.

    theme.py restates the logo's colours as hex strings, because make_logo works in
    RGB tuples and the GUI needs "#rrggbb". Restating them means they can disagree.
    """
    import make_logo
    import theme
    for name, rgb in (("base", make_logo.PAPER), ("ink", make_logo.INK),
                      ("accent", make_logo.WAVE)):
        want = "#%02x%02x%02x" % rgb
        got = getattr(theme.Light, name)
        assert got == want, f"theme.Light.{name} is {got}, logo uses {want}"

    # and the dark theme has to stay legible: every instrument colour must lift away
    # from the background rather than sink into it
    import drum_icons
    for pitch in (36, 38, 42, 49, 51):
        lifted = theme.readable(drum_icons.colour(pitch), theme.Dark)
        lum = sum(int(lifted[i:i + 2], 16) for i in (1, 3, 5)) / 3
        assert lum > 90, f"{pitch} stays dark on the dark theme: {lifted}"


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
