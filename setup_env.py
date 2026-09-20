"""One-shot setup: installs dependencies, fetches models, verifies the result.

Replaces the wall of copy-paste commands in the README. Everything is optional except
the core pipeline, so a minimal install stays small.

    python setup_env.py                 # core pipeline (MDX23C separator)
    python setup_env.py --with-larsnet  # also the fast LarsNet separator
    python setup_env.py --with-render   # also FluidSynth + soundfont for A/B audio
    python setup_env.py --check         # verify an existing install, change nothing
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
PY = sys.executable

LARSNET_WEIGHTS = "1U8-5924B1ii1cjv9p0MTPzayb00P4qoL"
FLUIDSYNTH_URL = ("https://github.com/FluidSynth/fluidsynth/releases/download/"
                  "v2.6.0/fluidsynth-v2.6.0-win10-x64-cpp11.zip")
SOUNDFONT_URL = ("https://ftp.osuosl.org/pub/musescore/soundfont/"
                 "MuseScore_General/MuseScore_General.sf3")

OK, WARN, BAD = "  ok   ", "  warn ", "  MISS "


def interpreter() -> str:
    """The interpreter that actually holds what was just installed or verified.

    Everything here installs into `sys.executable` and checks the same one, so advice
    that says a bare `python` is advice about a different program. On Windows a bare
    `python` is usually the Store alias rather than the project's virtualenv, which is
    how a passing --check turns into ModuleNotFoundError on the very next line.
    """
    exe = Path(sys.executable)
    try:
        return str(Path(".") / exe.relative_to(ROOT))
    except ValueError:
        return str(exe)


def path_python() -> str | None:
    """What a bare `python` would run, if that is not the interpreter in use."""
    found = shutil.which("python")
    if not found:
        return "nothing on PATH"
    try:
        if Path(found).resolve() == Path(sys.executable).resolve():
            return None
    except OSError:
        pass
    return found


def warn_if_bare_python_differs() -> None:
    other = path_python()
    if other is None:
        return
    print(f"\n{WARN}a bare `python` here runs {other},")
    print("       which is not the interpreter the checks above describe. Use the path")
    print("       shown in the commands, or activate the environment first.")


def run(*args, check=True) -> int:
    print(f"    $ {' '.join(str(a) for a in args[:6])}"
          + (" ..." if len(args) > 6 else ""))
    res = subprocess.run(list(args), cwd=str(ROOT))
    if check and res.returncode != 0:
        print(f"    failed with exit code {res.returncode}")
    return res.returncode


def pip(*args) -> int:
    return run(PY, "-m", "pip", "install", *args)


def have(module: str) -> bool:
    return subprocess.run([PY, "-c", f"import {module}"],
                          capture_output=True, cwd=str(ROOT)).returncode == 0


def ffmpeg_advice() -> str:
    """How to install ffmpeg on this platform. From PR #1 by mykolad."""
    if sys.platform == "darwin":
        return "brew install ffmpeg"
    if os.name != "nt":
        return "sudo apt install ffmpeg   (or your distribution's equivalent)"
    return ("winget install Gyan.FFmpeg   (then open a new terminal, because PATH is "
            "only read at launch)")


def install_core() -> bool:
    """Installs the core packages; False if the requirements install did not finish.

    The return value matters because pip resolves the whole file and can fail on
    one package having already installed others, which leaves a half-built
    environment. That exit code used to be printed by run() and then ignored, so
    the script carried on to [2/3], ended with the verification banner, and said
    everything required was in place. Someone ended up without Pillow that way and
    nothing reported it until a test errored out on `import PIL` much later.
    """
    print("\n[1/3] Python packages")
    ok = pip("-r", str(ROOT / "requirements.txt")) == 0
    if not ok:
        print()
        print(f"{BAD}requirements.txt did not install cleanly")
        print("       Some packages may still have gone in, so the checks at the end")
        print("       will pass for those and the environment is incomplete anyway --")
        print("       read pip's error above rather than the summary below.")

    print("\n[2/3] ADTOF-pytorch (not on PyPI)")
    target = ROOT / "ADTOF-pytorch"
    if target.exists():
        print("    already cloned")
    else:
        run("git", "clone", "--depth", "1",
            "https://github.com/xavriley/ADTOF-pytorch.git", str(target))
    if target.exists():
        pip("-e", str(target))

    print("\n[3/3] MDX23C drum separator")
    print("    downloads automatically on first conversion (~400 MB)")

    return ok


def install_larsnet() -> None:
    print("\n[extra] LarsNet")
    target = ROOT / "larsnet"
    if not target.exists():
        run("git", "clone", "--depth", "1",
            "https://github.com/polimi-ispl/larsnet.git", str(target))
    if not target.exists():
        print("    clone failed, skipping weights")
        return
    if (target / "pretrained_larsnet_models").exists():
        print("    weights already present")
        return
    pip("gdown", "torchaudio")
    zip_path = target / "weights.zip"
    if run(PY, "-m", "gdown", LARSNET_WEIGHTS, "-O", str(zip_path), check=False) == 0:
        try:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(target)
            zip_path.unlink()
            print("    weights extracted")
        except Exception as exc:
            print(f"    could not extract: {exc}")


def install_render() -> None:
    print("\n[extra] FluidSynth + soundfont (for render_midi.py)")
    tools = ROOT / "tools"
    tools.mkdir(exist_ok=True)

    if shutil.which("fluidsynth"):
        print(f"    FluidSynth already on PATH: {shutil.which('fluidsynth')}")
    elif list(tools.glob("fluidsynth/**/fluidsynth*")):
        print("    FluidSynth already present in tools/")
    elif os.name == "nt":
        # Windows has no package manager by default, so a build is fetched; elsewhere
        # FluidSynth belongs to the system package manager and bundling it would be rude
        zip_path = tools / "fs.zip"
        try:
            print("    downloading FluidSynth ...")
            urllib.request.urlretrieve(FLUIDSYNTH_URL, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(tools / "fluidsynth")
            zip_path.unlink()
        except Exception as exc:
            print(f"    failed: {exc}")
    else:
        hint = ("brew install fluid-synth" if sys.platform == "darwin"
                else "sudo apt install fluidsynth   (or dnf/pacman equivalent)")
        print(f"    FluidSynth is not bundled on this platform. Install it with:\n"
              f"        {hint}")

    sf = tools / "MuseScore_General.sf3"
    if sf.exists():
        print("    soundfont already present")
    else:
        try:
            print("    downloading soundfont (38 MB) ...")
            urllib.request.urlretrieve(SOUNDFONT_URL, sf)
        except Exception as exc:
            print(f"    failed: {exc}")


def _intel_gpu_present() -> bool:
    """Is there an Intel GPU that the XPU build of torch could drive?

    Checked through the OS rather than torch, because the whole point is to notice a
    GPU that the installed torch cannot see.
    """
    if os.name != "nt":
        return False
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False
    return any(k in out for k in ("Arc", "Iris", "Intel(R) UHD", "Intel(R) HD"))


def install_xpu() -> None:
    """Swaps in the Intel GPU build of torch. Measured 12x faster for separation."""
    print("\n[extra] PyTorch with Intel GPU (XPU) support")
    runtime = ["intel-cmplr-lib-rt==2026.1.0", "intel-cmplr-lib-ur==2026.1.0",
               "intel-cmplr-lic-rt==2026.1.0", "intel-sycl-rt==2026.1.0",
               "onemkl-license==2026.1.0", "onemkl-sycl-blas==2026.1.0",
               "onemkl-sycl-dft==2026.1.0", "onemkl-sycl-lapack==2026.1.0",
               "onemkl-sycl-rng==2026.1.0", "onemkl-sycl-sparse==2026.1.0",
               "dpcpp-cpp-rt==2026.1.0", "intel-opencl-rt==2026.1.0", "mkl==2026.1.0",
               "intel-openmp==2026.1.0", "tcmlib==1.5.0", "umf==1.1.0",
               "intel-pti==1.0.1"]
    # The Intel runtime lives on PyPI, but the torch build does not; installing them
    # in one command makes pip resolve both against the same index and fail.
    pip(*runtime)
    pip("--force-reinstall", "--no-deps", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/xpu")
    print(f"  installed; verify with:  {interpreter()} devices.py")


def check(install_failed: bool = False) -> int:
    """Reports what is present. `install_failed` is whether pip actually finished.

    Without that, this printed "everything required is in place" a few lines under
    the one saying requirements.txt had not installed cleanly, because nothing on
    the required list happened to be what pip dropped. The exit code was right and
    the last thing on screen still said the opposite.
    """
    print("\nverifying installation\n")
    problems = 0

    print("  Python packages")
    # numpy and scipy are listed here rather than left to arrive with something else:
    # 67 scripts in this repository import numpy directly, so it is a dependency in its
    # own right. Relying on librosa to drag it in meant --check could report that
    # everything required was in place and then test_smoke.py would fail on `import
    # numpy` one line later, which is exactly what happened to someone.
    #
    # Requiredness is a field of its own and is deliberately not inferred from the
    # explanation. It used to be `optional = why != "required"`, so any entry whose
    # note elaborated -- "required for the default separator" -- compared unequal and
    # went quietly optional. Measured: with audio_separator unimportable, --check
    # printed "everything required is in place" and exited 0, on a machine where the
    # default separator could not run at all. onnxruntime is listed for the related
    # reason that audio_separator imports it lazily, so checking the wrapper alone
    # says nothing about whether separation can actually start.
    for mod, why, needed in (("torch", "required", True),
                             ("numpy", "required", True),
                             ("scipy", "required", True),
                             ("librosa", "required", True),
                             ("pretty_midi", "required", True),
                             ("mido", "required", True),
                             ("soundfile", "required", True),
                             ("audio_separator", "required for the default separator", True),
                             ("onnxruntime", "required for the default separator", True),
                             ("mir_eval", "benchmarks only", False),
                             ("sklearn", "learned velocity models only", False),
                             ("PIL", "GUI icons and make_logo.py only", False),
                             ("tqdm", "benchmark and training scripts only", False),
                             ("yaml", "LarsNet only", False)):
        ok = have(mod)
        if ok:
            print(f"{OK}{mod}")
        else:
            print(f"{WARN if not needed else BAD}{mod}  ({why})")
            problems += 1 if needed else 0

    # ffmpeg is a program rather than a package, so it cannot be a line in
    # requirements.txt and does not go through have(). It is required all the same:
    # audio_separator runs `ffmpeg -version` at startup and dies without it. Measured
    # on a two-second clip with only ffmpeg's directory removed from PATH --
    #
    #   with ffmpeg:     wav ok, mp3 ok
    #   without ffmpeg:  both fail, "UVR separator failed: FileNotFoundError [WinError 2]"
    #
    # so it is not merely a decoder for compressed input; plain wav fails too. An
    # earlier version of this file called it "demo video tooling only" and let it
    # warn, which was wrong on both counts.
    if shutil.which("ffmpeg"):
        print(f"{OK}ffmpeg")
    else:
        print(f"{BAD}ffmpeg  (required for the default separator)")
        print(f"       {ffmpeg_advice()}")
        problems += 1

    print("\n  Models")
    if have("adtof_pytorch"):
        print(f"{OK}ADTOF-pytorch")
        # The two-machine protocol in docs/second-machine.md rests on comparing these
        # two strings, so they have to be printed rather than merely assumed equal.
        # An earlier version claimed to print them and did not, which made the
        # safeguard decorative.
        try:
            import hashlib

            from adtof_pytorch import get_default_weights_path
            weights = Path(get_default_weights_path())
            raw = weights.read_bytes()
            print(f"    checkpoint   {weights.name}")
            print(f"    size         {len(raw)} B")
            print(f"    sha256       {hashlib.sha256(raw).hexdigest()}")
        except Exception as exc:
            print(f"{WARN}could not resolve the ADTOF checkpoint: {exc}")
    else:
        print(f"{BAD}ADTOF-pytorch  (pip install -e ./ADTOF-pytorch)")
        problems += 1

    mdx = ROOT / "uvr" / "models" / "MDX23C-DrumSep-aufr33-jarredou.ckpt"
    print(f"{OK}MDX23C separator" if mdx.exists()
          else f"{WARN}MDX23C separator  (downloads on first run)")

    lars = ROOT / "larsnet" / "pretrained_larsnet_models"
    print(f"{OK}LarsNet weights" if lars.exists()
          else f"{WARN}LarsNet weights  (only needed for --separator larsnet)")

    # ffmpeg is checked with the packages above rather than here: it is required
    # for the default separator, and these two genuinely are optional.
    print("\n  Optional tools")
    fs = (shutil.which("fluidsynth")
          or (list((ROOT / "tools").glob("fluidsynth/**/fluidsynth*"))
              if (ROOT / "tools").exists() else []))
    print(f"{OK}FluidSynth" if fs else f"{WARN}FluidSynth  (render_midi.py only)")
    sf = ROOT / "tools" / "MuseScore_General.sf3"
    print(f"{OK}soundfont" if sf.exists() else f"{WARN}soundfont  (render_midi.py only)")

    print("\n  Hardware")
    try:
        sys.path.insert(0, str(ROOT))
        import devices as _dev
        found = _dev.available_devices()
        accel = [d for d in found if d != "cpu"]
        if accel:
            for d in accel:
                print(f"{OK}{d}: {_dev.describe_device(d)}")
            print(f"       {_dev.summary('auto')}")
        else:
            import torch
            from devices import unused_gpu
            idle = unused_gpu()
            if idle:
                card, how = idle
                print(f"{WARN}{card} is installed and not being used")
                for line in how.splitlines():
                    print(f"       {line.strip()}")
            else:
                print(f"{WARN}no GPU in use, running on the CPU")
    except Exception as exc:
        print(f"{WARN}could not query devices ({exc})")

    print("\n  Benchmark datasets (not needed to convert audio)")
    for path, name in ((ROOT / "mdbdrums", "MDB Drums"),
                       (ROOT / "gmd", "Groove MIDI"),
                       (ROOT / "idmt", "IDMT-SMT-Drums")):
        print(f"{OK}{name}" if path.exists() else f"{WARN}{name}  (see README)")

    if problems:
        print(f"\n{problems} required component(s) missing - run without --check to install")
        return 1
    if install_failed:
        print()
        print("nothing on the required list is missing, but the install above did")
        print("not finish, so this is not a clean bill of health -- read the error.")
        warn_if_bare_python_differs()
        return 1
    print("\neverything required is in place; try:")
    print(f"    {interpreter()} test_smoke.py")
    warn_if_bare_python_differs()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only, install nothing")
    ap.add_argument("--with-larsnet", action="store_true",
                    help="also install the fast LarsNet separator and its weights")
    ap.add_argument("--with-render", action="store_true",
                    help="also install FluidSynth and a General MIDI soundfont")
    ap.add_argument("--with-intel-gpu", action="store_true",
                    help="replace torch with the Intel GPU (XPU) build; separation "
                         "measured 12x faster")
    args = ap.parse_args()

    if args.check:
        return check()

    if shutil.which("git") is None:
        print("git is required and was not found on PATH")
        return 1

    core_ok = install_core()
    if args.with_larsnet:
        install_larsnet()
    if args.with_render:
        install_render()
    if args.with_intel_gpu:
        install_xpu()
    elif _intel_gpu_present():
        print("\nAn Intel GPU was detected. Separation runs about 12x faster on it:")
        print(f"    {interpreter()} setup_env.py --with-intel-gpu")
    return check(install_failed=not core_ok)


if __name__ == "__main__":
    raise SystemExit(main())
