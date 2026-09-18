#!/usr/bin/env bash
# Starts the drum2midi window on Linux and macOS.
#
# The Windows launcher has to work around taskbar icon handling; here the desktop
# takes its icon from the .desktop entry (Linux) or the app bundle (macOS), so this
# only needs to find the interpreter and get out of the way.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -x "$here/.venv/bin/python" ]; then
    python_bin="$here/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="$(command -v python3)"
    echo "note: no virtual environment at $here/.venv, using $python_bin" >&2
    echo "      create one with:  python3 -m venv .venv && .venv/bin/python setup_env.py" >&2
else
    echo "No Python found. Install Python 3.10 or newer." >&2
    exit 1
fi

if ! "$python_bin" -c "import tkinter" >/dev/null 2>&1; then
    echo "tkinter is missing, so the window cannot open." >&2
    echo "  Debian/Ubuntu:  sudo apt install python3-tk" >&2
    echo "  Fedora:         sudo dnf install python3-tkinter" >&2
    echo "  macOS (brew):   brew install python-tk" >&2
    echo "Command line still works:  $python_bin drum2midi.py drums.wav -o out.mid" >&2
    exit 1
fi

exec "$python_bin" "$here/drum2midi_gui.pyw" "$@"
