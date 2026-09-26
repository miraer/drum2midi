#!/usr/bin/env bash
# Starts the drum2midi window on Linux and macOS.
#
# The Windows launcher has to work around taskbar icon handling; here the desktop
# takes its icon from the .desktop entry (Linux) or the app bundle (macOS), so this
# only needs to find the interpreter and get out of the way.
#
# The checks are functions so the smoke tests can source this file and call them;
# sourcing runs nothing else.
set -euo pipefail

# ldconfig lives in /sbin, which a normal user's PATH lacks on Debian -- the distro
# where libxcb-cursor0 is most often missing -- so it is looked for there too.
find_ldconfig() {
    command -v ldconfig 2>/dev/null && return 0
    [ -x /sbin/ldconfig ] && echo /sbin/ldconfig && return 0
    return 1
}

# Whether the dynamic linker knows a library: 0 yes, 1 no, 2 cannot tell.
#
# The listing is read whole and matched in the shell, with no pipe. The first version
# piped `ldconfig -p` into `grep -q`, which exits at the first match; ldconfig, still
# writing a listing larger than the pipe buffer, then died of SIGPIPE (141), and with
# pipefail that read as "not found" -- blocking a working install. Measured 50 times
# out of 50 with a long listing and an early match.
lib_known() {
    local ldconfig listing
    ldconfig="$(find_ldconfig)" || return 2
    listing="$("$ldconfig" -p 2>/dev/null)" || return 2
    case "$listing" in
        *"$1"*) return 0 ;;
    esac
    return 1
}

main() {
    local here python_bin status
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

    if ! "$python_bin" -c "import PySide6.QtWidgets" >/dev/null 2>&1; then
        echo "PySide6 is missing, so the window cannot open." >&2
        echo "  $python_bin -m pip install PySide6" >&2
        echo "Command line still works:  $python_bin drum2midi.py drums.wav -o out.mid" >&2
        exit 1
    fi

    # Qt 6.5 and later need libxcb-cursor to open a window under X11, and most
    # distributions do not install it. Without it Qt stops with "Could not load the Qt
    # platform plugin xcb", which names neither the library nor the package. A Wayland
    # session uses Qt's Wayland plugin instead, so only X11 is checked. When the
    # listing cannot be read at all the check steps aside: a missed warning costs a
    # Qt error message, a false one blocks a window that would have opened.
    if [ "$(uname -s)" = "Linux" ] && [ -z "${WAYLAND_DISPLAY:-}" ]; then
        status=0
        lib_known "libxcb-cursor.so.0" || status=$?
        if [ "$status" -eq 1 ]; then
            echo "libxcb-cursor is missing, which Qt needs to open a window under X11." >&2
            echo "  Debian/Ubuntu:  sudo apt install libxcb-cursor0" >&2
            echo "  Fedora:         sudo dnf install xcb-util-cursor" >&2
            echo "  Arch:           sudo pacman -S xcb-util-cursor" >&2
            echo "Command line still works:  $python_bin drum2midi.py drums.wav -o out.mid" >&2
            exit 1
        fi
    fi

    exec "$python_bin" "$here/drum2midi_gui.pyw" "$@"
}

# run when executed, not when sourced by the tests
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
