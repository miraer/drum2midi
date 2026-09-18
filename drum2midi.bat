@echo off
rem Starts the drum2midi window without a console.
rem
rem runtime\drum2midi.exe is an embeddable Python carrying our icon. It exists because
rem a Microsoft Store interpreter lives under WindowsApps, and Windows binds the taskbar
rem button to that process regardless of the icons the application sets. The embedded
rem build is an ordinary file we can brand, and it shares the virtual environment's
rem packages, so nothing is installed twice. Build it with: python make_embedded.py
if exist "%~dp0runtime\drum2midi.exe" (
  start "" "%~dp0runtime\drum2midi.exe" "%~dp0drum2midi_gui.pyw"
) else if exist "%~dp0.venv\Scripts\drum2midi.exe" (
  start "" "%~dp0.venv\Scripts\drum2midi.exe" "%~dp0drum2midi_gui.pyw"
) else (
  start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0drum2midi_gui.pyw"
)
