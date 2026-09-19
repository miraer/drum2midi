# Re-runs one track through ReStem in the mode currently selected, and reports whether
# the result matches what we already have on file.
#
# The MDB comparison was made on 17 September without recording which offline quality
# mode was active. The exported JSON does not carry it, and the only mode marker on disk
# belongs to a later run. So the question is settled empirically: transcribe the same
# track again in a known mode and compare event for event. A match means the stored
# result was produced in this mode; a mismatch rules it out.
#
# Read the mode off the window first -- this script does not change it. Set it by hand,
# run this, note the answer, change it, run again.
#
#   powershell -File restem_one.ps1 -Track MusicDelta_Rock_Drum -Label best-plus

param(
    [string] $Track = "MusicDelta_Rock_Drum",
    [Parameter(Mandatory = $true)][string] $Label
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Invoke-Expression (Get-Content "$root\restem_ui.ps1" -Raw)

$src = Join-Path $root "restem_in\$Track.wav"
if (-not (Test-Path $src)) { "no such input: $src"; exit 1 }

$cacheJson = Join-Path $env:APPDATA "ReStem 2\cache\stems\trigger_events.json"
$outDir = Join-Path $root "bench\restem_modes"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$dest = Join-Path $outDir "$Track.$Label.json"

$win = Get-RestemWindow
if (-not $win) { "ReStem 2 is not running"; exit 1 }

# Report the mode showing in the window, so the label cannot silently disagree with it
$all = $win.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                    [System.Windows.Automation.Condition]::TrueCondition)
$mode = ($all | Where-Object { $_.Current.Name -match '(?i)\((Offline|Realtime)\)' } |
         Select-Object -First 1).Current.Name
"mode shown in the window : $mode"
"label being written      : $Label"
if (-not $mode) { "could not read the mode; aborting so the result is not mislabelled"; exit 1 }

$before = [datetime]::MinValue
if (Test-Path $cacheJson) { $before = (Get-Item $cacheJson).LastWriteTime }

$load = Find-ByName $win "Load"
if (-not $load) { "Load button unavailable"; exit 1 }
Invoke-Btn $load
Start-Sleep -Seconds 2
if (-not (Submit-FileDialog $src)) { "could not submit the path"; exit 1 }
# The batch runner waits for the dialog to close before timing the render. Without
# this the wait starts while the file chooser is still up, and times out against a
# render that never began.
Wait-DialogGone 20 | Out-Null

"waiting for the render ..."
$deadline = (Get-Date).AddSeconds(400)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    if ((Test-Path $cacheJson) -and (Get-Item $cacheJson).LastWriteTime -gt $before) {
        Start-Sleep -Seconds 3      # let it finish writing
        Copy-Item $cacheJson $dest -Force
        "wrote $dest"
        break
    }
    # a render error puts up a dialog; dismiss it and stop rather than waiting out
    # the full timeout
    $w = Get-RestemWindow
    $err = Find-ByName $w "OK"
    if ($err) {
        try { Invoke-Btn $err } catch {}
        "render error reported by ReStem"
        break
    }
}
if (-not (Test-Path $dest)) { "no trigger_events.json appeared"; exit 1 }

# Compare against the stored run from 17 September
$stored = Join-Path $root "restem_events\$Track.json"
if (Test-Path $stored) {
    & "$root\.venv\Scripts\python.exe" "$root\compare_restem_runs.py" $stored $dest
} else {
    "no stored run at $stored to compare against"
}
