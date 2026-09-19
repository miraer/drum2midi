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
    [Parameter(Mandatory = $true)][string] $Label,
    [int] $StallMinutes = 5,
    [int] $MaxMinutes = 120
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Invoke-Expression (Get-Content "$root\restem_ui.ps1" -Raw)

# How far through a render ReStem says it is, 0..1, or $null if it is not rendering.
# Kept here rather than in restem_ui.ps1 so a running batch that has already sourced
# that file is unaffected.
function Get-RenderProgress($win) {
    if (-not $win) { return $null }
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
        [System.Windows.Automation.ControlType]::ProgressBar)
    $el = $win.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $cond)
    if (-not $el) { return $null }
    foreach ($pat in @([System.Windows.Automation.RangeValuePattern]::Pattern,
                       [System.Windows.Automation.ValuePattern]::Pattern)) {
        try { return [double] $el.GetCurrentPattern($pat).Current.Value } catch { }
    }
    $null
}

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
# ReStem publishes a real progress value (0..1) on a ProgressBar element, so the wait
# does not have to guess how long a render takes. A fixed deadline is wrong in both
# directions: 400 s was generous for a 13 s track and expired 13 minutes early on a
# 125 s one, and when it expired the caller copied the *previous* track's stems, which
# still sit in the cache under the same names. Waiting on a stall instead fails only
# when nothing is happening, and reports where it stopped rather than how long it waited.
$stallAt = Get-Date
$last = -1.0
$deadline = (Get-Date).AddMinutes($MaxMinutes)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    if ((Test-Path $cacheJson) -and (Get-Item $cacheJson).LastWriteTime -gt $before) {
        Start-Sleep -Seconds 3      # let it finish writing
        Copy-Item $cacheJson $dest -Force
        "wrote $dest"
        break
    }

    $w = Get-RestemWindow
    $p = Get-RenderProgress $w
    if ($null -ne $p -and $p -gt $last + 1e-6) {
        $last = $p
        $stallAt = Get-Date
    }
    $idle = ((Get-Date) - $stallAt).TotalMinutes
    if ($idle -ge $StallMinutes) {
        if ($last -ge 0) {
            "render stalled at {0:P0} after {1:N1} min with no progress" -f $last, $idle
        } else {
            "no progress reported for {0:N1} min; the render may never have started" -f $idle
        }
        break
    }

    # a render error puts up a dialog; dismiss it and stop rather than waiting out
    # the full timeout
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
