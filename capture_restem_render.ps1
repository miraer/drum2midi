<#
    Captures both of ReStem's outputs from one render, so they can be compared.

    The question is whether ReStem's MIDI exporter writes notes its own
    `trigger_events.json` does not describe. Evidence already on disk says yes -- their
    export carries 1086 notes against the JSON's 1083, and the three extras are pitch 60,
    the Other stem -- but the two files came from renders 74 minutes apart, so they may
    simply have had different stems enabled. A conclusion drawn from two different runs
    is the error this project keeps retracting.

    Run this, then render once in ReStem with every stem's MIDI OUT enabled, then press
    a key. It snapshots both files with their timestamps so the pairing is provable.

        powershell -File capture_restem_render.ps1 -Track MusicDelta_Beatles_Drum
#>
param(
    [Parameter(Mandatory = $true)][string] $Track,
    [string] $OutDir = "bench/restem_pair"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$json = Join-Path $env:APPDATA "ReStem 2\cache\stems\trigger_events.json"
$drag = Join-Path $env:LOCALAPPDATA "Temp\ReStem 2\midi-drag"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function State {
    $j = if (Test-Path $json) { (Get-Item $json).LastWriteTime } else { $null }
    # Recurse: ReStem drops the file into a per-drag subdirectory, not into midi-drag
    # itself. A non-recursive filter finds nothing and reports it as "the drag failed".
    $m = if (Test-Path $drag) { @(Get-ChildItem $drag -Recurse -Filter *.mid -File) } else { @() }
    [pscustomobject]@{ Json = $j; Midi = $m }
}

$before = State
Write-Host "before the render:"
Write-Host "  trigger_events.json : $(if ($before.Json) { $before.Json } else { 'absent' })"
Write-Host "  midi-drag           : $($before.Midi.Count) file(s)"
Write-Host ""
Write-Host "Now, in ReStem:"
Write-Host "  1. load $Track"
Write-Host "  2. enable MIDI OUT on EVERY stem, Other included"
Write-Host "  3. render once"
Write-Host "  4. drag the MIDI out, which is what populates midi-drag"
Write-Host ""
Read-Host "press Enter when the render has finished and the MIDI has been dragged"

$after = State
if (-not $after.Json) { throw "no trigger_events.json found at $json" }
if ($before.Json -and $after.Json -le $before.Json) {
    Write-Warning "trigger_events.json did not change -- did the render run?"
}
if (-not $after.Midi) { throw "no .mid in $drag -- the MIDI has to be dragged out" }

$newest = $after.Midi | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$gap = [math]::Abs(($newest.LastWriteTime - $after.Json).TotalSeconds)

Copy-Item $json (Join-Path $OutDir "$Track.trigger_events.json") -Force
Copy-Item $newest.FullName (Join-Path $OutDir "$Track.restem_export.mid") -Force

Write-Host ""
Write-Host "captured into $OutDir"
Write-Host ("  json written {0}" -f $after.Json.ToString('HH:mm:ss'))
Write-Host ("  midi written {0}  ({1})" -f $newest.LastWriteTime.ToString('HH:mm:ss'), $newest.Name)
Write-Host ("  {0:N1} s apart" -f $gap)
if ($gap -gt 180) {
    Write-Warning "more than three minutes apart -- these may not be the same render, which is exactly the thing this script exists to rule out"
}
