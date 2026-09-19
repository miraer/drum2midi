# One half of the Better-versus-Best test: render the loaded track, capture what came out.
#
# The published claim is that Better and Best produce byte-identical MIDI. Two different
# separation models doing that is extraordinary, and the load-bearing part - that the
# selector was applied at all - is unproven, because the log recording the window's mode
# was overwritten and ReStem's guide describes a separate Reprocess action. A cache hit
# would advance the cache file's mtime, satisfy the old guard, and look exactly like this.
#
# Identical MIDI from two models could be coincidence. Identical STEMS could not, so this
# captures the seven stem wavs as well as the events JSON, and hashes them.
#
# The mode cannot be set by script - the menu is drawn, not in the accessibility tree - so
# it is read and checked rather than set, and the run refuses if it is not what was
# expected. Run it once per mode with a human changing the mode between.
#
#   powershell -File restem_mode_evidence.ps1 -Expect "Better (Offline)" -Label better
#   powershell -File restem_mode_evidence.ps1 -Expect "Best (Offline)"   -Label best
#
# Records the render duration, because a Best render that returns instantly is a cache hit
# and answers the question on its own.

param(
    [Parameter(Mandatory = $true)][string] $Expect,
    [Parameter(Mandatory = $true)][string] $Label,
    [string] $Track = "MusicDelta_FunkJazz_Drum",
    [int] $StallMinutes = 6,
    [int] $MaxMinutes = 90
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $root "restem_ui.ps1")

$cache = Join-Path $env:APPDATA "ReStem 2\cache\stems"
$json = Join-Path $cache "trigger_events.json"
$outDir = Join-Path $root "bench\mode_evidence\$Label"
$wav = Join-Path $root "restem_in\$Track.wav"

function Say($t) { $line = "[{0:HH:mm:ss}] {1}" -f (Get-Date), $t; Write-Host $line }

if (-not (Test-Path $wav)) { Say "no such track: $wav"; exit 1 }
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$win = Get-RestemWindow
if (-not $win) { Say "ReStem is not running"; exit 1 }

# --- the mode must be what the caller says, read from the window rather than assumed
$all = $win.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                    [System.Windows.Automation.Condition]::TrueCondition)
$mode = $null
foreach ($e in $all) {
    $n = $e.Current.Name
    if ($n -match '^(Good|Better|Best) \(') { $mode = $n; break }
}
Say "mode shown in the window : '$mode'"
if ($mode -ne $Expect) {
    Say "expected '$Expect' - refusing to run"
    Say "set it by hand; the menu is drawn and cannot be set by script"
    exit 1
}

# --- baseline, so 'new' means new rather than merely present
$before = @{}
foreach ($f in (Get-ChildItem $cache -Filter *.wav -EA SilentlyContinue)) {
    $before[$f.Name] = (Get-FileHash $f.FullName -Algorithm SHA256).Hash
}
$jsonBefore = if (Test-Path $json) { (Get-Item $json).LastWriteTime } else { [datetime]::MinValue }
Say ("baselined {0} cached stems" -f $before.Count)

# --- load
$load = Find-ByName $win "Load"
if (-not $load) { Say "Load button not found"; exit 1 }
Invoke-Btn $load
Start-Sleep -Seconds 2
if (-not (Submit-FileDialog $wav)) { Say "could not submit the path"; exit 1 }
Wait-DialogGone 20 | Out-Null
Say "loaded $Track"

# --- wait on progress, not on a guessed clock
$t0 = Get-Date
$deadline = $t0.AddMinutes($MaxMinutes)
$last = -1.0
$lastMove = Get-Date
$ok = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    if ((Test-Path $json) -and (Get-Item $json).LastWriteTime -gt $jsonBefore) {
        Start-Sleep -Seconds 4
        $ok = $true
        break
    }
    $w = Get-RestemWindow
    $p = $null
    if ($w) {
        foreach ($e in $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                                  [System.Windows.Automation.Condition]::TrueCondition)) {
            if ($e.Current.LocalizedControlType -eq "progress bar") {
                try {
                    $rv = $e.GetCurrentPattern([System.Windows.Automation.RangeValuePattern]::Pattern)
                    $p = [double]$rv.Current.Value
                } catch {}
                break
            }
        }
    }
    if ($null -ne $p -and [Math]::Abs($p - $last) -gt 0.001) {
        $last = $p
        $lastMove = Get-Date
    }
    if (((Get-Date) - $lastMove).TotalMinutes -ge $StallMinutes) {
        Say ("render stalled at {0:P0} after {1:N1} min" -f [Math]::Max($last, 0), $StallMinutes)
        exit 1
    }
}
$elapsed = ((Get-Date) - $t0).TotalSeconds
if (-not $ok) { Say ("gave up after {0:N1} min" -f ($elapsed / 60)); exit 1 }
Say ("render finished in {0:N1}s" -f $elapsed)
if ($elapsed -lt 12) {
    Say "THAT IS TOO FAST TO BE A RENDER. Very likely a cached result was re-emitted,"
    Say "which is exactly the failure this test exists to detect."
}

# --- capture, and check the stems belong to this track rather than the previous one
$dur = $null
try {
    Add-Type -AssemblyName PresentationCore
    $mp = New-Object System.Windows.Media.MediaPlayer
    $mp.Open([uri]$wav)
    Start-Sleep -Milliseconds 700
    if ($mp.NaturalDuration.HasTimeSpan) { $dur = $mp.NaturalDuration.TimeSpan.TotalSeconds }
    $mp.Close()
} catch {}

$manifest = @()
$changed = 0
foreach ($f in (Get-ChildItem $cache -Filter *.wav)) {
    $h = (Get-FileHash $f.FullName -Algorithm SHA256).Hash
    $was = $before[$f.Name]
    if ($was -ne $h) { $changed++ }
    Copy-Item $f.FullName (Join-Path $outDir $f.Name) -Force
    $manifest += [pscustomobject]@{ file = $f.Name; sha256 = $h; bytes = $f.Length
                                    written = (Get-Item $f.FullName).LastWriteTime.ToString("s")
                                    changed = ($was -ne $h) }
}
# The events file gets the same treatment as the stems rather than an assumption. An
# earlier version recorded changed = $true unconditionally, so the only evidence that it
# had been rewritten was the wait loop's exit condition -- which is sound, but leaves the
# manifest asserting something it did not check. Its write time is recorded too, because
# "identical" and "never rewritten" look the same in a hash.
$jsonHash = (Get-FileHash $json -Algorithm SHA256).Hash
$jsonWritten = (Get-Item $json).LastWriteTime
$manifest += [pscustomobject]@{ file = "trigger_events.json"; sha256 = $jsonHash
                                bytes = (Get-Item $json).Length
                                written = $jsonWritten.ToString("s")
                                changed = ($jsonWritten -gt $jsonBefore) }
Copy-Item $json (Join-Path $outDir "trigger_events.json") -Force

$meta = [pscustomobject]@{
    label = $Label; mode = $mode; track = $Track
    input_sha256 = (Get-FileHash $wav -Algorithm SHA256).Hash
    input_seconds = $dur
    render_seconds = [Math]::Round($elapsed, 1)
    stems_changed = $changed
    captured = (Get-Date).ToString("s")
    files = $manifest
}
$meta | ConvertTo-Json -Depth 4 | Out-File (Join-Path $outDir "manifest.json") -Encoding UTF8

Say ("{0} of {1} stems differ from the previous render" -f $changed, $before.Count)
if ($changed -eq 0 -and $before.Count -gt 0) {
    Say "NO STEM CHANGED. Either the mode had no effect or nothing was recomputed."
}
Say "captured to $outDir"
