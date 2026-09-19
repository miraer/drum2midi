# Runs several tracks through ReStem in whatever mode the window currently shows.
#
# Written for the overnight Best + Bleed Reduction measurement. Two tracks already
# showed that mode losing -- 0.466 against 0.500 on a track with no toms, 0.915 against
# 0.931 on the tom-heaviest -- but two tracks are a sample, and Bleed Reduction can only
# affect a tom score where toms exist. Six of MDB's seven tom-bearing tracks remain.
#
# It refuses to start unless the window reports the mode it was told to expect, because
# an overnight run that silently records the wrong label is worse than no run.
#
#   powershell -File restem_batch_mode.ps1 -Expect "Best (Offline) +" -Label best-plus
#   powershell -File restem_batch_mode.ps1 -Expect "Best (Offline) +" -Label best-plus -Tracks a,b

param(
    [Parameter(Mandatory = $true)][string] $Expect,
    [Parameter(Mandatory = $true)][string] $Label,
    [string[]] $Tracks,
    [int] $TimeoutMinutes = 90
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Invoke-Expression (Get-Content "$root\restem_ui.ps1" -Raw)

# MDB's tom-bearing tracks, minus Beatles which is already measured. These carry 60 of
# the corpus's 90 tom onsets; the other 16 tracks have none, so no tom score can move.
if (-not $Tracks) {
    $Tracks = @(
        "MusicDelta_FreeJazz_Drum",
        "MusicDelta_FusionJazz_Drum",
        "MusicDelta_Grunge_Drum",
        "MusicDelta_Punk_Drum",
        "MusicDelta_FunkJazz_Drum",
        "MusicDelta_LatinJazz_Drum"
    )
}

$cacheJson = Join-Path $env:APPDATA "ReStem 2\cache\stems\trigger_events.json"
$outDir = Join-Path $root "bench\restem_modes"
$log = Join-Path $root "bench\restem_batch_mode.log"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Say([string] $m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    Write-Output $line
    Add-Content -Path $log -Value $line
}

function Current-Mode {
    $w = Get-RestemWindow
    if (-not $w) { return $null }
    $all = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                      [System.Windows.Automation.Condition]::TrueCondition)
    ($all | Where-Object { $_.Current.Name -match '\((Offline|Realtime)\)' } |
       Select-Object -First 1).Current.Name
}

$mode = Current-Mode
if (-not $mode) { Say "ReStem is not running or its window is unreadable"; exit 1 }
if ($mode -ne $Expect) {
    Say "mode is '$mode' but '$Expect' was expected - refusing to run"
    exit 1
}
Say "=== batch start: mode '$mode', label '$Label', $($Tracks.Count) tracks"

$done = 0
$failed = @()
foreach ($track in $Tracks) {
    $src = Join-Path $root "restem_in\$track.wav"
    $dest = Join-Path $outDir "$track.$Label.json"
    if (-not (Test-Path $src)) { Say "missing input: $track"; $failed += $track; continue }
    if (Test-Path $dest) { Say "already have $track"; $done++; continue }

    # re-check every time: a crash and restart could reset the selector
    $now = Current-Mode
    if ($now -ne $Expect) { Say "mode changed to '$now' - stopping"; break }

    $before = [datetime]::MinValue
    if (Test-Path $cacheJson) { $before = (Get-Item $cacheJson).LastWriteTime }

    $win = Get-RestemWindow
    $load = Find-ByName $win "Load"
    if (-not $load) { Say "Load unavailable for $track"; $failed += $track; continue }
    Invoke-Btn $load
    Start-Sleep -Seconds 2
    if (-not (Submit-FileDialog $src)) { Say "could not submit $track"; $failed += $track; continue }
    Wait-DialogGone 20 | Out-Null

    $t0 = Get-Date
    $deadline = $t0.AddMinutes($TimeoutMinutes)
    $ok = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 10
        if ((Test-Path $cacheJson) -and (Get-Item $cacheJson).LastWriteTime -gt $before) {
            Start-Sleep -Seconds 4
            Copy-Item $cacheJson $dest -Force
            $ok = $true
            break
        }
        $w = Get-RestemWindow
        if (-not $w) { Say "window disappeared during $track"; break }
        $err = Find-ByName $w "OK"
        if ($err) { try { Invoke-Btn $err } catch {}; Say "render error on $track"; break }
    }

    if ($ok) {
        $done++
        Say ("{0} done in {1:N1} min ({2}/{3})" -f $track, ((Get-Date) - $t0).TotalMinutes, $done, $Tracks.Count)
    } else {
        $failed += $track
        Say ("{0} timed out after {1:N0} min" -f $track, ((Get-Date) - $t0).TotalMinutes)
    }
}

Say "=== batch end: $done of $($Tracks.Count) collected"
if ($failed.Count) { Say "failed: $($failed -join ', ')" }
