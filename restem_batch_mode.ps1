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

function Close-StaleDialog {
    # A batch that dies mid-track leaves the file dialog open, and the dialog's own
    # 160 elements then hide the main window's mode text -- so every later track fails
    # with "Load unavailable" and the guard reports the mode as unreadable. Closing it
    # first makes a restart resume instead of failing 59 times in a row.
    try {
        $w = Get-RestemWindow
        if (-not $w) { return }
        $wins = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
          (New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ControlTypeProperty,
            [System.Windows.Automation.ControlType]::Window)))
        foreach ($d in $wins) {
            try {
                $d.GetCurrentPattern([System.Windows.Automation.WindowPattern]::Pattern).Close()
                Say ("closed a stale dialog: " + $d.Current.Name)
                Start-Sleep -Seconds 2
            } catch { }
        }
    } catch { }
}

function Current-Mode {
    # Retried, because FindAll throws ElementNotAvailableException when the window is
    # mid-redraw between renders. An overnight run died on the second track that way:
    # one transient failure of the guard looked exactly like the guard refusing, and
    # 59 recordings did not happen. A guard that cannot survive a repaint is not a
    # guard, it is another way to lose a night.
    for ($try = 1; $try -le 5; $try++) {
        try {
            $w = Get-RestemWindow
            if (-not $w) { Start-Sleep -Seconds 3; continue }
            $all = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                              [System.Windows.Automation.Condition]::TrueCondition)
            $name = ($all | Where-Object { $_.Current.Name -match '\((Offline|Realtime)\)' } |
                       Select-Object -First 1).Current.Name
            if ($name) { return $name }
        } catch {
            # ElementNotAvailableException and friends: the window moved under us
        }
        Start-Sleep -Seconds 3
    }
    return $null
}

function Settled-Mode {
    # Current-Mode returns the first value it can read. That is not enough to act on,
    # for two reasons measured on this machine, and they need different treatment.
    #
    # While a render runs the selector leaves the accessibility tree altogether: a probe
    # caught it reading nothing at the moment a render began, with the neighbouring
    # controls going disabled at the same time. The in-loop guard read that as "the mode
    # is unreadable" and stopped the batch. It has never fired in practice only because
    # the loop happens to check between tracks rather than during one, which is safety by
    # accident. Polling through the gap fixes that one.
    #
    # The label also lags a human changing the selector by a few seconds, so a read taken
    # just after a change returns the previous mode. That produced one false refusal. Two
    # agreeing reads do NOT fix it -- a stale label read twice agrees with itself, which
    # is what the first version of this function got wrong, and a test caught it before a
    # batch did.
    #
    # So the wait is asymmetric, which is the point: a read matching what was asked for is
    # acted on at once, because a lagging label cannot fabricate agreement it has not seen
    # yet. Anything else is given the whole window to become the expected value before it
    # is reported as a disagreement. The cost of waiting falls on the refusal, where a
    # wrong answer loses a night, rather than on the happy path.
    param([string]$Want, [int]$Seconds = 45)

    $deadline = (Get-Date).AddSeconds($Seconds)
    $last = $null
    $announced = $false
    while ($true) {
        $now = Current-Mode
        if ($now) {
            if (-not $Want -or $now -eq $Want) { return $now }
            if (-not $announced -or $now -ne $last) {
                Say "mode reads '$now', waiting for '$Want' in case the label is behind"
                $announced = $true
            }
            $last = $now
        }
        if ((Get-Date) -ge $deadline) { break }
        Start-Sleep -Seconds 3
    }
    if ($last) { Say "mode stayed '$last' for ${Seconds}s" }
    return $last
}

$mode = Settled-Mode -Want $Expect
if (-not $mode) { Close-StaleDialog; $mode = Settled-Mode -Want $Expect }
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
    $now = Settled-Mode -Want $Expect
    if (-not $now) { Close-StaleDialog; $now = Settled-Mode -Want $Expect }
    if (-not $now) { Say "mode unreadable after retries - stopping"; break }
    if ($now -ne $Expect) { Say "mode changed to '$now' - stopping"; break }

    $before = [datetime]::MinValue
    if (Test-Path $cacheJson) { $before = (Get-Item $cacheJson).LastWriteTime }

    $ok = $false
    $t0 = Get-Date
    try {
        $win = Get-RestemWindow
        $load = Find-ByName $win "Load"
        if (-not $load) { Say "Load unavailable for $track"; $failed += $track; continue }
        Invoke-Btn $load
        Start-Sleep -Seconds 2
        if (-not (Submit-FileDialog $src)) { Say "could not submit $track"; $failed += $track; continue }
        Wait-DialogGone 20 | Out-Null

        $t0 = Get-Date
        $deadline = $t0.AddMinutes($TimeoutMinutes)
        while ((Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 10
            if ((Test-Path $cacheJson) -and (Get-Item $cacheJson).LastWriteTime -gt $before) {
                Start-Sleep -Seconds 4
                Copy-Item $cacheJson $dest -Force
                $ok = $true
                break
            }
            try {
                $w = Get-RestemWindow
                if (-not $w) { Say "window disappeared during $track"; break }
                $err = Find-ByName $w "OK"
                if ($err) { try { Invoke-Btn $err } catch {}; Say "render error on $track"; break }
            } catch {
                # a repaint during the poll is not a failure; the cache check above is
                # what decides whether the render finished
            }
        }
    } catch {
        # One unavailable UI element used to end the batch. Losing a track is a cost
        # worth paying to keep the other 59.
        Say ("{0} threw: {1}" -f $track, $_.Exception.Message)
        $failed += $track
        continue
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
