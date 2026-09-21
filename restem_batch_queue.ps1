# Drives ReStem's own Batch Process dialog instead of loading one file at a time.
#
# Why it matters: the per-file path costs about 150 seconds of fixed overhead per
# render. On MDB, averaging 57 seconds a track, that is tolerable. On ENST, averaging
# 30, it is most of the cost -- measured at 6-11x realtime against 2.4x on MDB, which
# turns 210 recordings into eleven hours.
#
# The dialog was missed twice. A UI Automation scan reported no batch control, because
# the queue list is drawn rather than built from controls, exactly like the quality
# menu. Then a probe submitted three paths, saw nothing render, and concluded the
# product takes one file at a time -- when in fact the dialog had appeared, listed all
# three as "waiting", and was waiting for Start to be pressed. Both conclusions were
# wrong in the same direction: absence of evidence read as evidence of absence.
#
# The buttons ARE real: Start, Close and Change... all carry Invoke. Only the list is
# painted. And the dialog writes each result to an output folder, so attribution needs
# no cache-watching at all.
#
# One trap, learned expensively: `Close` is not a unique name. The window's own title
# bar has one, and invoking the first match by name closed ReStem outright. Every
# button here is looked up by name AND checked against the set the batch dialog is
# known to carry.
#
#   powershell -File restem_batch_queue.ps1 -Tracks a,b,c -Out bench\restem_batch_out

param(
    [Parameter(Mandatory = $true)][string[]] $Tracks,
    [string] $Expect = "Best (Offline) +",
    [int] $TimeoutMinutes = 240,
    [string] $Out = "bench\restem_batch_out"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Invoke-Expression (Get-Content "$root\restem_ui.ps1" -Raw)

$log = Join-Path $root "bench\restem_batch_queue.log"
$outDir = Join-Path $root $Out
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Say([string] $m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    Write-Output $line
    Add-Content -Path $log -Value $line
}

function Get-Elements {
    $w = Get-RestemWindow
    if (-not $w) { return $null }
    $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
               [System.Windows.Automation.Condition]::TrueCondition)
}

function Find-DialogButton([string] $name) {
    # Scoped by geometry as well as by name, because neither alone is enough. `Close`
    # exists twice: once on the dialog and once in the window chrome, in the same case,
    # and invoking the chrome one shuts ReStem down -- which it did, twice, before this
    # was written. The dialog's buttons sit well below the title bar; the chrome's do
    # not. Anything in the top strip of the window is refused.
    $all = Get-Elements
    if (-not $all) { return $null }
    $w = Get-RestemWindow
    if (-not $w) { return $null }
    $wr = $w.Current.BoundingRectangle
    $floor = $wr.Y + 150
    foreach ($e in $all) {
        if ($e.Current.ControlType.ProgrammaticName -ne 'ControlType.Button') { continue }
        if ($e.Current.Name -cne $name) { continue }
        $r = $e.Current.BoundingRectangle
        if ($r.Y -lt $floor) { continue }          # title bar strip
        if ($r.Width -le 0 -or $r.Height -le 0) { continue }
        return $e
    }
    return $null
}

function Current-Mode {
    for ($try = 1; $try -le 5; $try++) {
        try {
            $all = Get-Elements
            if ($all) {
                $n = ($all | Where-Object { $_.Current.Name -match '\((Offline|Realtime)\)' } |
                        Select-Object -First 1).Current.Name
                if ($n) { return $n }
            }
        } catch { }
        Start-Sleep -Seconds 3
    }
    return $null
}

$mode = Current-Mode
if ($mode -ne $Expect) { Say "mode is '$mode' but '$Expect' was expected - refusing"; exit 1 }

# Leftover dialogs are NOT closed by this script any more, and the reason is worth the
# space. `Close` appears twice in this window -- on the dialog and in the chrome -- in
# the same case, and the dialog's caption is drawn so it cannot be used to tell them
# apart. Matching by name closed ReStem. Matching case-sensitively closed it again.
# Matching case-sensitively and refusing anything in the top 150 pixels closed it a
# third time.
#
# Restarting is safe, verified, and keeps the quality mode, which the application
# remembers. So when Load is missing the script restarts rather than clicking: a
# restart costs twenty seconds and a mis-click costs the run.
if (-not (Find-ByName (Get-RestemWindow) "Load")) {
    Say "Load is unavailable -- restarting ReStem rather than clicking anything"
    $p = Get-Process -Name "ReStem*" -ErrorAction SilentlyContinue
    foreach ($proc in $p) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    Start-Sleep -Seconds 5
    & (Join-Path $root "restem_launch.ps1") | ForEach-Object { Say "  $_" }
    Start-Sleep -Seconds 3
    $mode = Current-Mode
    if ($mode -ne $Expect) {
        Say "after the restart the mode reads '$mode', not '$Expect' - refusing"
        exit 1
    }
}

$paths = @()
foreach ($t in $Tracks) {
    $p = Join-Path $root "restem_in\$t.wav"
    if (-not (Test-Path $p)) { Say "missing input: $t"; exit 1 }
    $paths += '"' + $p + '"'
}

$before = @{}
$watch = Join-Path $root "restem_export"
Get-ChildItem $watch -Recurse -Filter "*_midi.mid" -File -ErrorAction SilentlyContinue |
    ForEach-Object { $before[$_.FullName] = $true }

Say "=== queue: $($Tracks.Count) file(s), mode '$mode'"
$win = Get-RestemWindow
$load = Find-ByName $win "Load"
if (-not $load) { Say "Load unavailable - is a render in progress?"; exit 1 }
Invoke-Btn $load
Start-Sleep -Seconds 2
if (-not (Submit-FileDialog ($paths -join ' '))) { Say "could not submit the list"; exit 1 }
Wait-DialogGone 20 | Out-Null
Start-Sleep -Seconds 3

$start = Find-DialogButton "Start"
if (-not $start) { Say "no Batch Process dialog appeared - the product took one file only"; exit 1 }
Say "Batch Process dialog is up; output folder is whatever it shows - pressing Start"
$start.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()

$t0 = Get-Date
$deadline = $t0.AddMinutes($TimeoutMinutes)
$seen = 0
$quiet = 0
$stuckFor = 0
$lastCpu = $null
# The dialog closes when the batch STARTS, not when it ends -- an earlier version read
# that as completion and reported the run finished after 16 seconds. Completion is
# instead "no new output for long enough", measured against the folder ReStem writes
# one subdirectory per track into.
#
# "Long enough" cannot be a fixed silence, though. ReStem renders in a separate process,
# restem_offline, and a long recording can exceed any quiet window that is short enough
# to be useful -- this loop once declared the run over at 2 of 8 while that process was
# sitting at 99% of a core, six tracks still to go. So silence only counts while the
# renderer is idle: if its CPU time is still climbing, it is working and the wait resets.
function Renderer-Cpu {
    $p = @(Get-Process -Name "restem_offline" -ErrorAction SilentlyContinue)
    if (-not $p.Count) { return $null }
    ($p | Measure-Object -Property CPU -Sum).Sum
}
# A busy renderer resets the wait, but not forever. ReStem hangs reproducibly on at
# least one ENST recording -- 100% of a core, nothing written, for as long as it is
# left alone -- and with the reset alone that burned 48 minutes before a human stopped
# it. Silence this long means stuck whatever the CPU says.
$hardQuiet = 50   # 25 minutes at 30 s a turn
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 30
    $now = @(Get-ChildItem $watch -Recurse -Filter "*_midi.mid" -File -ErrorAction SilentlyContinue |
             Where-Object { -not $before.ContainsKey($_.FullName) })
    if ($now.Count -ne $seen) {
        $seen = $now.Count
        $quiet = 0
        $stuckFor = 0
        Say ("{0} of {1} done after {2:N1} min" -f $seen, $Tracks.Count,
             ((Get-Date) - $t0).TotalMinutes)
        if ($seen -ge $Tracks.Count) { break }
    } else {
        $cpu = Renderer-Cpu
        $stuckFor++
        if ($stuckFor -ge $hardQuiet) {
            Say ("nothing written for {0:N0} minutes though the renderer is busy - " +
                 "this is the hang, not a long track; stopping" -f ($stuckFor * 0.5))
            break
        }
        if ($null -ne $cpu -and $null -ne $lastCpu -and ($cpu - $lastCpu) -gt 1) {
            if ($quiet -ge 4) {
                Say ("still rendering: restem_offline took {0:N0}s of CPU in the last " +
                     "{1:N1} min, so the wait resets" -f ($cpu - $lastCpu), ($quiet * 0.5))
            }
            $quiet = 0
        } else {
            $quiet++
            if ($quiet -ge 20) {
                Say "no new output for 10 minutes and the renderer is idle - stopping"
                break
            }
        }
        $lastCpu = $cpu
    }
}

Say ("=== queue end after {0:N1} min, {1} new file(s)" -f
     ((Get-Date) - $t0).TotalMinutes, $seen)

# ReStem writes where its own dialog points, which is restem_export, and -Out did not
# reach it: the parameter used to create an empty folder and the closing line then named
# that folder as the location of the results. That is how two arms of a comparison end up
# pooled in one directory -- which happened, and was caught only because a drummer-3 batch
# in Best+ landed on top of nine renders made in the other mode. Names and timestamps
# separated them that time. Rather than trust that twice, the run now moves its own output
# out of the shared folder, so each arm is isolated by construction.
$moved = 0
foreach ($t in $Tracks) {
    $src = Join-Path $watch $t
    if (-not (Test-Path $src)) { continue }
    $fresh = @(Get-ChildItem $src -Recurse -Filter "*_midi.mid" -File -ErrorAction SilentlyContinue |
               Where-Object { -not $before.ContainsKey($_.FullName) })
    if (-not $fresh.Count) { continue }     # pre-existing render, not ours to move
    $dst = Join-Path $outDir $t
    if (Test-Path $dst) { Say "refusing to overwrite $t in $Out"; continue }
    Move-Item -LiteralPath $src -Destination $dst
    $moved++
}
Say ("=== {0} of {1} render(s) moved into {2}" -f $moved, $seen, $Out)
if ($moved -ne $seen) {
    Say "WARNING: $seen file(s) appeared but $moved moved - check $watch for strays"
}
