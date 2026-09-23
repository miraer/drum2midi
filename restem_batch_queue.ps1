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
    [string[]] $Tracks,
    [string] $Expect = "Best (Offline) +",
    [int] $TimeoutMinutes = 240,
    [string] $Out = "bench\restem_batch_out",
    [switch] $DefineOnly
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Invoke-Expression (Get-Content "$root\restem_ui.ps1" -Raw)

$log = Join-Path $root "bench\restem_batch_queue.log"

function Say([string] $m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    Write-Output $line
    Add-Content -Path $log -Value $line
}

# ReStem writes one folder per track into the folder its dialog shows, restem_export, and a
# render of a track that is already there lands on top of it. For a second arm that is not
# an edge case, it is the whole job: the ten drummer-2 recordings queued for Best+ on 22.09
# each already had a folder there holding the OTHER arm, and the move below would then have
# carried the overwritten folder off into -Out. The top-up would have deleted exactly the
# pairs it existed to complete. So existing folders of queued tracks are moved aside before
# Start and put back afterwards, and each arm stays isolated even when ReStem is not.
function Stash-Existing([string[]] $names, [string] $from, [string] $to) {
    $held = @()
    foreach ($t in $names) {
        $src = Join-Path $from $t
        if (-not (Test-Path -LiteralPath $src)) { continue }
        New-Item -ItemType Directory -Force -Path $to | Out-Null
        Move-Item -LiteralPath $src -Destination (Join-Path $to $t)
        $held += $t
    }
    , $held
}

function Restore-Stash([string[]] $names, [string] $from, [string] $to) {
    $back = 0
    foreach ($t in $names) {
        $src = Join-Path $from $t
        if (-not (Test-Path -LiteralPath $src)) { continue }
        $dst = Join-Path $to $t
        if (Test-Path -LiteralPath $dst) {
            Say "cannot put $t back: a render this run did not move is in its place; both kept, the original is in $from" | Out-Host
            continue
        }
        Move-Item -LiteralPath $src -Destination $dst
        $back++
    }
    if ((Test-Path -LiteralPath $from) -and -not (Get-ChildItem -LiteralPath $from -Force)) {
        Remove-Item -LiteralPath $from
    }
    $back
}

# ReStem names every file it gives up on in its own telemetry log, and nowhere else: the
# window shows a drawn dialog, the batch simply moves on, and the log is deleted when ReStem
# exits. Without reading it, a renderer that crashed on every file looked exactly like one
# that hung -- 25 minutes of silence either way -- and that is how the 22.09 batch was
# reported as a hang while the real message was "ReStem stopped unexpectedly (Error 106)",
# file after file. Only the latest run in the log counts; earlier runs share the file.
function Batch-Failures([string[]] $lines) {
    $begin = -1
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '\[batch\] run started:') { $begin = $i }
    }
    $fails = @()
    if ($begin -lt 0) { return , $fails }
    for ($i = $begin + 1; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '\[batch\] file (\d+)/(\d+) terminal state=4 \((.*)\)\s*$') {
            $fails += [pscustomobject]@{ N = [int]$Matches[1]; Of = [int]$Matches[2]; Why = $Matches[3] }
        }
    }
    , $fails
}

function Telemetry-Log {
    Get-ChildItem (Join-Path $env:APPDATA "ReStem 2\telemetry") -Filter "log-*.txt" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
}

# True when this ReStem session's last batch ran to the end and nothing has opened since,
# which is when its "N files finished" dialog is still on screen. While it is, Load still
# takes files but no Batch Process dialog appears, and the queue used to report that as
# "the product took one file only" -- twice on 23.09, after a probe batch had finished
# cleanly. The dialog is drawn and its button shares a name with the window's own close
# button, so it is cleared by restarting ReStem, never by clicking.
function Batch-Ended([string[]] $lines) {
    $last = $null
    foreach ($l in $lines) {
        if ($l -match '\[life\] .* created') { $last = $null }
        elseif ($l -match '\[batch\] (run started|run finished|setup opened)') { $last = $Matches[1] }
    }
    $last -eq "run finished"
}

# -DefineOnly loads the functions above and stops, so the stash can be tested without
# ReStem, the way restem_batch_mode.ps1 lets its guard be tested.
if ($DefineOnly) { return }
if (-not $Tracks) { throw "-Tracks is required" }

$outDir = Join-Path $root $Out
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$stash = Join-Path $root "bench\restem_stash"
if ((Test-Path -LiteralPath $stash) -and (Get-ChildItem -LiteralPath $stash -Force)) {
    Say "bench\restem_stash still holds folders from an interrupted run - move them back into restem_export first"
    exit 1
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
# Restarting is safe and keeps the quality mode -- but only if ReStem is asked to close
# rather than killed. The mode lives in memory and is written to its settings file on a
# clean exit: `Stop-Process -Force` discards it, and this script used to do exactly that
# while a comment here claimed the application remembered. It did not. The guard below
# is the only reason that never produced renders filed under the wrong arm.
#
# WM_CLOSE via CloseMainWindow() exits in about two seconds and updates the settings
# file. Force is kept as a fallback for a hung renderer, where the mode is lost anyway
# and the guard will catch it.
$tl = Telemetry-Log
$leftover = [bool]($tl -and (Batch-Ended @(Get-Content -LiteralPath $tl.FullName -ErrorAction SilentlyContinue)))
if ($leftover -or -not (Find-ByName (Get-RestemWindow) "Load")) {
    if ($leftover) {
        Say "the last batch's 'finished' dialog is still up, and ReStem opens no new batch while it is - restarting ReStem rather than clicking it"
    } else {
        Say "Load is unavailable -- restarting ReStem rather than clicking anything"
    }
    foreach ($proc in @(Get-Process -Name "ReStem 2" -ErrorAction SilentlyContinue)) {
        $null = $proc.CloseMainWindow()
        for ($i = 0; $i -lt 15; $i++) {
            Start-Sleep -Seconds 1
            if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) { break }
        }
        if (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue) {
            Say "  it would not close on request; forcing, and the mode will be lost"
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        } else {
            Say "  closed cleanly, so the quality mode is saved"
        }
    }
    foreach ($proc in @(Get-Process -Name "restem_offline" -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
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

$watch = Join-Path $root "restem_export"

# Progress is "how many of MY tracks have a MIDI written since the queue started", not
# "how many paths are new". Those differ whenever a track is rendered a second time --
# a re-render overwrites a path that already existed, so the old test counted it as
# nothing and a queue of seventeen sat at six while thirteen were on disk. Asking about
# the queued tracks by name and by timestamp is correct either way.
function Queue-Done([datetime] $since) {
    $n = 0
    foreach ($t in $Tracks) {
        $m = Join-Path $watch "$t\$t`_midi.mid"
        if ((Test-Path $m) -and ((Get-Item $m).LastWriteTime -gt $since)) { $n++ }
    }
    $n
}

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

$sameDir = ((Resolve-Path $outDir).Path.TrimEnd('\') -eq (Resolve-Path $watch).Path.TrimEnd('\'))
$held = @()
if (-not $sameDir) {
    $held = Stash-Existing $Tracks $watch $stash
    if ($held.Count) {
        Say ("moved {0} existing folder(s) of queued tracks aside to bench\restem_stash, so this run cannot overwrite them" -f $held.Count)
    }
}

try {
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
$failSeen = 0
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 30
    $tl = Telemetry-Log
    if ($tl) {
        $fails = Batch-Failures @(Get-Content -LiteralPath $tl.FullName -ErrorAction SilentlyContinue)
        for ($k = $failSeen; $k -lt $fails.Count; $k++) {
            Say ("file {0} of {1} failed inside ReStem: {2}" -f $fails[$k].N, $fails[$k].Of, $fails[$k].Why)
        }
        $failSeen = $fails.Count
        # Two failures and nothing rendered is the renderer failing, not bad luck with a file:
        # tonight a recording that rendered cleanly on 21.09 failed the same way.
        if ($seen -eq 0 -and $failSeen -ge 2) {
            Say "ReStem failed $failSeen file(s) outright and rendered none - the renderer is failing, not hanging; stopping"
            break
        }
    }
    $now = Queue-Done $t0
    if ($now -ne $seen) {
        $seen = $now
        $quiet = 0
        $stuckFor = 0
        Say ("{0} of {1} done after {2:N1} min" -f $seen, $Tracks.Count,
             ((Get-Date) - $t0).TotalMinutes)
        if ($seen -ge $Tracks.Count) { break }
    } else {
        $cpu = Renderer-Cpu
        $stuckFor++
        if ($stuckFor -ge $hardQuiet) {
            Say (("nothing written for {0:N0} minutes though the renderer is busy - " +
                  "this is the hang, not a long track; stopping") -f ($stuckFor * 0.5))
            break
        }
        if ($null -ne $cpu -and $null -ne $lastCpu -and ($cpu - $lastCpu) -gt 1) {
            if ($quiet -ge 4) {
                Say (("still rendering: restem_offline took {0:N0}s of CPU in the last " +
                      "{1:N1} min, so the wait resets") -f ($cpu - $lastCpu), ($quiet * 0.5))
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

# Kept because ReStem deletes it on exit, and it is the file their error message asks for.
$tl = Telemetry-Log
if ($tl) {
    $keep = Join-Path $root ("bench\restem_telemetry\{0}.txt" -f (Get-Date -Format "yyyyMMdd-HHmm"))
    New-Item -ItemType Directory -Force -Path (Split-Path $keep) | Out-Null
    Copy-Item -LiteralPath $tl.FullName -Destination $keep -Force
    Say "ReStem's own log for this run kept as bench\restem_telemetry\$(Split-Path $keep -Leaf)"
}

# ReStem writes where its own dialog points, which is restem_export, and -Out did not
# reach it: the parameter used to create an empty folder and the closing line then named
# that folder as the location of the results. That is how two arms of a comparison end up
# pooled in one directory -- which happened, and was caught only because a drummer-3 batch
# in Best+ landed on top of nine renders made in the other mode. Names and timestamps
# separated them that time. Rather than trust that twice, the run now moves its own output
# out of the shared folder, so each arm is isolated by construction.
$moved = 0
if ($sameDir) {
    Say "-Out is the folder ReStem already writes to, so nothing needs moving"
}
foreach ($t in $Tracks) {
    if ($sameDir) { break }
    $src = Join-Path $watch $t
    if (-not (Test-Path $src)) { continue }
    $m = Join-Path $src "$t`_midi.mid"
    # Same test as the progress counter, for the same reason: a re-render overwrites an
    # existing path, so asking whether the path is new would leave it behind.
    if (-not ((Test-Path $m) -and ((Get-Item $m).LastWriteTime -gt $t0))) { continue }
    $dst = Join-Path $outDir $t
    if (Test-Path $dst) { Say "refusing to overwrite $t in $Out"; continue }
    Move-Item -LiteralPath $src -Destination $dst
    $moved++
}
Say ("=== {0} of {1} render(s) moved into {2}" -f $moved, $seen, $Out)
if (-not $sameDir -and $moved -ne $seen) {
    Say "WARNING: $seen file(s) appeared but $moved moved - check $watch for strays"
}
} finally {
    # In `finally` so an exception or an early break still puts the other arm back. A kill
    # by PID skips it; the check at the top then refuses to run until the stash is emptied.
    if ($held.Count) {
        $back = Restore-Stash $held $stash $watch
        Say ("=== {0} of {1} stashed folder(s) put back into restem_export" -f $back, $held.Count)
    }
}
