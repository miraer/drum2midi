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
    # Scoped by parent, not by name. `Close` is not unique -- the window's title bar
    # has one too, and invoking the first match by name closed ReStem outright, which
    # cost a restart and very nearly the mode setting with it. So: find the dialog's
    # own `Change...` button, walk to its parent, and look only among that parent's
    # descendants. Nothing outside the dialog can match.
    $all = Get-Elements
    if (-not $all) { return $null }
    $anchor = $null
    foreach ($e in $all) {
        if ($e.Current.Name -eq 'Change...' -and
            $e.Current.ControlType.ProgrammaticName -eq 'ControlType.Button') { $anchor = $e; break }
    }
    if (-not $anchor) { return $null }
    $walker = [System.Windows.Automation.TreeWalker]::ControlViewWalker
    $parent = $walker.GetParent($anchor)
    if (-not $parent) { return $null }
    $kids = $parent.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                            [System.Windows.Automation.Condition]::TrueCondition)
    foreach ($k in $kids) {
        if ($k.Current.Name -eq $name -and
            $k.Current.ControlType.ProgrammaticName -eq 'ControlType.Button') { return $k }
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

# A dialog left over from an earlier run holds its own queue and swallows the next
# Load, so every later track reports a render error. Clear it before submitting.
$stale = Find-DialogButton "Close"
if ($stale) {
    Say "closing a leftover Batch Process dialog"
    $stale.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
    Start-Sleep -Seconds 3
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
# The dialog closes when the batch STARTS, not when it ends -- an earlier version read
# that as completion and reported the run finished after 16 seconds. Completion is
# instead "no new output for long enough", measured against the folder ReStem writes
# one subdirectory per track into.
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 30
    $now = @(Get-ChildItem $watch -Recurse -Filter "*_midi.mid" -File -ErrorAction SilentlyContinue |
             Where-Object { -not $before.ContainsKey($_.FullName) })
    if ($now.Count -ne $seen) {
        $seen = $now.Count
        $quiet = 0
        Say ("{0} of {1} done after {2:N1} min" -f $seen, $Tracks.Count,
             ((Get-Date) - $t0).TotalMinutes)
        if ($seen -ge $Tracks.Count) { break }
    } else {
        $quiet++
        if ($quiet -ge 20) { Say "no new output for 10 minutes - stopping"; break }
    }
}

Say ("=== queue end after {0:N1} min, {1} new file(s) in {2}" -f
     ((Get-Date) - $t0).TotalMinutes, $seen, $Out)
