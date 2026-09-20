# Does ReStem's Load dialog accept a queue, and can each result be attributed?
#
# The per-file path costs about 150 seconds of fixed overhead per render. On MDB, where
# tracks average 57 seconds, that is tolerable. On ENST, where they average 30, it is
# most of the cost: measured renders ran at 6-11x realtime against 2.4x on MDB. If the
# dialog's CanSelectMultiple=True means the product will queue, the overhead amortises
# and a 210-recording run becomes affordable.
#
# Two things have to be true, and this tests both rather than assuming the first implies
# the second:
#
#   1. submitting several quoted paths renders several files, not just the first
#   2. each result can be told apart. ReStem overwrites one cache file per render, so
#      the only way to attribute an output is the input name it leaves in
#      cache\.stems.render-N-<hash>\input\, read at the moment the output changes.
#
# Read-only apart from the renders it asks for; it copies results into bench\ so nothing
# of the product's is touched.
#
#   powershell -File restem_batch_probe.ps1 -Tracks a,b,c

param(
    [Parameter(Mandatory = $true)][string[]] $Tracks,
    [int] $TimeoutMinutes = 20,
    [string] $Label = "probe"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Invoke-Expression (Get-Content "$root\restem_ui.ps1" -Raw)

$cacheRoot = Join-Path $env:APPDATA "ReStem 2\cache"
$cacheJson = Join-Path $cacheRoot "stems\trigger_events.json"
$outDir = Join-Path $root "bench\restem_modes"
$log = Join-Path $root "bench\restem_batch_probe.log"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

function Say([string] $m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    Write-Output $line
    Add-Content -Path $log -Value $line
}

function Current-Input {
    # Whichever render slot was touched most recently names the file being worked on.
    $d = Get-ChildItem $cacheRoot -Directory -Filter ".stems.render-*" -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $d) { return $null }
    $f = Get-ChildItem (Join-Path $d.FullName "input") -File -ErrorAction SilentlyContinue |
         Select-Object -First 1
    if ($f) { return [IO.Path]::GetFileNameWithoutExtension($f.Name) }
    return $null
}

$paths = @()
foreach ($t in $Tracks) {
    $p = Join-Path $root "restem_in\$t.wav"
    if (-not (Test-Path $p)) { Say "missing input: $t"; exit 1 }
    $paths += '"' + $p + '"'
}
$arg = $paths -join ' '

$before = [datetime]::MinValue
if (Test-Path $cacheJson) { $before = (Get-Item $cacheJson).LastWriteTime }

Say "=== queue probe: $($Tracks.Count) file(s) submitted in one dialog"
$win = Get-RestemWindow
$load = Find-ByName $win "Load"
if (-not $load) { Say "Load unavailable"; exit 1 }
Invoke-Btn $load
Start-Sleep -Seconds 2
if (-not (Submit-FileDialog $arg)) { Say "could not submit"; exit 1 }
Wait-DialogGone 20 | Out-Null

$t0 = Get-Date
$deadline = $t0.AddMinutes($TimeoutMinutes)
$seen = @{}
$last = $before
# Two seconds, not ten: if the queue amortises the overhead the renders arrive close
# together, and a poll slower than the gap silently merges two results into one.
while ((Get-Date) -lt $deadline -and $seen.Count -lt $Tracks.Count) {
    Start-Sleep -Seconds 2
    if (-not (Test-Path $cacheJson)) { continue }
    $mt = (Get-Item $cacheJson).LastWriteTime
    if ($mt -le $last) { continue }
    Start-Sleep -Seconds 3   # let the write settle
    $who = Current-Input
    $mt = (Get-Item $cacheJson).LastWriteTime
    $name = if ($who) { $who } else { "unattributed-$($seen.Count + 1)" }
    if ($seen.ContainsKey($name)) { $name = "$name-again" }
    Copy-Item $cacheJson (Join-Path $outDir "$name.$Label.json") -Force
    $seen[$name] = $true
    Say ("result {0} of {1} after {2:N1} min -> {3}" -f $seen.Count, $Tracks.Count,
         ((Get-Date) - $t0).TotalMinutes, $name)
    $last = $mt
}

Say ("=== probe end: {0} of {1} results in {2:N1} min" -f $seen.Count, $Tracks.Count,
     ((Get-Date) - $t0).TotalMinutes)
if ($seen.Count -lt $Tracks.Count) {
    Say "fewer results than files: the dialog took the selection but the product"
    Say "renders one at a time, so the per-file path stays."
} else {
    Say "every file produced a result: a queue exists and the overhead amortises."
}
foreach ($k in $seen.Keys) { Say "  attributed: $k" }
