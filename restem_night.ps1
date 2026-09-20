# Keeps ReStem working through the night without a gap between batches.
#
# The declared 60 are rendered first and analysed on their own, because that sample was
# fixed before anything was rendered and its value is that it cannot be reshaped by what
# came back. Everything after it is an extension and is reported as one. Rendering more
# recordings does not touch the declared sample; quietly folding them into it would.
#
# Why a watcher rather than one enormous queue: ReStem's batch dialog takes the list it
# is given and the run cannot be added to once started, so the only way to use the whole
# night is to submit the next batch the moment the previous one stops producing.
#
#   powershell -File restem_night.ps1

$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$log = Join-Path $root "bench\restem_night.log"
$watch = Join-Path $root "restem_export"

function Say([string] $m) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    Write-Output $line
    Add-Content -Path $log -Value $line
}

function Rendered {
    @(Get-ChildItem $watch -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -notmatch '^MusicDelta' } | ForEach-Object { $_.Name })
}

function Wait-Idle([int] $quietMinutes = 8, [int] $capMinutes = 600) {
    # "Idle" is no new output folder for a while. The dialog closes when the batch
    # starts, so it says nothing about the end, and the worker process comes and goes
    # between tracks -- neither is a completion signal.
    $lastCount = (Rendered).Count
    $lastChange = Get-Date
    $cap = (Get-Date).AddMinutes($capMinutes)
    while ((Get-Date) -lt $cap) {
        Start-Sleep -Seconds 60
        $n = (Rendered).Count
        if ($n -ne $lastCount) {
            $lastCount = $n
            $lastChange = Get-Date
            Say "  $n rendered"
        } elseif (((Get-Date) - $lastChange).TotalMinutes -ge $quietMinutes) {
            Say "  quiet for $quietMinutes min at $n rendered"
            return $true
        }
    }
    Say "  cap reached"
    return $false
}

$declared = @(Get-Content (Join-Path $root "bench\enst60.txt") -Encoding UTF8)
$rest = @(Get-Content (Join-Path $root "bench\enst_rest.txt") -Encoding UTF8)

Say "=== night run: $($declared.Count) declared, $($rest.Count) extension"

# Stage 1: whatever is left of the declared sample.
Wait-Idle | Out-Null
$todo = @($declared | Where-Object { (Rendered) -notcontains $_ })
if ($todo.Count) {
    Say "declared sample incomplete: $($todo.Count) left, resubmitting"
    & (Join-Path $root "restem_batch_queue.ps1") -Tracks $todo -TimeoutMinutes 300
    Wait-Idle | Out-Null
}
$done = @($declared | Where-Object { (Rendered) -contains $_ })
Say "declared sample: $($done.Count) of $($declared.Count) rendered"

# Stage 2: the extension, in chunks so one crash cannot cost the whole night.
$chunk = 25
for ($i = 0; $i -lt $rest.Count; $i += $chunk) {
    $slice = @($rest[$i..([Math]::Min($i + $chunk - 1, $rest.Count - 1))] |
               Where-Object { (Rendered) -notcontains $_ })
    if (-not $slice.Count) { continue }
    Say "extension chunk: $($slice.Count) recordings"
    & (Join-Path $root "restem_batch_queue.ps1") -Tracks $slice -TimeoutMinutes 300
    if (-not (Wait-Idle)) { break }
}

Say ("=== night run end: " + (Rendered).Count + " recordings rendered in total")
