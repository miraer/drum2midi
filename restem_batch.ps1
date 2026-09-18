# Batch-runs every wav from restem_in through ReStem 2 and collects trigger_events.json.
# The engine writes into its cache, so the file is copied out after each render.
# Vulkan on Intel Arc crashes the engine, so the GUI must be started with it disabled.

$root = $PSScriptRoot
$inDir = "$root\restem_in"
$outDir = "$root\restem_events"
$cacheJson = Join-Path $env:APPDATA "ReStem 2\cache\stems\trigger_events.json"

Invoke-Expression (Get-Content "$root\restem_ui.ps1" -Raw)
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$files = Get-ChildItem $inDir -Filter *.wav | Sort-Object Name
$done = 0
$failed = @()
$t0 = Get-Date

foreach ($f in $files) {
    $dest = Join-Path $outDir "$($f.BaseName).json"
    if (Test-Path $dest) { $done++; continue }

    $before = [datetime]::MinValue
    if (Test-Path $cacheJson) { $before = (Get-Item $cacheJson).LastWriteTime }

    $win = Get-RestemWindow
    if (-not $win) { Write-Output "ReStem window gone"; break }

    $load = Find-ByName $win "Load"
    if (-not $load) {
        Write-Output "Load unavailable, skipping $($f.Name)"
        $failed += $f.Name
        continue
    }

    Invoke-Btn $load
    Start-Sleep -Seconds 2
    if (-not (Submit-FileDialog $f.FullName)) {
        Write-Output "could not submit path: $($f.Name)"
        $failed += $f.Name
        continue
    }
    Wait-DialogGone 20 | Out-Null

    $deadline = (Get-Date).AddSeconds(400)
    $ok = $false
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        if (Test-Path $cacheJson) {
            $mt = (Get-Item $cacheJson).LastWriteTime
            if ($mt -gt $before) { Start-Sleep -Seconds 3; $ok = $true; break }
        }
        $w = Get-RestemWindow
        $err = Find-ByName $w "OK"
        if ($err) {
            try { Invoke-Btn $err } catch {}
            Write-Output "render error on $($f.Name)"
            break
        }
    }

    if ($ok) {
        Copy-Item $cacheJson $dest -Force
        $done++
        $el = ((Get-Date) - $t0).TotalSeconds
        $left = ($el / [Math]::Max($done, 1)) * ($files.Count - $done) / 60
        Write-Output ("[{0,2}/{1}] {2}  (eta {3:N0} min)" -f $done, $files.Count, $f.BaseName, $left)
    } else {
        $failed += $f.Name
        Write-Output "timed out: $($f.Name)"
    }
}

Write-Output ""
Write-Output "collected $done of $($files.Count)"
if ($failed.Count -gt 0) { Write-Output "failed: $($failed -join ', ')" }
