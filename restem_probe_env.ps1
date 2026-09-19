# Reports whether ReStem can be driven by UI Automation on this machine.
#
# The automation in restem_ui.ps1 finds controls by name and AutomationId, which is
# resolution-independent and should move between machines. What does not move is the
# environment: UI Automation needs an interactive desktop session, so a headless server,
# a disconnected RDP session or a service account will fail here rather than in the
# middle of a four-hour batch.
#
# Run this first on any new machine. It changes nothing and clicks nothing.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File restem_probe_env.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File restem_probe_env.ps1 -Tree
#
# Send the output back rather than a summary of it: the element names are the thing
# that differs between versions, and a paraphrase loses exactly the detail needed.

param(
    [switch] $Tree,
    [string] $OutFile = ""
)

$ErrorActionPreference = "Continue"
$report = New-Object System.Collections.ArrayList
function Say($text) { [void]$report.Add($text); Write-Host $text }

Say "=== session ==="
Say ("  user                 : {0}" -f $env:USERNAME)
Say ("  session name         : {0}" -f $env:SESSIONNAME)
try {
    $ws = [System.Windows.Forms.SystemInformation]::UserInteractive
} catch { $ws = $null }
Add-Type -AssemblyName System.Windows.Forms -ErrorAction SilentlyContinue
Say ("  interactive          : {0}" -f [Environment]::UserInteractive)
$virt = $null
try {
    Add-Type -AssemblyName System.Drawing -ErrorAction Stop
    foreach ($s in [System.Windows.Forms.Screen]::AllScreens) {
        Say ("  screen {0,-12} primary={1,-6} {2},{3}  {4}x{5}" -f `
            $s.DeviceName.Replace("\\.\", ""), $s.Primary,
            $s.Bounds.X, $s.Bounds.Y, $s.Bounds.Width, $s.Bounds.Height)
    }
    # A window legitimately on a second monitor sits outside the primary screen, so
    # "outside the primary" is not a fault. The virtual screen is the union of all of
    # them, and that is the boundary a coordinate click actually has to respect.
    $virt = [System.Windows.Forms.SystemInformation]::VirtualScreen
    Say ("  virtual screen       : {0},{1}  {2}x{3}" -f `
        $virt.X, $virt.Y, $virt.Width, $virt.Height)
} catch {
    Say "  screens              : UNAVAILABLE - no desktop. UI Automation will not work."
}

Say ""
Say "=== process ==="
$procs = @(Get-Process -Name "ReStem*" -ErrorAction SilentlyContinue)
if ($procs.Count -eq 0) {
    Say "  ReStem is not running. Start it, then run this again."
} else {
    foreach ($p in $procs) {
        Say ("  pid {0}  {1}  mem {2} MB  title '{3}'" -f `
            $p.Id, $p.ProcessName, [math]::Round($p.WorkingSet64 / 1MB), $p.MainWindowTitle)
    }
}

Say ""
Say "=== UI Automation ==="
try {
    Add-Type -AssemblyName UIAutomationClient -ErrorAction Stop
    Add-Type -AssemblyName UIAutomationTypes -ErrorAction Stop
    Say "  assemblies loaded    : yes"
} catch {
    Say "  assemblies loaded    : NO - $($_.Exception.Message)"
    Say "  Without these the automation cannot run at all."
    if ($OutFile) { $report | Out-File -FilePath $OutFile -Encoding UTF8 }
    exit 1
}

$root = [System.Windows.Automation.AutomationElement]::RootElement
$cond = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ClassNameProperty, "Window")
$win = $null
foreach ($p in $procs) {
    $byPid = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $p.Id)
    $found = $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $byPid)
    if ($found) { $win = $found; break }
}

if (-not $win) {
    Say "  ReStem window        : NOT FOUND by UI Automation"
    Say "  If the process is running and the screen exists, this usually means the"
    Say "  session is not the one owning the desktop - check for a disconnected RDP."
    if ($OutFile) { $report | Out-File -FilePath $OutFile -Encoding UTF8 }
    exit 1
}

$r = $win.Current.BoundingRectangle
Say ("  ReStem window        : found, '{0}'" -f $win.Current.Name)
Say ("  bounds               : {0},{1} {2}x{3}" -f `
    [int]$r.X, [int]$r.Y, [int]$r.Width, [int]$r.Height)
if ($r.X -lt -10000 -or $r.Y -lt -10000) {
    Say "  STATE                : MINIMIZED. Reading the tree still works, but anything"
    Say "                         that clicks a screen coordinate will miss. Restore the"
    Say "                         window before running the batch."
} elseif ($virt -and ($r.X + $r.Width -le $virt.X -or $r.X -ge $virt.X + $virt.Width -or
                      $r.Y + $r.Height -le $virt.Y -or $r.Y -ge $virt.Y + $virt.Height)) {
    Say "  STATE                : OUTSIDE the virtual screen. Coordinate clicks will miss."
} elseif ($virt) {
    Say "  STATE                : on-screen, within the virtual desktop"
}

$all = $win.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                    [System.Windows.Automation.Condition]::TrueCondition)
Say ("  descendant elements  : {0}" -f $all.Count)

Say ""
Say "=== the controls the batch depends on ==="
$named = @{}
foreach ($e in $all) {
    $n = $e.Current.Name
    if ($n) { $named[$n] = $e.Current.LocalizedControlType }
}
foreach ($want in @("Model Quality Selector", "Bleed Reduction", "Load", "Process",
                    "Export", "Reprocess", "MIDI")) {
    if ($named.ContainsKey($want)) {
        Say ("  FOUND   '{0}' [{1}]" -f $want, $named[$want])
    } else {
        $near = @($named.Keys | Where-Object { $_ -like "*$want*" })
        if ($near.Count) { Say ("  similar '{0}' -> {1}" -f $want, ($near -join "; ")) }
        else { Say ("  absent  '{0}'" -f $want) }
    }
}

Say ""
Say "=== combo boxes, which is where this got hard last time ==="
$combos = @($all | Where-Object { $_.Current.LocalizedControlType -eq "combo box" })
Say ("  count                : {0}" -f $combos.Count)
$i = 0
foreach ($c in $combos) {
    $cr = $c.Current.BoundingRectangle
    $val = ""
    try {
        $vp = $c.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
        $val = $vp.Current.Value
    } catch {
        try {
            $sp = $c.GetCurrentPattern([System.Windows.Automation.SelectionPattern]::Pattern)
            $sel = $sp.Current.GetSelection()
            if ($sel.Count) { $val = $sel[0].Current.Name }
        } catch { $val = "(no value or selection pattern)" }
    }
    # IsEnabled is not decoration. A control that is present but greyed out reads
    # exactly like an active one if only the value is printed, and a probe that cannot
    # see the difference will report "nothing changed" when the thing that changed is
    # precisely this.
    $state = if ($c.Current.IsEnabled) { "enabled " } else { "DISABLED" }
    Say ("  [{0}] {1}  name '{2}' autoid '{3}' at {4},{5} {6}x{7}  value: {8}" -f `
        $i, $state, $c.Current.Name, $c.Current.AutomationId, [int]$cr.X, [int]$cr.Y,
        [int]$cr.Width, [int]$cr.Height, $val)
    $i++
}

$offCount = @($combos | Where-Object { -not $_.Current.IsEnabled }).Count
if ($offCount) {
    Say ("  {0} of {1} combo boxes are present but disabled" -f $offCount, $combos.Count)
}

if ($Tree) {
    Say ""
    Say "=== every named element ==="
    foreach ($k in ($named.Keys | Sort-Object)) {
        Say ("  '{0}' [{1}]" -f $k, $named[$k])
    }
}

Say ""
Say "=== verdict ==="
if ($combos.Count -gt 0 -and $all.Count -gt 10) {
    Say "  The window is readable. Next step is restem_quality_probe.ps1 to see whether"
    Say "  the quality mode can be READ, before anything tries to change it."
} else {
    Say "  The window was found but exposes almost nothing. Automation by name will not"
    Say "  work here; say so rather than falling back to blind coordinate clicking."
}

if ($OutFile) {
    $report | Out-File -FilePath $OutFile -Encoding UTF8
    Write-Host ""
    Write-Host "written to $OutFile"
}
