# Probes the ReStem window for the QUALITY selector, without changing anything.
#
# We need to know which offline mode the MDB comparison actually ran in, and the
# exported JSON does not record it. Before deciding whether the three modes can be
# tested automatically, find out whether the control is reachable through UI
# Automation at all -- some plugin frameworks draw their own widgets and expose
# nothing to the accessibility tree.
#
# Read-only: it lists what it finds and stops.

$ErrorActionPreference = "Stop"
Invoke-Expression (Get-Content "$PSScriptRoot\restem_ui.ps1" -Raw)

$win = Get-RestemWindow
if (-not $win) { "ReStem 2 is not running"; exit 1 }
"window: $($win.Current.Name)"

$all = $win.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                    [System.Windows.Automation.Condition]::TrueCondition)
"elements exposed: $($all.Count)"

$interesting = @()
foreach ($e in $all) {
    $n = $e.Current.Name
    $t = $e.Current.LocalizedControlType
    $id = $e.Current.AutomationId
    if ($n -match '(?i)quality|offline|better|best|bleed|realtime|live' -or
        $id -match '(?i)quality|offline|bleed') {
        $interesting += [pscustomobject]@{ Name = $n; Type = $t; AutomationId = $id }
    }
}

if ($interesting) {
    "`n=== controls that look like the quality selector ==="
    $interesting | Format-Table -AutoSize | Out-String -Width 110
} else {
    "`n no element mentions quality, offline, better, best or bleed"
}

"`n=== every combo box, list and button (first 40) ==="
$all | Where-Object {
    $_.Current.LocalizedControlType -in @('combo box', 'list', 'list item', 'button')
} | Select-Object -First 40 @{n='Name';e={$_.Current.Name}},
                            @{n='Type';e={$_.Current.LocalizedControlType}},
                            @{n='Id';e={$_.Current.AutomationId}} |
  Format-Table -AutoSize | Out-String -Width 110
