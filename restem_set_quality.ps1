# Switches ReStem's offline quality mode and reports what it landed on.
#
# The control is a combo box with no accessible name; it only becomes findable as
# "Model Quality Selector" when no menu is open. Expanding it exposes the list items,
# which do have names - "Better (Offline)" and "Best (Offline) +".
#
#   powershell -File restem_set_quality.ps1 -Mode "Better (Offline)"
#   powershell -File restem_set_quality.ps1            # just report the current mode

param([string] $Mode)

$ErrorActionPreference = "Stop"
Invoke-Expression (Get-Content "$PSScriptRoot\restem_ui.ps1" -Raw)

$win = Get-RestemWindow
if (-not $win) { "ReStem 2 is not running"; exit 1 }

function Current-Mode($w) {
    $all = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                      [System.Windows.Automation.Condition]::TrueCondition)
    ($all | Where-Object { $_.Current.Name -match '\((Offline|Realtime)\)' } |
       Select-Object -First 1).Current.Name
}

"current mode: $(Current-Mode $win)"
if (-not $Mode) { exit 0 }

$sel = Find-ByName $win "Model Quality Selector"
if (-not $sel) { "no 'Model Quality Selector' - is a menu open?"; exit 1 }

# expand it so the options become part of the tree
try {
    $sel.GetCurrentPattern(
      [System.Windows.Automation.ExpandCollapsePattern]::Pattern).Expand()
} catch {
    try { Invoke-Btn $sel } catch { "could not open the selector: $_"; exit 1 }
}
Start-Sleep -Seconds 2

$root = [System.Windows.Automation.AutomationElement]::RootElement
$proc = (Get-Process "ReStem 2" | Select-Object -First 1).Id
$cond = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc)
$tops = $root.FindAll([System.Windows.Automation.TreeScope]::Children, $cond)

$target = $null
foreach ($t in $tops) {
    $all = $t.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                      [System.Windows.Automation.Condition]::TrueCondition)
    foreach ($e in $all) {
        if ($e.Current.Name -like "$Mode*") { $target = $e; break }
    }
    if ($target) { break }
}

if (-not $target) {
    "could not find an option starting with '$Mode'. What is on offer:"
    foreach ($t in $tops) {
        $all = $t.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                          [System.Windows.Automation.Condition]::TrueCondition)
        $all | Where-Object { $_.Current.Name -match '(?i)offline|realtime|better|best' } |
          ForEach-Object { "   '$($_.Current.Name)'  [$($_.Current.LocalizedControlType)]" }
    }
    exit 1
}

"selecting '$($target.Current.Name)'"
try {
    $target.GetCurrentPattern(
      [System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
} catch {
    try { Invoke-Btn $target } catch { "found it but could not select: $_"; exit 1 }
}
Start-Sleep -Seconds 2
"mode is now: $(Current-Mode (Get-RestemWindow))"
