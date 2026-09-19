# Dumps every UI Automation element under the ReStem window, named or not.
#
# restem_probe_env.ps1 lists named elements, which is enough to find controls by name but
# hides anything anonymous -- and opening ReStem's trigger-range editor adds exactly one
# element with no name. A tool that only prints names reports "nothing new", which is the
# same failure as printing values without enabled state.
#
# Writes one line per element so two states can be diffed with Compare-Object.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File restem_dump_tree.ps1 -OutFile a.txt
#   Compare-Object (gc a.txt) (gc b.txt)

param([string] $OutFile = "")

Add-Type -AssemblyName UIAutomationClient -ErrorAction Stop
Add-Type -AssemblyName UIAutomationTypes -ErrorAction Stop

$procs = @(Get-Process -Name "ReStem*" -ErrorAction SilentlyContinue)
if ($procs.Count -eq 0) { Write-Output "ReStem is not running"; exit 1 }

$root = [System.Windows.Automation.AutomationElement]::RootElement
$win = $null
foreach ($p in $procs) {
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $p.Id)
    $found = $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $cond)
    if ($found) { $win = $found; break }
}
if (-not $win) { Write-Output "no ReStem window in the automation tree"; exit 1 }

$all = $win.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                    [System.Windows.Automation.Condition]::TrueCondition)

$lines = New-Object System.Collections.ArrayList
foreach ($e in $all) {
    $c = $e.Current
    $r = $c.BoundingRectangle
    $value = ""
    foreach ($pat in @([System.Windows.Automation.ValuePattern]::Pattern,
                       [System.Windows.Automation.RangeValuePattern]::Pattern)) {
        try {
            $p = $e.GetCurrentPattern($pat)
            $value = [string]$p.Current.Value
            if ($value) { break }
        } catch {}
    }
    $patterns = @()
    foreach ($nm in "Invoke", "Toggle", "Value", "RangeValue", "Selection",
                     "SelectionItem", "ExpandCollapse", "Scroll") {
        try {
            $prop = [System.Windows.Automation.AutomationElement]::"Is${nm}PatternAvailableProperty"
            if ($e.GetCurrentPropertyValue($prop)) { $patterns += $nm }
        } catch {}
    }
    [void]$lines.Add(("{0,-18} name='{1}' id='{2}' enabled={3} at {4},{5} {6}x{7} value='{8}' pat={9}" -f `
        $c.LocalizedControlType, $c.Name, $c.AutomationId, $c.IsEnabled,
        [int]$r.X, [int]$r.Y, [int]$r.Width, [int]$r.Height, $value, ($patterns -join "+")))
}

$sorted = $lines | Sort-Object
Write-Output ("{0} elements" -f $all.Count)
$sorted | ForEach-Object { Write-Output $_ }

if ($OutFile) {
    $sorted | Out-File -FilePath $OutFile -Encoding UTF8
    Write-Output ""
    Write-Output "written to $OutFile"
}
