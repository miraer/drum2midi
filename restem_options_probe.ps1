# Opens ReStem's Options and lists what is there, so the GPU backend can be found.
#
# On this machine ReStem's engine fails on Vulkan (Intel Arc, error 106) and has to run
# on D3D12. The 17 September comparison was made that way. A later run left at the
# default estimated 6583 hours for one track, which is the same symptom, so the backend
# has to be set before any timing or quality measurement means anything.
#
# Read-only by default: it opens the panel and reports. Pass -Set <text> to click a
# named item once you know what to look for.

param([string] $Set)

$ErrorActionPreference = "Stop"
Invoke-Expression (Get-Content "$PSScriptRoot\restem_ui.ps1" -Raw)

$win = Get-RestemWindow
if (-not $win) { "ReStem 2 is not running"; exit 1 }

function Show-Tree($scope, $label) {
    $all = $scope.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                          [System.Windows.Automation.Condition]::TrueCondition)
    "$label : $($all.Count) elements"
    $all | Where-Object { $_.Current.Name -or $_.Current.AutomationId } |
      Select-Object @{n='Name';e={$_.Current.Name}},
                    @{n='Type';e={$_.Current.LocalizedControlType}},
                    @{n='Id';e={$_.Current.AutomationId}} |
      Format-Table -AutoSize | Out-String -Width 110
}

foreach ($name in @("Options", "Settings")) {
    $btn = Find-ByName $win $name
    if (-not $btn) { "no '$name' button"; continue }
    "=== opening $name ==="
    try { Invoke-Btn $btn } catch { "could not invoke $name : $_"; continue }
    Start-Sleep -Seconds 2

    # the panel may be a child window or drawn inside the main one
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    $proc = (Get-Process "ReStem 2" | Select-Object -First 1).Id
    $cond = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $proc)
    $tops = $root.FindAll([System.Windows.Automation.TreeScope]::Children, $cond)
    "top-level windows now: $($tops.Count)"
    foreach ($t in $tops) { Show-Tree $t "  window '$($t.Current.Name)'" }

    if ($Set) {
        $item = $null
        foreach ($t in $tops) { if (-not $item) { $item = Find-ByName $t $Set } }
        if ($item) {
            "clicking '$Set'"
            try { Invoke-Btn $item } catch {
                # menu items often expose SelectionItem rather than Invoke
                try {
                    $item.GetCurrentPattern(
                      [System.Windows.Automation.SelectionItemPattern]::Pattern).Select()
                    "selected '$Set'"
                } catch { "found '$Set' but could neither invoke nor select it" }
            }
        } else { "no element named '$Set'" }
    }

    Start-Sleep -Seconds 1
    break
}
