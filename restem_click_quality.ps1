# Clicks ReStem's quality selector with the mouse, because its combo box does not
# respond to UI Automation's Expand or Invoke patterns -- the framework draws its own
# widgets and exposes only a label.
#
# Moves the physical cursor, so do not use the machine while it runs.
#
#   powershell -File restem_click_quality.ps1              # open the list and report
#   powershell -File restem_click_quality.ps1 -Pick "Better"

param([string] $Pick)

$ErrorActionPreference = "Stop"
Invoke-Expression (Get-Content "$PSScriptRoot\restem_ui.ps1" -Raw)

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Mouse {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(
      uint dwFlags, uint dx, uint dy, uint cButtons, UIntPtr dwExtraInfo);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int cmd);
  [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr h);
  public const uint LEFTDOWN = 0x0002, LEFTUP = 0x0004;
  public const int RESTORE = 9;
  public static void Click(int x, int y) {
    SetCursorPos(x, y);
    System.Threading.Thread.Sleep(150);
    mouse_event(LEFTDOWN, 0, 0, 0, UIntPtr.Zero);
    System.Threading.Thread.Sleep(60);
    mouse_event(LEFTUP, 0, 0, 0, UIntPtr.Zero);
  }
}
"@ -ErrorAction SilentlyContinue

$win = Get-RestemWindow
if (-not $win) { "ReStem 2 is not running"; exit 1 }
$p = Get-Process "ReStem 2" | Select-Object -First 1
# a minimised window has no on-screen rectangles, so nothing can be clicked until it
# is restored -- and SetForegroundWindow alone does not restore it
if ([Mouse]::IsIconic($p.MainWindowHandle)) {
    [Mouse]::ShowWindow($p.MainWindowHandle, [Mouse]::RESTORE) | Out-Null
    Start-Sleep -Seconds 2
}
[W32]::SetForegroundWindow($p.MainWindowHandle) | Out-Null
Start-Sleep -Milliseconds 800
$win = Get-RestemWindow

$sel = Find-ByName $win "Model Quality Selector"
if (-not $sel) {
    # The named selector is not always in the tree; the mode label always is, and it
    # sits on the control, so clicking the label opens the same list.
    $all = $win.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                        [System.Windows.Automation.Condition]::TrueCondition)
    $sel = $all | Where-Object {
        $_.Current.Name -match '\((Offline|Realtime)\)' -and
        $_.Current.BoundingRectangle.Width -gt 0
    } | Select-Object -First 1
}
if (-not $sel) { "neither the selector nor a mode label is on screen"; exit 1 }
$r = $sel.Current.BoundingRectangle
"selector at {0},{1} size {2}x{3}" -f [int]$r.X, [int]$r.Y, [int]$r.Width, [int]$r.Height
if ($r.Width -le 0) { "selector has no on-screen rectangle"; exit 1 }

[Mouse]::Click([int]($r.X + $r.Width / 2), [int]($r.Y + $r.Height / 2))
Start-Sleep -Seconds 2

# the popup is usually a separate top-level window
$root = [System.Windows.Automation.AutomationElement]::RootElement
$cond = New-Object System.Windows.Automation.PropertyCondition(
    [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $p.Id)
$tops = $root.FindAll([System.Windows.Automation.TreeScope]::Children, $cond)

$options = @()
foreach ($t in $tops) {
    $all = $t.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                      [System.Windows.Automation.Condition]::TrueCondition)
    foreach ($e in $all) {
        if ($e.Current.Name -match '(?i)\((Offline|Realtime)\)') {
            $rect = $e.Current.BoundingRectangle
            $options += [pscustomobject]@{
                Name = $e.Current.Name; Type = $e.Current.LocalizedControlType
                X = [int]($rect.X + $rect.Width / 2); Y = [int]($rect.Y + $rect.Height / 2)
                W = [int]$rect.Width
            }
        }
    }
}

"options visible now:"
$options | Format-Table -AutoSize | Out-String -Width 90

if ($Pick) {
    $t = $options | Where-Object { $_.Name -like "*$Pick*" -and $_.W -gt 0 } | Select-Object -First 1
    if (-not $t) { "nothing matching '$Pick' is clickable"; exit 1 }
    "clicking '$($t.Name)' at $($t.X),$($t.Y)"
    [Mouse]::Click($t.X, $t.Y)
    Start-Sleep -Seconds 2
    $w2 = Get-RestemWindow
    $all = $w2.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                       [System.Windows.Automation.Condition]::TrueCondition)
    $now = ($all | Where-Object { $_.Current.Name -match '\((Offline|Realtime)\)' } |
            Select-Object -First 1).Current.Name
    "mode is now: $now"
}
