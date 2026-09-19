# Does ReStem's window width change which controls exist in its automation tree?
#
# Two machines read different trees from the same version: 1363x916 here shows six combo
# boxes including four per-tom note selectors, 1090x917 there shows two. Two explanations
# have already been tested and refuted -- session state, by probing with a file loaded,
# and the NOTE mode setting, by switching it here and finding the selectors unmoved.
#
# Window width is the remaining visible difference. A layout that drops controls when it
# runs out of room would produce exactly this, and it is testable without a second
# machine: resize, re-read, resize back.
#
# Restores the original size whatever happens, because leaving somebody's window a
# different shape is a rude way to run an experiment.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File restem_width_probe.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File restem_width_probe.ps1 -Widths 1090,1200

param(
    [int[]] $Widths = @(1090),
    [int] $SettleSeconds = 2
)

Add-Type -AssemblyName UIAutomationClient -ErrorAction Stop
Add-Type -AssemblyName UIAutomationTypes -ErrorAction Stop

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win {
    [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int t, bool repaint);
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
    [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
}
"@ -ErrorAction SilentlyContinue

function Get-Win {
    $procs = @(Get-Process -Name "ReStem*" -ErrorAction SilentlyContinue)
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    foreach ($p in $procs) {
        $cond = New-Object System.Windows.Automation.PropertyCondition(
            [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $p.Id)
        $el = $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $cond)
        if ($el) { return @{ El = $el; Handle = [IntPtr]$el.Current.NativeWindowHandle } }
    }
    return $null
}

function Read-Tree($el) {
    $all = $el.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                       [System.Windows.Automation.Condition]::TrueCondition)
    $combos = @()
    foreach ($e in $all) {
        if ($e.Current.LocalizedControlType -eq "combo box") {
            $v = ""
            try {
                $vp = $e.GetCurrentPattern([System.Windows.Automation.ValuePattern]::Pattern)
                $v = $vp.Current.Value
            } catch { $v = "?" }
            $combos += $v
        }
    }
    return @{ Count = $all.Count; Combos = $combos }
}

$w = Get-Win
if (-not $w) { Write-Output "ReStem is not running"; exit 1 }

$rect = New-Object Win+RECT
[void][Win]::GetWindowRect($w.Handle, [ref]$rect)
$origW = $rect.Right - $rect.Left
$origH = $rect.Bottom - $rect.Top
Write-Output ("original  {0}x{1} at {2},{3}" -f $origW, $origH, $rect.Left, $rect.Top)

$before = Read-Tree $w.El
Write-Output ("  elements {0}, combos {1}: {2}" -f `
    $before.Count, $before.Combos.Count, ($before.Combos -join ", "))

try {
    foreach ($width in $Widths) {
        [void][Win]::MoveWindow($w.Handle, $rect.Left, $rect.Top, $width, $origH, $true)
        Start-Sleep -Seconds $SettleSeconds
        $w2 = Get-Win
        $after = Read-Tree $w2.El
        Write-Output ""
        Write-Output ("at width {0}" -f $width)
        Write-Output ("  elements {0}, combos {1}: {2}" -f `
            $after.Count, $after.Combos.Count, ($after.Combos -join ", "))
        if ($after.Combos.Count -ne $before.Combos.Count) {
            Write-Output "  <-- the count CHANGED with width"
        } else {
            Write-Output "  same count as at the original width"
        }
    }
} finally {
    [void][Win]::MoveWindow($w.Handle, $rect.Left, $rect.Top, $origW, $origH, $true)
    Start-Sleep -Seconds 1
    Write-Output ""
    Write-Output ("restored to {0}x{1}" -f $origW, $origH)
}
