# Enables MIDI OUTPUT on every stem tab in ReStem 2, verifying each one.
# The panel set differs per stem (Kick has a Drumagog replacer, Hi-Hat does not),
# so the MIDI panel is always the RIGHTMOST one in the row and its X position moves.
# The tree must be re-read after the tab click, otherwise the previous layout is seen.

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class MouseNat {
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);
  [DllImport("user32.dll")] public static extern void mouse_event(int f, int x, int y, int d, int e);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool BringWindowToTop(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool c);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, IntPtr p);
  [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
}
"@ -ErrorAction SilentlyContinue

$root = $PSScriptRoot
Invoke-Expression (Get-Content "$root\restem_ui.ps1" -Raw)

function Focus-ReStem {
    $p = Get-Process "ReStem 2" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $p) { return $false }
    $h = $p.MainWindowHandle
    [MouseNat]::ShowWindow($h, 9) | Out-Null
    [MouseNat]::BringWindowToTop($h) | Out-Null
    # attach to the foreground thread, otherwise Windows refuses to hand over focus
    $fg = [MouseNat]::GetForegroundWindow()
    $t1 = [MouseNat]::GetWindowThreadProcessId($fg, [IntPtr]::Zero)
    $t2 = [MouseNat]::GetCurrentThreadId()
    [MouseNat]::AttachThreadInput($t1, $t2, $true) | Out-Null
    [MouseNat]::SetForegroundWindow($h) | Out-Null
    [MouseNat]::AttachThreadInput($t1, $t2, $false) | Out-Null
    Start-Sleep -Milliseconds 400
    [MouseNat]::GetForegroundWindow() -eq $h
}

function Click-Point($x, $y) {
    Focus-ReStem | Out-Null
    [MouseNat]::SetCursorPos($x, $y) | Out-Null
    Start-Sleep -Milliseconds 300
    [MouseNat]::mouse_event(0x02, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 90
    [MouseNat]::mouse_event(0x04, 0, 0, 0, 0)
    Start-Sleep -Milliseconds 700
}

function Get-PanelToggles {
    $w = Get-RestemWindow
    $els = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition)
    $out = @()
    foreach ($e in $els) {
        $b = $e.Current.BoundingRectangle
        if ($e.Current.LocalizedControlType -eq "check box" -and
            $b.Y -gt 1040 -and $b.Y -lt 1070 -and $b.Width -gt 35) {
            $out += [pscustomobject]@{
                El = $e
                X  = [int]$b.X
                CX = [int]($b.X + $b.Width / 2)
                CY = [int]($b.Y + $b.Height / 2)
            }
        }
    }
    $out | Sort-Object X
}

function Get-TabPoints {
    $w = Get-RestemWindow
    $els = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition)
    $t = @{}
    foreach ($e in $els) {
        $n = $e.Current.Name
        if ($n -in @("Kick", "Snare", "Hi-Hat", "Toms", "Ride", "Crash", "Other")) {
            $b = $e.Current.BoundingRectangle
            $t[$n] = @([int]($b.X + $b.Width / 2), [int]($b.Y + $b.Height / 2))
        }
    }
    $t
}

$tabs = Get-TabPoints
$order = @("Kick", "Snare", "Hi-Hat", "Toms", "Ride", "Crash", "Other")
$results = @()

foreach ($name in $order) {
    if (-not $tabs.ContainsKey($name)) { Write-Output "$name : tab not found"; continue }

    Click-Point $tabs[$name][0] $tabs[$name][1]
    Start-Sleep -Milliseconds 900          # let the panel row redraw

    $toggles = Get-PanelToggles
    if ($toggles.Count -eq 0) { Write-Output "$name : no panel toggles"; continue }
    $midi = $toggles[-1]                   # MIDI OUTPUT is always the rightmost panel

    $state = $midi.El.GetCurrentPattern([System.Windows.Automation.TogglePattern]::Pattern).Current.ToggleState
    if ($state -ne "On") {
        Click-Point $midi.CX $midi.CY      # mouse click is more reliable than Toggle() in JUCE
        Start-Sleep -Milliseconds 500
    }

    # re-read to confirm it really stuck
    $toggles2 = Get-PanelToggles
    $final = "?"
    if ($toggles2.Count -gt 0) {
        $final = $toggles2[-1].El.GetCurrentPattern(
            [System.Windows.Automation.TogglePattern]::Pattern).Current.ToggleState
    }
    $results += [pscustomobject]@{ Stem = $name; Panels = $toggles.Count; X = $midi.X; Was = $state; Now = $final }
    Write-Output ("  {0,-8} panels={1} midiX={2}  {3} -> {4}" -f $name, $toggles.Count, $midi.X, $state, $final)
}

Write-Output ""
$bad = $results | Where-Object { $_.Now -ne "On" }
if ($bad) { Write-Output "STILL OFF: $(($bad | ForEach-Object { $_.Stem }) -join ', ')" }
else { Write-Output "MIDI OUTPUT enabled on all $($results.Count) stems" }
