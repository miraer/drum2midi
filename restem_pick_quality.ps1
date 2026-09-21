# Changes ReStem's quality mode with the keyboard, verifying the label after every step.
#
# DO NOT SPEND MORE TIME ON THIS. The mode cannot be set from a script on this machine,
# and that is now established four independent ways rather than assumed from one failure:
#
#   1. the environment will not give ReStem the foreground. SetForegroundWindow reports
#      success and GetForegroundWindow still returns another window, so SendKeys goes
#      somewhere else and this script refuses to press anything -- which is the only
#      reason it has never typed into an unrelated application.
#   2. the list items are drawn, not controls. With the list open, a full UI Automation
#      sweep of the window finds nothing inside it to invoke.
#   3. the mode label is a Text element carrying a ValuePattern, which looks promising
#      until you read it: IsReadOnly is True, so SetValue is refused. That is fortunate
#      as well as final -- a writable label would have let a script change the text
#      without changing the mode, and every guard we have reads that text to confirm
#      which arm a render belongs to.
#   4. the dropdown is not a separate top-level window either. Invoking the selector
#      adds no window to the desktop root, so there is no popup to search.
#
# So a human sets it -- ONCE. The mode is held in memory and written to ReStem's own
# settings file on a clean exit, so it survives restarts as long as the application is
# asked to close rather than killed. Closing it with Stop-Process -Force throws the
# setting away and the next launch comes back in whatever was last saved, which is how
# an afternoon was spent believing the mode "did not stick".
# Close with CloseMainWindow(); it exits in about two seconds and writes the file.
#
# What remains is a human clicking it. The launcher and the batch queue both read the
# label and refuse to start when it does not match what was asked for, so a wrong mode
# cannot silently produce renders filed under the wrong arm.
#
# Kept rather than deleted because the keyboard approach is sound and would work in a
# session that can focus the window -- and because a script that records why it cannot
# work is worth more than its absence, which invites someone to write it again.
#
# The selector is drawn: UI Automation exposes its current label and nothing else, so the
# list items cannot be found and clicked. restem_set_quality.ps1 tried the Expand pattern
# and stranded the application; restem_click_quality.ps1 opens the list but finds nothing
# inside it to click.
#
# Coordinates were the obvious next idea and are rejected: guessing where an item is
# painted means a mis-click lands on whatever is behind the list, and the settings this
# window carries are not ones to change blind.
#
# The keyboard is bounded in a way the mouse is not. With the list open, Up and Down can
# only move within it, Enter can only choose one of its items, and Escape closes it. The
# result is read back from the label after every keystroke, so the script knows what it
# did rather than assuming.
#
#   powershell -File restem_pick_quality.ps1 -Want "Better (Offline)"
#   powershell -File restem_pick_quality.ps1 -Want "Best (Offline) +"

param(
    [Parameter(Mandatory = $true)][string] $Want,
    [int] $MaxSteps = 6
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
Add-Type -AssemblyName System.Windows.Forms

Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Fg {
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int n);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool c);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, IntPtr p);
  [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
}
"@ -ErrorAction SilentlyContinue

function Win {
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, 'ReStem 2')
    $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $c)
}

function Mode {
    try {
        $w = Win
        if (-not $w) { return $null }
        $all = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                          [System.Windows.Automation.Condition]::TrueCondition)
        ($all | Where-Object { $_.Current.Name -match '\((Offline|Realtime)\)' } |
           Select-Object -First 1).Current.Name
    } catch { $null }
}

function Selector {
    $w = Win
    if (-not $w) { return $null }
    $all = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                      [System.Windows.Automation.Condition]::TrueCondition)
    $all | Where-Object {
        $_.Current.ControlType.ProgrammaticName -eq 'ControlType.Button' -and
        $_.Current.Name -ceq 'Model Quality Selector'
    } | Select-Object -First 1
}

function Focus {
    $p = Get-Process "ReStem 2" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $p) { return $false }
    $h = $p.MainWindowHandle
    [Fg]::ShowWindow($h, 9) | Out-Null
    $fg = [Fg]::GetForegroundWindow()
    $t1 = [Fg]::GetWindowThreadProcessId($fg, [IntPtr]::Zero)
    $t2 = [Fg]::GetCurrentThreadId()
    [Fg]::AttachThreadInput($t2, $t1, $true) | Out-Null
    [Fg]::SetForegroundWindow($h) | Out-Null
    [Fg]::AttachThreadInput($t2, $t1, $false) | Out-Null
    Start-Sleep -Milliseconds 400
    # SetForegroundWindow returns false in plenty of cases where it worked, so the
    # return value is not the test. What the window actually is, is.
    return ([Fg]::GetForegroundWindow() -eq $h)
}

$start = Mode
Write-Output "mode now: [$start]"
if (-not $start) { Write-Output "cannot read the mode; refusing to press keys blind"; exit 1 }
if ($start -ceq $Want) { Write-Output "already there"; exit 0 }

if (-not (Focus)) { Write-Output "could not bring ReStem to the foreground"; exit 1 }
Start-Sleep -Milliseconds 400

$sel = Selector
if (-not $sel) { Write-Output "quality selector not found"; exit 1 }
$sel.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
Start-Sleep -Milliseconds 800

# Up first: the list is ordered Better, Best, Best+, so from Best+ the target is above.
foreach ($key in @('{UP}', '{UP}', '{DOWN}', '{DOWN}')) {
    for ($i = 0; $i -lt $MaxSteps; $i++) {
        [System.Windows.Forms.SendKeys]::SendWait($key)
        Start-Sleep -Milliseconds 350
        $now = Mode
        Write-Output "  after $key -> [$now]"
        if ($now -ceq $Want) {
            [System.Windows.Forms.SendKeys]::SendWait('{ENTER}')
            Start-Sleep -Milliseconds 600
            $final = Mode
            Write-Output "committed: [$final]"
            if ($final -ceq $Want) { exit 0 }
            Write-Output "the label changed back after Enter; not what was asked for"
            exit 1
        }
        if (-not $now) { Write-Output "  label unreadable; stopping"; break }
    }
}

[System.Windows.Forms.SendKeys]::SendWait('{ESC}')
Start-Sleep -Milliseconds 400
Write-Output "could not reach '$Want'; left at [$(Mode)] with the list closed"
exit 1
