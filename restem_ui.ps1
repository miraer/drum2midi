Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class W32 {
  [DllImport("user32.dll", CharSet=CharSet.Unicode)]
  public static extern int SendMessage(IntPtr h, int msg, IntPtr w, string l);
  [DllImport("user32.dll")]
  public static extern int PostMessage(IntPtr h, int msg, IntPtr w, IntPtr l);
  [DllImport("user32.dll")]
  public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")]
  public static extern IntPtr GetForegroundWindow();
}
"@ -ErrorAction SilentlyContinue

$WM_SETTEXT = 0x000C
$BM_CLICK   = 0x00F5

function Get-RestemWindow {
    $p = Get-Process "ReStem 2" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $p) { return $null }
    $rootEl = [System.Windows.Automation.AutomationElement]::RootElement
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::ProcessIdProperty, $p.Id)
    $rootEl.FindFirst([System.Windows.Automation.TreeScope]::Children, $c)
}

function Find-ByName($scope, $name) {
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, $name)
    $scope.FindFirst([System.Windows.Automation.TreeScope]::Descendants, $c)
}

function Invoke-Btn($el) {
    $el.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
}

function Find-ByAutoId($scope, $id, $className) {
    $all = $scope.FindAll([System.Windows.Automation.TreeScope]::Descendants,
        [System.Windows.Automation.Condition]::TrueCondition)
    foreach ($e in $all) {
        if ($e.Current.AutomationId -eq $id) {
            if (-not $className -or $e.Current.ClassName -eq $className) { return $e }
        }
    }
    $null
}

# Fills the open file dialog with a path and confirms it.
function Submit-FileDialog($path, $timeout = 20) {
    $deadline = (Get-Date).AddSeconds($timeout)
    while ((Get-Date) -lt $deadline) {
        $win = Get-RestemWindow
        if ($win) {
            $edit = Find-ByAutoId $win "1148" "Edit"
            $open = Find-ByAutoId $win "1" "Button"
            if ($edit -and $open) {
                $eh = [IntPtr]$edit.Current.NativeWindowHandle
                $oh = [IntPtr]$open.Current.NativeWindowHandle
                if ($eh -ne [IntPtr]::Zero -and $oh -ne [IntPtr]::Zero) {
                    [W32]::SendMessage($eh, $WM_SETTEXT, [IntPtr]::Zero, $path) | Out-Null
                    Start-Sleep -Milliseconds 400
                    [W32]::PostMessage($oh, $BM_CLICK, [IntPtr]::Zero, [IntPtr]::Zero) | Out-Null
                    return $true
                }
            }
        }
        Start-Sleep -Milliseconds 500
    }
    $false
}

# Waits for the dialog to close
function Wait-DialogGone($timeout = 30) {
    $deadline = (Get-Date).AddSeconds($timeout)
    while ((Get-Date) -lt $deadline) {
        $win = Get-RestemWindow
        if (-not (Find-ByAutoId $win "1148" "Edit")) { return $true }
        Start-Sleep -Milliseconds 500
    }
    $false
}
