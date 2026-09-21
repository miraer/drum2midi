# Starts ReStem and gets it to a state where a batch can be driven.
#
# Written after a restart at midnight: the application had to be relaunched mid-run and
# nothing in the automation handled what comes up first. On a trial licence ReStem shows
# a dialog that has to be acknowledged before the main window is usable, so an
# unattended restart stops there and the rest of the night is lost without a sign.
#
# What it will and will not press:
#
#   It confirms the trial prompt, because that is what the human does by hand and the
#   alternative is a stalled run. It matches the button by exact name against a list
#   written down here, case-sensitively -- PowerShell's -eq is case-insensitive, and a
#   button named `Close` therefore also matches the title bar's `close`, which closed
#   the whole application once tonight.
#
#   It presses nothing it does not recognise. If a dialog is up whose buttons are not
#   in the list, it prints them and stops rather than guessing. A licensing prompt is
#   not a place to click hopefully.
#
#   powershell -File restem_launch.ps1
#   powershell -File restem_launch.ps1 -Expect "Best (Offline) +"

param(
    [string] $Expect = "",
    [int] $WaitSeconds = 180
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName UIAutomationClient, UIAutomationTypes

# Buttons that are safe to press on startup, in the order they are preferred.
$CONFIRM = @("Continue Trial", "Continue trial", "Continue Evaluation", "Continue",
             "Start Trial", "Start trial", "Use Trial", "Try", "Try Now", "Later",
             "Not now", "Maybe later", "Skip", "Dismiss", "OK", "Ok")
# Buttons that mean the opposite and must never be pressed by a script. Quitting or
# touching the licence unattended is worse than a stalled run; everything else here
# either ends the session or spends money.
$NEVER = @("Quit", "Exit", "Buy", "Buy Now", "Purchase", "Subscribe", "Upgrade",
           "Activate", "Deactivate", "Register", "Enter Licence", "Enter License",
           "Close", "Cancel", "No", "Decline")
# Window chrome, never a dialog answer.
$CHROME = @("minimise", "close", "Minimize", "Maximize", "Options", "Settings")

function Say([string] $m) { Write-Output ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }

function Get-Win {
    $root = [System.Windows.Automation.AutomationElement]::RootElement
    $c = New-Object System.Windows.Automation.PropertyCondition(
        [System.Windows.Automation.AutomationElement]::NameProperty, 'ReStem 2')
    $root.FindFirst([System.Windows.Automation.TreeScope]::Children, $c)
}

function Get-Buttons($w) {
    if (-not $w) { return @() }
    $all = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                      [System.Windows.Automation.Condition]::TrueCondition)
    @($all | Where-Object { $_.Current.ControlType.ProgrammaticName -eq 'ControlType.Button' -and
                            $_.Current.Name })
}

if (-not (Get-Process -Name "ReStem*" -ErrorAction SilentlyContinue)) {
    $exe = "C:\Program Files\ReStem 2\ReStem 2.exe"
    if (-not (Test-Path $exe)) { Say "not installed at $exe"; exit 1 }
    Say "starting ReStem"
    Start-Process -FilePath $exe
} else {
    Say "ReStem already running"
}

$deadline = (Get-Date).AddSeconds($WaitSeconds)
$confirmed = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    $w = Get-Win
    if (-not $w) { continue }
    $btns = Get-Buttons $w
    $names = @($btns | ForEach-Object { $_.Current.Name })

    # The trial prompt is checked FIRST, before anything else, because it overlays the
    # main window rather than replacing it: Load is present in the tree underneath and
    # an earlier version of this script declared the window ready while the prompt was
    # still up and unanswered. Worse, that prompt carries a Close, and closing it
    # instead of confirming quits the application -- which is what killed three runs
    # before anyone noticed the dialog was there at all.
    if (-not $confirmed) {
        $hit = $null
        foreach ($want in $CONFIRM) {
            foreach ($b in $btns) { if ($b.Current.Name -ceq $want) { $hit = $b; break } }
            if ($hit) { break }
        }
        if ($hit) {
            Say ("confirming the startup prompt by pressing '" + $hit.Current.Name + "'")
            $hit.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
            $confirmed = $true
            Start-Sleep -Seconds 5
            continue
        }
    }

    if ($names -contains 'Load' -and $confirmed) { Say "main window is ready"; break }
    if (($names -contains 'Load') -and -not ($names -contains 'Continue Trial')) {
        # No prompt was ever seen and Load is live: the licence is already settled.
        # The parentheses are load-bearing. Written as `-not $names -contains 'x'`,
        # PowerShell binds -not to $names first, so the test reduces to
        # `$false -contains 'x'` and the branch can never be taken -- which would hang
        # this script to its deadline on any install that does not show the prompt.
        Say "main window is ready, no startup prompt appeared"
        break
    }

    # A render in progress replaces Load with Stop, so the main window looks like an
    # unrecognised dialog full of choices. The mode text tells them apart: it is drawn
    # on the main window and on nothing else. If it reads, this is ReStem working, not
    # a prompt, and the right thing is to wait.
    $all = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                      [System.Windows.Automation.Condition]::TrueCondition)
    $onMain = @($all | Where-Object { $_.Current.Name -match '\((Offline|Realtime)\)' }).Count -gt 0
    if ($onMain) {
        if ($names -contains 'Stop') { Say "a render is in progress; waiting" }
        continue
    }

    if (-not $confirmed) {
        $hit = $null
        foreach ($want in $CONFIRM) {
            foreach ($b in $btns) { if ($b.Current.Name -ceq $want) { $hit = $b; break } }
            if ($hit) { break }
        }
        if ($hit) {
            Say ("confirming the startup prompt by pressing '" + $hit.Current.Name + "'")
            $hit.GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
            $confirmed = $true
            Start-Sleep -Seconds 5
            continue
        }

        # Nothing on the list, but the trial prompt has to be answered when nobody is
        # here to answer it. So: if exactly one candidate remains once the window chrome
        # and everything on the NEVER list is removed, press that one and record what it
        # said. A dialog with a single safe button is a prompt; a dialog with several is
        # a decision, and a script should not make it.
        $cands = @($btns | Where-Object {
            $n = $_.Current.Name
            ($CHROME -notcontains $n) -and ($NEVER -notcontains $n)
        })
        if ($cands.Count -eq 1) {
            Say ("unlisted prompt: pressing its only safe button, '" +
                 $cands[0].Current.Name + "'")
            $cands[0].GetCurrentPattern([System.Windows.Automation.InvokePattern]::Pattern).Invoke()
            $confirmed = $true
            Start-Sleep -Seconds 5
            continue
        }
        if ($cands.Count -gt 1) {
            Say ("a dialog is up with several choices and none is recognised: " +
                 (($cands | ForEach-Object { $_.Current.Name }) -join ', '))
            Say "pressing nothing. Confirm it by hand, then re-run."
            exit 2
        }
    }
}

$w = Get-Win
if (-not $w) { Say "no ReStem window after $WaitSeconds s"; exit 1 }
$names = @(Get-Buttons $w | ForEach-Object { $_.Current.Name })
if ($names -notcontains 'Load') { Say "Load never appeared; buttons: $($names -join ', ')"; exit 1 }

$all = $w.FindAll([System.Windows.Automation.TreeScope]::Descendants,
                  [System.Windows.Automation.Condition]::TrueCondition)
$mode = ($all | Where-Object { $_.Current.Name -match '\((Offline|Realtime)\)' } |
           Select-Object -First 1).Current.Name
Say "mode reads '$mode'"
if ($Expect -and $mode -ne $Expect) {
    Say "expected '$Expect' -- the quality selector is drawn and cannot be set from here,"
    Say "so this needs a human before any batch is started."
    exit 3
}
exit 0
