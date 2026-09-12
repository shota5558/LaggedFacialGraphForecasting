[CmdletBinding()]
param(
    [ValidateSet("core", "sensitivity", "all")]
    [string]$Suite = "core",

    [switch]$RecreateVenv,
    [switch]$SkipInstall,
    [switch]$KeepGoing
)

$ErrorActionPreference = "Stop"
$Runner = Join-Path $PSScriptRoot "run_local_ci.py"
$ArgsList = @($Runner, "--suite", $Suite)

if ($RecreateVenv) {
    $ArgsList += "--recreate-venv"
}
if ($SkipInstall) {
    $ArgsList += "--skip-install"
}
if ($KeepGoing) {
    $ArgsList += "--keep-going"
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    & py -3.11 @ArgsList
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    & python @ArgsList
} else {
    throw "Python was not found. Install Python 3.11 or make it available as 'py -3.11' / 'python'."
}

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
