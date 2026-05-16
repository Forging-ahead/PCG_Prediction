param(
    [string]$CondaEnv = "",
    [string]$HostName = "",
    [int]$Port = 0,
    [string]$Config = "web_frontend_config.json"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$argsList = @("web_frontend.py", "--config", $Config)

if ($CondaEnv.Trim().Length -gt 0) {
    $argsList += @("--conda-env", $CondaEnv.Trim())
}

if ($HostName.Trim().Length -gt 0) {
    $argsList += @("--host", $HostName.Trim())
}

if ($Port -gt 0) {
    $argsList += @("--port", [string]$Port)
}

python @argsList
