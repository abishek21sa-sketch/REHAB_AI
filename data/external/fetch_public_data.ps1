$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$url = "https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
Invoke-WebRequest -Uri $url -OutFile (Join-Path $root "uci_postural_transitions.zip")
Write-Host "EXTERNAL_DATA_REFRESH=PASS"
