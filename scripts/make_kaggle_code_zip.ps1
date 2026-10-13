# Creates parking-detection-code.zip for uploading as a Kaggle Dataset.
# Uses forward-slash paths inside the zip (required by Kaggle; Windows \ is rejected).
#
# Run from PowerShell:
#   cd "D:\RMS 4\parking-detection"
#   powershell -ExecutionPolicy Bypass -File scripts\make_kaggle_code_zip.ps1
# Or:
#   python scripts\make_kaggle_code_zip.py

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
python (Join-Path $PSScriptRoot "make_kaggle_code_zip.py")
if ($LASTEXITCODE -ne 0) { throw "Zip creation failed" }
Write-Host "Upload this zip as a Kaggle Dataset named e.g. parking-detection-code"
Write-Host "Also upload your PKLot folder as a separate dataset (train/valid/test + data.yaml)."
