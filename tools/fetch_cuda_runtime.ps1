<#
.SYNOPSIS
  Fetches the CUDA 12.9 + cuDNN 9 runtime DLLs required by the ONNX Runtime CUDA execution
  provider, without installing the CUDA toolkit and without admin rights.

.DESCRIPTION
  ONNX Runtime 1.31 (CUDA 12 build) needs CUDA >= 12.8 runtime libraries and cuDNN 9.
  NVIDIA publishes those DLLs as Python wheels on PyPI (redistributable). This script downloads
  the wheels with pip (any Python 3 works, it never installs them) and extracts only the DLLs
  into third_party/cuda-runtime/bin. The application prepends that folder to PATH at startup.

  A driver >= 525 is enough thanks to CUDA minor-version compatibility (tested on 538.08 /
  CUDA 12.2 with CUDA_MODULE_LOADING=LAZY, which the application sets automatically).

.PARAMETER Python
  Python interpreter used to run "pip download". Defaults to the PoC venv, then "python".
#>
param(
    [string]$Python = "C:\pro\tracking\.venv\Scripts\python.exe"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$wheels = Join-Path $root "third_party\wheels"
$bin = Join-Path $root "third_party\cuda-runtime\bin"
New-Item -ItemType Directory -Force $wheels, $bin | Out-Null
if (-not (Test-Path $Python)) { $Python = "python" }

$packages = @(
    "nvidia-cuda-runtime-cu12==12.9.*",
    "nvidia-cublas-cu12==12.9.*",
    "nvidia-cufft-cu12",
    "nvidia-curand-cu12",
    "nvidia-cuda-nvrtc-cu12==12.9.*",
    "nvidia-nvjitlink-cu12==12.9.*",
    "nvidia-cudnn-cu12==9.*"
)
& $Python -m pip download --no-deps --only-binary=:all: --platform win_amd64 --python-version 3.12 -d $wheels @packages
if ($LASTEXITCODE -ne 0) { throw "pip download failed" }

Add-Type -AssemblyName System.IO.Compression.FileSystem
foreach ($whl in Get-ChildItem $wheels -Filter *.whl) {
    $zip = [System.IO.Compression.ZipFile]::OpenRead($whl.FullName)
    try {
        foreach ($entry in $zip.Entries | Where-Object { $_.FullName -like "*.dll" }) {
            $dest = Join-Path $bin $entry.Name
            [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $dest, $true)
        }
    } finally { $zip.Dispose() }
}
Get-ChildItem $bin -Filter *.dll | Sort-Object Name | ForEach-Object { "{0,-50} {1,10:N0} KB" -f $_.Name, ($_.Length / 1KB) }
Write-Host "CUDA runtime ready in $bin"
