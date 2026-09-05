#Requires -Version 5.0
<#
.SYNOPSIS
    启动音频分离界面（separator_webui.py）。
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$FfmpegBin = 'E:\Programs\ffmpeg-master-latest-win64-gpl\bin'
if (Test-Path $FfmpegBin) {
    $env:Path = "$FfmpegBin;$env:Path"
}

$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    Write-Host '=== GPU 状态 ===' -ForegroundColor Cyan
    & nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
}

$Python = 'e:\Pro2\audio-demucs\python-audio-separator\.venv\Scripts\python.exe'
$Script = Join-Path $PSScriptRoot 'separator_webui.py'

if (-not (Test-Path $Python)) {
    throw "未找到 Python 环境: $Python 。请确认 audio-separator 项目路径是否正确。"
}
if (-not (Test-Path $Script)) {
    throw "未找到启动脚本: $Script"
}

Write-Host '============================================================'
Write-Host '   音频分离界面 - 一键启动'
Write-Host '   基于 python-audio-separator / Demucs'
Write-Host '============================================================'
Write-Host ''
Write-Host '正在启动，初次加载模型可能需要几十秒，请稍候...'
Write-Host '优先 7860，占用则自动顺延；浏览器会打开实际地址。'
Write-Host ''

& $Python $Script
if ($LASTEXITCODE -ne 0) {
    throw "separator_webui.py 退出码: $LASTEXITCODE"
}