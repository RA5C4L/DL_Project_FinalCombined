$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $pythonExe)) { throw 'Create .venv and install requirements first; see README.md.' }
$env:HF_HOME = Join-Path $PSScriptRoot '.cache/huggingface'
$env:NLTK_DATA = Join-Path $PSScriptRoot '.cache/nltk_data'
function Run-Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]] $Arguments)
    & $pythonExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python stage failed: $Arguments" }
}
Run-Python @('-m', 'pytest', '-q')
Run-Python @('-m', 'nltk.downloader', '-d', $env:NLTK_DATA, 'wordnet', 'omw-1.4')
if (!(Test-Path 'data/prepared')) { Run-Python @('-m', 'navneet_dl.prepare') }
if (!(Test-Path 'models/distilbert/experiment.json')) { Run-Python @('-m', 'navneet_dl.train') }
Run-Python @('-m', 'navneet_dl.data_stats')
if (!(Test-Path 'data/perturbed/manifest.json')) { Run-Python @('-m', 'navneet_dl.perturb') }
Run-Python @('-m', 'navneet_dl.verify')
Run-Python @('-m', 'navneet_dl.evaluate')
Run-Python @('-m', 'navneet_dl.report')
