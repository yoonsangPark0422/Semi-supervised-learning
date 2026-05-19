$ErrorActionPreference = 'Continue'

$Root = $PSScriptRoot
$Python = 'C:\Users\FORYOUCOM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$Script = Join-Path $Root 'compare_baselines.py'
$Out = Join-Path $Root 'result'
$Stdout = Join-Path $Out 'runner.stdout.log'
$Stderr = Join-Path $Out 'runner.stderr.log'

New-Item -ItemType Directory -Force -Path $Out | Out-Null
Set-Location -LiteralPath $Root

& $Python $Script `
    --out $Out `
    --methods 'crest,daso,abc,proposed' `
    --epochs 100 `
    --eval-step 1024 `
    --batch-size 32 `
    --mu 7 `
    --num-workers 0 `
    *> $Stdout
