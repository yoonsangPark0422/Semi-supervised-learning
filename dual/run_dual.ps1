param(
    [int]$Seed = 5,
    [string]$Out = "results\dual_cifar10_imb100"
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\FORYOUCOM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

& $Python "train.py" `
    --dataset cifar10 `
    --num-labeled 4000 `
    --arch wideresnet `
    --batch-size 64 `
    --lr 0.03 `
    --expand-labels `
    --imb-ratio 100 `
    --seed $Seed `
    --out $Out
