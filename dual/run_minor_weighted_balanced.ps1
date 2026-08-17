param(
    [int]$ImbRatio = 100,
    [int]$Seed = 1,
    [int]$Epochs = 50,
    [int]$EvalStep = 512,
    [int]$BatchSize = 32,
    [int]$Mu = 7,
    [double]$EnsembleMinorWeight = 0.8,
    [string]$OutRoot = "results\minor_weighted_balanced"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$totalSteps = $Epochs * $EvalStep
$out = Join-Path $OutRoot ("imb{0}_seed{1}_e{2}_s{3}" -f $ImbRatio, $Seed, $Epochs, $EvalStep)
New-Item -ItemType Directory -Force -Path $out | Out-Null

python .\train_minor_weighted_balanced.py `
    --dataset cifar10 `
    --num-labeled 4000 `
    --imb-ratio $ImbRatio `
    --total-steps $totalSteps `
    --eval-step $EvalStep `
    --batch-size $BatchSize `
    --mu $Mu `
    --seed $Seed `
    --ensemble-minor-weight $EnsembleMinorWeight `
    --out $out
