param(
    [int[]]$ImbRatios = @(50, 100),
    [int[]]$Seeds = @(1, 2, 3),
    [string[]]$ExchangeModes = @("none", "major_to_minor", "minor_to_major", "bidirectional"),
    [int]$Epochs = 50,
    [int]$EvalStep = 512,
    [int]$BatchSize = 32,
    [int]$Mu = 7,
    [double]$EnsembleMinorWeight = 0.8,
    [string]$OutRoot = "results\exchange_direction_ablation_e50_s512_seed1_2_3"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
$totalSteps = $Epochs * $EvalStep

foreach ($ratio in $ImbRatios) {
    foreach ($seed in $Seeds) {
        foreach ($mode in $ExchangeModes) {
            $out = Join-Path $OutRoot ("imb{0}\seed_{1}\{2}" -f $ratio, $seed, $mode)
            New-Item -ItemType Directory -Force -Path $out | Out-Null
            python .\train_exchange_direction_ablation.py `
                --dataset cifar10 `
                --num-labeled 4000 `
                --imb-ratio $ratio `
                --total-steps $totalSteps `
                --eval-step $EvalStep `
                --batch-size $BatchSize `
                --mu $Mu `
                --seed $seed `
                --exchange-mode $mode `
                --ensemble-minor-weight $EnsembleMinorWeight `
                --out $out `
                --num-workers 0 `
                --no-progress `
                --no-save-checkpoint
        }
    }
}
