$ErrorActionPreference = 'Stop'

$Python = 'C:\Users\FORYOUCOM\Desktop\DeCon\.venv-cu128\Scripts\python.exe'
$DeConRoot = 'C:\Users\FORYOUCOM\Desktop\DeCon'
$ResultRoot = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\results\official_test_decon_dual_matched_cifar10_svhn_e50_s512_seed1_2_3'
$LogDir = Join-Path $ResultRoot 'logs'
$RunnerLog = Join-Path $LogDir 'runner.log'

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-RunnerLog([string]$Message) {
    Add-Content -LiteralPath $RunnerLog -Value "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
}

function Invoke-MatchedRun([string]$Dataset, [int]$Seed) {
    $Out = Join-Path $ResultRoot "$Dataset\decon\seed_$Seed"
    $PerClass = Join-Path $Out 'per_class_eval.csv'
    $TrainLog = Join-Path $Out 'log.txt'
    if ((Test-Path -LiteralPath $PerClass) -and (Test-Path -LiteralPath $TrainLog)) {
        Write-RunnerLog "skip completed dataset=$Dataset seed=$Seed out=$Out"
        return
    }

    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    $Stdout = Join-Path $LogDir "$Dataset`_decon_dual_matched_seed$Seed.stdout.log"
    $Stderr = Join-Path $LogDir "$Dataset`_decon_dual_matched_seed$Seed.stderr.log"
    if ($Dataset -eq 'cifar10') {
        $BatchSize = '32'
        $DatasetSpecific = @('--expand-labels')
    } else {
        $BatchSize = '64'
        $DatasetSpecific = @()
    }

    $ArgsList = @(
        '.\train_dual_matched.py',
        '--dataset', $Dataset,
        '--num-labeled', '4000',
        '--num-max', '1614',
        '--num-max-u', '4000',
        '--imb-ratio-label', '100',
        '--imb-ratio-unlabel', '100',
        '--arch', 'wideresnet',
        '--batch-size', $BatchSize,
        '--mu', '7',
        '--lr', '0.03',
        '--total-steps', '25600',
        '--eval-step', '512',
        '--tau', '1.0',
        '--ema-u', '0.99',
        '--tau3', '2.0',
        '--tau4', '1.0',
        '--val-samples-per-class', '0',
        '--seed', "$Seed",
        '--out', $Out,
        '--num-workers', '0',
        '--no-progress',
        '--no-save-checkpoint',
        '--eval-final-only',
        '--match-dual-data'
    ) + $DatasetSpecific

    Write-RunnerLog "start dataset=$Dataset seed=$Seed out=$Out"
    $Process = Start-Process -FilePath $Python -ArgumentList $ArgsList `
        -WorkingDirectory $DeConRoot -WindowStyle Hidden -Wait -PassThru `
        -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
    if ($Process.ExitCode -ne 0) {
        Write-RunnerLog "failed dataset=$Dataset seed=$Seed exit=$($Process.ExitCode) stderr=$Stderr"
        throw "DeCon Dual-matched run failed: dataset=$Dataset seed=$Seed exit=$($Process.ExitCode)"
    }
    Write-RunnerLog "done dataset=$Dataset seed=$Seed out=$Out"
}

Write-RunnerLog 'Dual-matched DeCon official-test queue started'
foreach ($Dataset in @('cifar10', 'svhn')) {
    foreach ($Seed in @(1, 2, 3)) {
        Invoke-MatchedRun -Dataset $Dataset -Seed $Seed
    }
}
Write-RunnerLog 'Dual-matched DeCon official-test queue finished'
