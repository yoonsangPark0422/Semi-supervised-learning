param(
    [int[]]$Ratios = @(50, 100),
    [int]$Seed = 5,
    [int]$Epochs = 50,
    [int]$EvalStep = 512,
    [int]$BatchSize = 32,
    [int]$Mu = 7,
    [string]$ResultRoot = "results\fixmatch_vs_dual_imb50_100_e50_s512_seed5"
)

$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$BaselineRoot = Join-Path $RepoRoot "baseline compare"
$DualRoot = Join-Path $RepoRoot "dual"
if ($env:PYTHON) {
    $Python = $env:PYTHON
} else {
    $Python = "python"
}

$ResultRootAbs = Join-Path $RepoRoot $ResultRoot
New-Item -ItemType Directory -Force -Path $ResultRootAbs | Out-Null
$SummaryPath = Join-Path $ResultRootAbs "summary.csv"
$RunnerLog = Join-Path $ResultRootAbs "runner.log"
$TotalSteps = $Epochs * $EvalStep

"started=$(Get-Date -Format s) seed=$Seed epochs=$Epochs eval_step=$EvalStep total_steps=$TotalSteps batch_size=$BatchSize mu=$Mu" | Set-Content -LiteralPath $RunnerLog
"ratio,method,out,status,exit_code,last_epoch,best_acc,final_acc,macro_f1,macro_f1_final,started_at,finished_at" | Set-Content -LiteralPath $SummaryPath

function Add-SummaryRow {
    param(
        [int]$Ratio,
        [string]$Method,
        [string]$Out,
        [string]$Status,
        [int]$ExitCode,
        [string]$StartedAt,
        [string]$FinishedAt
    )
    $metricsPath = Join-Path $Out "train_metrics.csv"
    $lastEpoch = ""
    $bestAcc = ""
    $finalAcc = ""
    $macroF1 = ""
    $macroF1Final = ""
    if (Test-Path -LiteralPath $metricsPath) {
        $rows = @(Import-Csv -LiteralPath $metricsPath)
        if ($rows.Count -gt 0) {
            $last = $rows[-1]
            $lastEpoch = $last.epoch
            if ($Method -eq "baseline") {
                $bestAcc = $last.test_acc
                $finalAcc = $last.test_acc
                $macroF1 = $last.macro_f1
            } else {
                $bestAcc = $last.test_acc_final
                $finalAcc = $last.test_acc_final
                $macroF1 = $last.macro_f1_ensemble
                $macroF1Final = $last.macro_f1_final
            }
        }
    }
    $line = '{0},{1},"{2}",{3},{4},{5},{6},{7},{8},{9},{10},{11}' -f $Ratio,$Method,$Out,$Status,$ExitCode,$lastEpoch,$bestAcc,$finalAcc,$macroF1,$macroF1Final,$StartedAt,$FinishedAt
    Add-Content -LiteralPath $SummaryPath -Value $line
}

function Invoke-Run {
    param(
        [int]$Ratio,
        [string]$Method,
        [string]$WorkDir,
        [string[]]$ArgsList,
        [string]$Out
    )
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    $stdout = Join-Path $Out "stdout.log"
    $stderr = Join-Path $Out "stderr.log"
    $startedAt = Get-Date -Format s
    "[$startedAt] START ratio=$Ratio method=$Method out=$Out" | Tee-Object -FilePath $RunnerLog -Append
    $proc = Start-Process -FilePath $Python -ArgumentList $ArgsList -WorkingDirectory $WorkDir -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $finishedAt = Get-Date -Format s
    if ($proc.ExitCode -eq 0) {
        "[$finishedAt] DONE ratio=$Ratio method=$Method" | Tee-Object -FilePath $RunnerLog -Append
        Add-SummaryRow -Ratio $Ratio -Method $Method -Out $Out -Status "done" -ExitCode $proc.ExitCode -StartedAt $startedAt -FinishedAt $finishedAt
    } else {
        "[$finishedAt] FAIL ratio=$Ratio method=$Method exit=$($proc.ExitCode)" | Tee-Object -FilePath $RunnerLog -Append
        Add-SummaryRow -Ratio $Ratio -Method $Method -Out $Out -Status "failed" -ExitCode $proc.ExitCode -StartedAt $startedAt -FinishedAt $finishedAt
        throw "Training failed: ratio=$Ratio method=$Method exit=$($proc.ExitCode). See $stderr"
    }
}

foreach ($Ratio in $Ratios) {
    $BaseOut = Join-Path $ResultRootAbs ("imb$Ratio\baseline_fixmatch")
    $BaseArgs = @(
        "train.py",
        "--dataset", "cifar10",
        "--num-labeled", "4000",
        "--arch", "wideresnet",
        "--batch-size", "$BatchSize",
        "--mu", "$Mu",
        "--lr", "0.03",
        "--expand-labels",
        "--imb-ratio", "$Ratio",
        "--total-steps", "$TotalSteps",
        "--eval-step", "$EvalStep",
        "--ablation-preset", "fixmatch",
        "--seed", "$Seed",
        "--out", $BaseOut,
        "--num-workers", "0",
        "--no-progress",
        "--no-save-checkpoint"
    )
    Invoke-Run -Ratio $Ratio -Method "baseline" -WorkDir $BaselineRoot -ArgsList $BaseArgs -Out $BaseOut

    $DualOut = Join-Path $ResultRootAbs ("imb$Ratio\dual_sampler_weighted_ce")
    $DualArgs = @(
        "train.py",
        "--dataset", "cifar10",
        "--num-labeled", "4000",
        "--arch", "wideresnet",
        "--batch-size", "$BatchSize",
        "--mu", "$Mu",
        "--lr", "0.03",
        "--expand-labels",
        "--imb-ratio", "$Ratio",
        "--total-steps", "$TotalSteps",
        "--eval-step", "$EvalStep",
        "--dual-train-mode", "dual_sampler_weighted_ce",
        "--seed", "$Seed",
        "--out", $DualOut,
        "--num-workers", "0",
        "--no-progress",
        "--no-save-checkpoint"
    )
    Invoke-Run -Ratio $Ratio -Method "dual" -WorkDir $DualRoot -ArgsList $DualArgs -Out $DualOut
}

"finished=$(Get-Date -Format s)" | Tee-Object -FilePath $RunnerLog -Append

