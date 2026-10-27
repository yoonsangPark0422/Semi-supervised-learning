param(
    [int[]]$Ratios = @(50, 100),
    [double[]]$Weights = @(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    [int]$Seed = 5,
    [int]$Epochs = 50,
    [int]$EvalStep = 512,
    [int]$BatchSize = 32,
    [int]$Mu = 7,
    [string]$ResultRoot = "results\ensemble_weight_sweep_imb50_100_e50_s512_seed5"
)

$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$DualRoot = Join-Path $RepoRoot "dual"
if ($env:PYTHON) {
    $Python = $env:PYTHON
} else {
    $RuntimePython = "C:\Users\FORYOUCOM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $RuntimePython) {
        $Python = $RuntimePython
    } else {
        $Python = "python"
    }
}

$ResultRootAbs = Join-Path $RepoRoot $ResultRoot
New-Item -ItemType Directory -Force -Path $ResultRootAbs | Out-Null
$RunnerLog = Join-Path $ResultRootAbs "runner.log"
$SummaryPath = Join-Path $ResultRootAbs "ensemble_weight_summary.csv"
$TotalSteps = $Epochs * $EvalStep

"started=$(Get-Date -Format s) seed=$Seed epochs=$Epochs eval_step=$EvalStep total_steps=$TotalSteps batch_size=$BatchSize mu=$Mu" | Set-Content -LiteralPath $RunnerLog
"ratio,weight,acc,macro_f1,minority_5_9_recall,minority_5_9_precision,minority_5_9_f1,tail_7_9_recall,tail_7_9_precision,tail_7_9_f1,eval_out" | Set-Content -LiteralPath $SummaryPath

function Invoke-CommandRun {
    param(
        [string]$Name,
        [string]$WorkDir,
        [string[]]$ArgsList,
        [string]$Out
    )
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    $stdout = Join-Path $Out "stdout.log"
    $stderr = Join-Path $Out "stderr.log"
    $startedAt = Get-Date -Format s
    "[$startedAt] START $Name" | Tee-Object -FilePath $RunnerLog -Append
    $proc = Start-Process -FilePath $Python -ArgumentList $ArgsList -WorkingDirectory $WorkDir -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    $finishedAt = Get-Date -Format s
    if ($proc.ExitCode -ne 0) {
        "[$finishedAt] FAIL $Name exit=$($proc.ExitCode)" | Tee-Object -FilePath $RunnerLog -Append
        throw "Failed: $Name. See $stderr"
    }
    "[$finishedAt] DONE $Name" | Tee-Object -FilePath $RunnerLog -Append
}

function Get-PrecisionFromRecallF1 {
    param([double]$Recall, [double]$F1)
    $r = $Recall / 100.0
    $f = $F1 / 100.0
    $denom = 2.0 * $r - $f
    if ($denom -le 0) { return 0.0 }
    return ($f * $r / $denom) * 100.0
}

function Add-EvalSummary {
    param([int]$Ratio, [double]$Weight, [string]$EvalOut)
    $perClassPath = Join-Path $EvalOut "per_class_eval.csv"
    $rows = @(Import-Csv -LiteralPath $perClassPath | Where-Object { $_.model -eq "ensemble" })
    if ($rows.Count -eq 0) { throw "No ensemble rows in $perClassPath" }
    $acc = ($rows | Measure-Object -Property accuracy -Average).Average
    $macroF1 = ($rows | Measure-Object -Property f1 -Average).Average

    $minor = @($rows | Where-Object { [int]$_.class -ge 5 -and [int]$_.class -le 9 })
    $tail = @($rows | Where-Object { [int]$_.class -ge 7 -and [int]$_.class -le 9 })

    $minorRecall = ($minor | Measure-Object -Property accuracy -Average).Average
    $minorF1 = ($minor | Measure-Object -Property f1 -Average).Average
    $minorPrecVals = @($minor | ForEach-Object { Get-PrecisionFromRecallF1 -Recall ([double]$_.accuracy) -F1 ([double]$_.f1) })
    $minorPrecision = ($minorPrecVals | Measure-Object -Average).Average

    $tailRecall = ($tail | Measure-Object -Property accuracy -Average).Average
    $tailF1 = ($tail | Measure-Object -Property f1 -Average).Average
    $tailPrecVals = @($tail | ForEach-Object { Get-PrecisionFromRecallF1 -Recall ([double]$_.accuracy) -F1 ([double]$_.f1) })
    $tailPrecision = ($tailPrecVals | Measure-Object -Average).Average

    $relOut = $EvalOut.Replace($RepoRoot + "\", "").Replace("\", "/")
    $line = '{0},{1:F1},{2:F4},{3:F4},{4:F4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4},"{10}"' -f $Ratio,$Weight,$acc,$macroF1,$minorRecall,$minorPrecision,$minorF1,$tailRecall,$tailPrecision,$tailF1,$relOut
    Add-Content -LiteralPath $SummaryPath -Value $line
}

foreach ($Ratio in $Ratios) {
    $TrainOut = Join-Path $ResultRootAbs ("imb$Ratio\train_dual")
    $TrainArgs = @(
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
        "--ensemble-minor-weight", "0.7",
        "--seed", "$Seed",
        "--out", $TrainOut,
        "--num-workers", "0",
        "--no-progress"
    )
    Invoke-CommandRun -Name "train imb=$Ratio dual" -WorkDir $DualRoot -ArgsList $TrainArgs -Out $TrainOut

    $Checkpoint = Join-Path $TrainOut "checkpoint.pth.tar"
    if (-not (Test-Path -LiteralPath $Checkpoint)) {
        throw "Missing checkpoint: $Checkpoint"
    }

    foreach ($Weight in $Weights) {
        $WeightName = ("w{0:F1}" -f $Weight).Replace(".", "p")
        $EvalOut = Join-Path $ResultRootAbs ("imb$Ratio\eval_$WeightName")
        $EvalArgs = @(
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
            "--ensemble-minor-weight", ("{0:F1}" -f $Weight),
            "--resume", $Checkpoint,
            "--eval-only",
            "--seed", "$Seed",
            "--out", $EvalOut,
            "--num-workers", "0",
            "--no-progress"
        )
        Invoke-CommandRun -Name "eval imb=$Ratio weight=$Weight" -WorkDir $DualRoot -ArgsList $EvalArgs -Out $EvalOut
        Add-EvalSummary -Ratio $Ratio -Weight $Weight -EvalOut $EvalOut
    }
}

"finished=$(Get-Date -Format s)" | Tee-Object -FilePath $RunnerLog -Append
