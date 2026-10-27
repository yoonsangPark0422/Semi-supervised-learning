param(
    [int[]]$Ratios = @(50, 100),
    [int[]]$Seeds = @(1, 2, 3),
    [int]$Epochs = 50,
    [int]$EvalStep = 512,
    [int]$BatchSize = 32,
    [int]$Mu = 7,
    [double]$EnsembleMinorWeight = 0.8,
    [string]$ResultRoot = "results\ablation_imb50_100_seed1_2_3_e50_s512"
)

$ErrorActionPreference = "Stop"

$RepoRoot = $PSScriptRoot
$BaselineRoot = Join-Path $RepoRoot "baseline compare"
$DualRoot = Join-Path $RepoRoot "dual"
if ($env:PYTHON) {
    $Python = $env:PYTHON
} else {
    $RuntimePython = "C:\Users\FORYOUCOM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
    if (Test-Path -LiteralPath $RuntimePython) { $Python = $RuntimePython } else { $Python = "python" }
}

$ResultRootAbs = Join-Path $RepoRoot $ResultRoot
New-Item -ItemType Directory -Force -Path $ResultRootAbs | Out-Null
$RunnerLog = Join-Path $ResultRootAbs "runner.log"
$RunSummary = Join-Path $ResultRootAbs "run_summary.csv"
$AggregateSummary = Join-Path $ResultRootAbs "aggregate_summary.csv"
$TotalSteps = $Epochs * $EvalStep

if (-not (Test-Path -LiteralPath $RunnerLog)) {
    "started=$(Get-Date -Format s) seeds=$($Seeds -join ',') ratios=$($Ratios -join ',') epochs=$Epochs eval_step=$EvalStep total_steps=$TotalSteps batch_size=$BatchSize mu=$Mu ensemble_minor_weight=$EnsembleMinorWeight" | Set-Content -LiteralPath $RunnerLog
}
if (-not (Test-Path -LiteralPath $RunSummary)) {
    "ratio,seed,method,status,last_epoch,acc,macro_f1,minority_5_9_recall,minority_5_9_precision,minority_5_9_f1,tail_7_9_recall,tail_7_9_precision,tail_7_9_f1,out" | Set-Content -LiteralPath $RunSummary
}

$Methods = @(
    @{ Name = "baseline_fixmatch"; Kind = "baseline"; Mode = "" },
    @{ Name = "dual_none"; Kind = "dual"; Mode = "dual" },
    @{ Name = "dual_sampler"; Kind = "dual"; Mode = "dual_sampler" },
    @{ Name = "dual_weighted_ce"; Kind = "dual"; Mode = "dual_weighted_ce" },
    @{ Name = "dual_sampler_weighted_ce"; Kind = "dual"; Mode = "dual_sampler_weighted_ce" }
)

function Get-PrecisionFromRecallF1 {
    param([double]$Recall, [double]$F1)
    $r = $Recall / 100.0
    $f = $F1 / 100.0
    $denom = 2.0 * $r - $f
    if ($denom -le 0) { return 0.0 }
    return ($f * $r / $denom) * 100.0
}

function Get-GroupMetrics {
    param([object[]]$Rows, [int]$MinClass, [int]$MaxClass)
    $selected = @($Rows | Where-Object { [int]$_.class -ge $MinClass -and [int]$_.class -le $MaxClass })
    $recall = ($selected | Measure-Object -Property accuracy -Average).Average
    $f1 = ($selected | Measure-Object -Property f1 -Average).Average
    $precVals = @($selected | ForEach-Object { Get-PrecisionFromRecallF1 -Recall ([double]$_.accuracy) -F1 ([double]$_.f1) })
    $precision = ($precVals | Measure-Object -Average).Average
    return @{ Recall = $recall; Precision = $precision; F1 = $f1 }
}

function Add-Or-ReplaceSummaryRow {
    param(
        [int]$Ratio,
        [int]$Seed,
        [string]$Method,
        [string]$Status,
        [string]$Out
    )
    $metricsPath = Join-Path $Out "train_metrics.csv"
    $perClassPath = Join-Path $Out "per_class_eval.csv"
    if (-not (Test-Path -LiteralPath $metricsPath)) { throw "Missing $metricsPath" }
    if (-not (Test-Path -LiteralPath $perClassPath)) { throw "Missing $perClassPath" }

    $metricRows = @(Import-Csv -LiteralPath $metricsPath)
    $last = $metricRows[-1]
    $lastEpoch = [int]$last.epoch
    if ($Method -eq "baseline_fixmatch") {
        $acc = [double]$last.test_acc
        $macroF1 = [double]$last.macro_f1
        $modelName = "fixmatch"
    } else {
        $acc = [double]$last.test_acc_final
        $macroF1 = [double]$last.macro_f1_final
        $modelName = "ensemble"
    }

    $pcRows = @(Import-Csv -LiteralPath $perClassPath | Where-Object { [int]$_.epoch -eq $lastEpoch -and $_.model -eq $modelName })
    $minor = Get-GroupMetrics -Rows $pcRows -MinClass 5 -MaxClass 9
    $tail = Get-GroupMetrics -Rows $pcRows -MinClass 7 -MaxClass 9
    $relOut = $Out.Replace($RepoRoot + "\", "").Replace("\", "/")
    $newLine = '{0},{1},{2},{3},{4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4},{10:F4},{11:F4},{12:F4},"{13}"' -f $Ratio,$Seed,$Method,$Status,$lastEpoch,$acc,$macroF1,$minor.Recall,$minor.Precision,$minor.F1,$tail.Recall,$tail.Precision,$tail.F1,$relOut

    $existing = @()
    if (Test-Path -LiteralPath $RunSummary) {
        $existing = @(Get-Content -LiteralPath $RunSummary | Where-Object { $_ -and ($_ -notmatch "^$Ratio,$Seed,$Method,") })
    }
    if ($existing.Count -eq 0) {
        $existing = @("ratio,seed,method,status,last_epoch,acc,macro_f1,minority_5_9_recall,minority_5_9_precision,minority_5_9_f1,tail_7_9_recall,tail_7_9_precision,tail_7_9_f1,out")
    }
    $existing + $newLine | Set-Content -LiteralPath $RunSummary
}

function Update-AggregateSummary {
    $rows = @(Import-Csv -LiteralPath $RunSummary | Where-Object { $_.status -eq "done" })
    $header = "ratio,method,n,acc_mean,acc_std,macro_f1_mean,macro_f1_std,minority_f1_mean,minority_f1_std,tail_f1_mean,tail_f1_std"
    $lines = @($header)
    foreach ($group in ($rows | Group-Object ratio,method)) {
        $items = @($group.Group)
        $ratio = $items[0].ratio
        $method = $items[0].method
        $n = $items.Count
        function Mean([double[]]$vals) { if ($vals.Count -eq 0) { return 0 }; return ($vals | Measure-Object -Average).Average }
        function Std([double[]]$vals) {
            if ($vals.Count -le 1) { return 0 }
            $m = Mean $vals
            return [math]::Sqrt((($vals | ForEach-Object { ($_ - $m) * ($_ - $m) }) | Measure-Object -Sum).Sum / ($vals.Count - 1))
        }
        $acc = [double[]]@($items | ForEach-Object { [double]$_.acc })
        $macro = [double[]]@($items | ForEach-Object { [double]$_.macro_f1 })
        $minor = [double[]]@($items | ForEach-Object { [double]$_.minority_5_9_f1 })
        $tail = [double[]]@($items | ForEach-Object { [double]$_.tail_7_9_f1 })
        $lines += ('{0},{1},{2},{3:F4},{4:F4},{5:F4},{6:F4},{7:F4},{8:F4},{9:F4},{10:F4}' -f $ratio,$method,$n,(Mean $acc),(Std $acc),(Mean $macro),(Std $macro),(Mean $minor),(Std $minor),(Mean $tail),(Std $tail))
    }
    $lines | Set-Content -LiteralPath $AggregateSummary
}

function Test-RunComplete {
    param([string]$Out)
    $metricsPath = Join-Path $Out "train_metrics.csv"
    if (-not (Test-Path -LiteralPath $metricsPath)) { return $false }
    $rows = @(Import-Csv -LiteralPath $metricsPath)
    if ($rows.Count -eq 0) { return $false }
    return ([int]$rows[-1].epoch -ge $Epochs)
}

function Invoke-TrainingRun {
    param(
        [int]$Ratio,
        [int]$Seed,
        [hashtable]$Method
    )
    $Out = Join-Path $ResultRootAbs ("imb$Ratio\seed_$Seed\$($Method.Name)")
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    if (Test-RunComplete -Out $Out) {
        "[$(Get-Date -Format s)] SKIP ratio=$Ratio seed=$Seed method=$($Method.Name) already complete" | Tee-Object -FilePath $RunnerLog -Append
        Add-Or-ReplaceSummaryRow -Ratio $Ratio -Seed $Seed -Method $Method.Name -Status "done" -Out $Out
        Update-AggregateSummary
        return
    }

    $stdout = Join-Path $Out "stdout.log"
    $stderr = Join-Path $Out "stderr.log"
    if ($Method.Kind -eq "baseline") {
        $WorkDir = $BaselineRoot
        $Args = @(
            "train.py", "--dataset", "cifar10", "--num-labeled", "4000",
            "--arch", "wideresnet", "--batch-size", "$BatchSize", "--mu", "$Mu",
            "--lr", "0.03", "--expand-labels", "--imb-ratio", "$Ratio",
            "--total-steps", "$TotalSteps", "--eval-step", "$EvalStep",
            "--ablation-preset", "fixmatch", "--seed", "$Seed", "--out", $Out,
            "--num-workers", "0", "--no-progress", "--no-save-checkpoint"
        )
    } else {
        $WorkDir = $DualRoot
        $Args = @(
            "train.py", "--dataset", "cifar10", "--num-labeled", "4000",
            "--arch", "wideresnet", "--batch-size", "$BatchSize", "--mu", "$Mu",
            "--lr", "0.03", "--expand-labels", "--imb-ratio", "$Ratio",
            "--total-steps", "$TotalSteps", "--eval-step", "$EvalStep",
            "--dual-train-mode", $Method.Mode,
            "--ensemble-minor-weight", ("{0:F1}" -f $EnsembleMinorWeight),
            "--seed", "$Seed", "--out", $Out,
            "--num-workers", "0", "--no-progress", "--no-save-checkpoint"
        )
    }

    "[$(Get-Date -Format s)] START ratio=$Ratio seed=$Seed method=$($Method.Name)" | Tee-Object -FilePath $RunnerLog -Append
    $proc = Start-Process -FilePath $Python -ArgumentList $Args -WorkingDirectory $WorkDir -Wait -PassThru -NoNewWindow -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    if ($proc.ExitCode -ne 0) {
        "[$(Get-Date -Format s)] FAIL ratio=$Ratio seed=$Seed method=$($Method.Name) exit=$($proc.ExitCode)" | Tee-Object -FilePath $RunnerLog -Append
        throw "Training failed. See $stderr"
    }
    "[$(Get-Date -Format s)] DONE ratio=$Ratio seed=$Seed method=$($Method.Name)" | Tee-Object -FilePath $RunnerLog -Append
    Add-Or-ReplaceSummaryRow -Ratio $Ratio -Seed $Seed -Method $Method.Name -Status "done" -Out $Out
    Update-AggregateSummary
}

foreach ($Ratio in $Ratios) {
    foreach ($Seed in $Seeds) {
        foreach ($Method in $Methods) {
            Invoke-TrainingRun -Ratio $Ratio -Seed $Seed -Method $Method
        }
    }
}

Update-AggregateSummary
"finished=$(Get-Date -Format s)" | Tee-Object -FilePath $RunnerLog -Append
