param(
    [int[]]$Seeds = @(5),
    [string]$ResultRoot = "results\dual_mode_compare",
    [int]$Epochs = 200,
    [int]$EvalStep = 1024,
    [int]$BatchSize = 64,
    [int]$Mu = 7
)

$ErrorActionPreference = "Stop"

if ($env:PYTHON) {
    $Python = $env:PYTHON
} else {
    $Python = "python"
}

$TotalSteps = $Epochs * $EvalStep
$CommonArgs = @(
    "train.py",
    "--dataset", "cifar10",
    "--num-labeled", "4000",
    "--arch", "wideresnet",
    "--batch-size", "$BatchSize",
    "--mu", "$Mu",
    "--lr", "0.03",
    "--expand-labels",
    "--imb-ratio", "100",
    "--total-steps", "$TotalSteps",
    "--eval-step", "$EvalStep",
    "--no-progress"
)

$Experiments = @(
    @{ Name = "01_dual"; Mode = "dual" },
    @{ Name = "02_dual_sampler"; Mode = "dual_sampler" },
    @{ Name = "03_dual_sampler_weighted_ce"; Mode = "dual_sampler_weighted_ce" }
)

foreach ($Seed in $Seeds) {
    foreach ($Experiment in $Experiments) {
        $Out = Join-Path $ResultRoot ("seed_$Seed\$($Experiment.Name)")
        New-Item -ItemType Directory -Force -Path $Out | Out-Null
        $Args = $CommonArgs + @(
            "--seed", "$Seed",
            "--dual-train-mode", $Experiment.Mode,
            "--out", $Out
        )
        $Log = Join-Path $Out "train.log"
        "[$(Get-Date -Format s)] Starting seed=$Seed mode=$($Experiment.Mode)" |
            Tee-Object -FilePath $Log
        $StdoutLog = Join-Path $Out "stdout.tmp.log"
        $StderrLog = Join-Path $Out "stderr.tmp.log"
        $Process = Start-Process -FilePath $Python -ArgumentList $Args `
            -Wait -PassThru -NoNewWindow `
            -RedirectStandardOutput $StdoutLog `
            -RedirectStandardError $StderrLog
        Get-Content -Path $StdoutLog, $StderrLog -ErrorAction SilentlyContinue |
            Add-Content -Path $Log
        Remove-Item -Path $StdoutLog, $StderrLog -Force -ErrorAction SilentlyContinue
        if ($Process.ExitCode -ne 0) {
            throw "Training failed for mode=$($Experiment.Mode) seed=$Seed with exit code $($Process.ExitCode)"
        }
        "[$(Get-Date -Format s)] Finished seed=$Seed mode=$($Experiment.Mode)" |
            Tee-Object -FilePath $Log -Append
    }
}

