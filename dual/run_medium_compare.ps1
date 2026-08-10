param(
    [int[]]$Seeds = @(5),
    [string]$ResultRoot = "results\medium_compare",
    [int]$TotalSteps = 2048,
    [int]$EvalStep = 512,
    [int]$BatchSize = 32,
    [int]$Mu = 3
)

$ErrorActionPreference = "Stop"

$Python = "C:\Users\FORYOUCOM\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

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
    "--pseudo-warmup", "256",
    "--unsup-warmup", "512",
    "--minority-bias-warmup", "1024",
    "--no-progress",
    "--no-save-checkpoint"
)

$Experiments = @(
    @{ Name = "01_fixmatch"; Preset = "fixmatch"; Extra = @() },
    @{ Name = "02_dual_only"; Preset = "dual_only"; Extra = @() },
    @{ Name = "03_full"; Preset = "full"; Extra = @("--dual-bias-cotrain") }
)

foreach ($Seed in $Seeds) {
    foreach ($Experiment in $Experiments) {
        $Out = Join-Path $ResultRoot ("seed_$Seed\$($Experiment.Name)")
        New-Item -ItemType Directory -Force -Path $Out | Out-Null
        $Args = $CommonArgs + @(
            "--seed", "$Seed",
            "--ablation-preset", $Experiment.Preset,
            "--out", $Out
        ) + $Experiment.Extra
        $Log = Join-Path $Out "train.log"
        "[$(Get-Date -Format s)] Starting seed=$Seed $($Experiment.Name)" |
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
        $ExitCode = $Process.ExitCode
        if ($ExitCode -ne 0) {
            throw "Training failed for $($Experiment.Name) seed=$Seed with exit code $ExitCode"
        }
        "[$(Get-Date -Format s)] Finished seed=$Seed $($Experiment.Name)" |
            Tee-Object -FilePath $Log -Append
    }
}

& $Python "summarize_ablation.py" --root $ResultRoot
