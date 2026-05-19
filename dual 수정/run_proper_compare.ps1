param(
    [int[]]$Seeds = @(5),
    [string]$ResultRoot = "results\proper_compare",
    [int]$TotalSteps = 1048576,
    [int]$EvalStep = 1024,
    [int]$BatchSize = 64,
    [int]$Mu = 7
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
    "--no-progress"
)

$Experiments = @(
    @{ Name = "00_fixmatch"; Extra = @("--ablation-preset", "fixmatch") },
    @{ Name = "01_dual_only"; Extra = @("--ablation-preset", "full", "--no-minor-balanced-labeled", "--no-minor-class-weight", "--no-minority-pseudo-bias", "--no-classwise-pseudo-mask", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "02_best_sampler_class_weight_semantic_proxy_topk"; Extra = @("--ablation-preset", "full", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "03_full_dual_with_agreement_consistency"; Extra = @("--ablation-preset", "full") }
)

foreach ($Seed in $Seeds) {
    foreach ($Experiment in $Experiments) {
        $Out = Join-Path $ResultRoot ("seed_$Seed\$($Experiment.Name)")
        New-Item -ItemType Directory -Force -Path $Out | Out-Null
        $Args = $CommonArgs + @(
            "--seed", "$Seed",
            "--out", $Out
        ) + $Experiment.Extra

        $Checkpoint = Join-Path $Out "checkpoint.pth.tar"
        if (Test-Path $Checkpoint) {
            $Args += @("--resume", $Checkpoint)
        }

        $Log = Join-Path $Out "train.log"
        "[$(Get-Date -Format s)] Starting seed=$Seed $($Experiment.Name)" |
            Tee-Object -FilePath $Log -Append
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
