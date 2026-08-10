param(
    [int[]]$Seeds = @(5),
    [string]$ResultRoot = "results\factorial_compare",
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
    @{ Name = "00_fixmatch"; Extra = @("--ablation-preset", "fixmatch") },
    @{ Name = "01_dual_only"; Extra = @("--ablation-preset", "full", "--no-minor-balanced-labeled", "--no-minor-class-weight", "--no-minority-pseudo-bias", "--no-classwise-pseudo-mask", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "02_dual_sampler"; Extra = @("--ablation-preset", "full", "--no-minor-class-weight", "--no-minority-pseudo-bias", "--no-classwise-pseudo-mask", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "03_dual_class_weight"; Extra = @("--ablation-preset", "full", "--no-minor-balanced-labeled", "--no-minority-pseudo-bias", "--no-classwise-pseudo-mask", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "04_dual_semantic_proxy"; Extra = @("--ablation-preset", "full", "--no-minor-balanced-labeled", "--no-minor-class-weight", "--no-classwise-pseudo-mask", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "05_dual_topk"; Extra = @("--ablation-preset", "full", "--no-minor-balanced-labeled", "--no-minor-class-weight", "--no-minority-pseudo-bias", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "06_dual_sampler_class_weight"; Extra = @("--ablation-preset", "full", "--no-minority-pseudo-bias", "--no-classwise-pseudo-mask", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "07_dual_sampler_semantic_proxy"; Extra = @("--ablation-preset", "full", "--no-minor-class-weight", "--no-classwise-pseudo-mask", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "08_dual_sampler_topk"; Extra = @("--ablation-preset", "full", "--no-minor-class-weight", "--no-minority-pseudo-bias", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "09_dual_class_weight_semantic_proxy"; Extra = @("--ablation-preset", "full", "--no-minor-balanced-labeled", "--no-classwise-pseudo-mask", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "10_dual_class_weight_topk"; Extra = @("--ablation-preset", "full", "--no-minor-balanced-labeled", "--no-minority-pseudo-bias", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "11_dual_semantic_proxy_topk"; Extra = @("--ablation-preset", "full", "--no-minor-balanced-labeled", "--no-minor-class-weight", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "12_dual_sampler_class_weight_semantic_proxy"; Extra = @("--ablation-preset", "full", "--no-classwise-pseudo-mask", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "13_dual_sampler_class_weight_topk"; Extra = @("--ablation-preset", "full", "--no-minority-pseudo-bias", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "14_dual_sampler_semantic_proxy_topk"; Extra = @("--ablation-preset", "full", "--no-minor-class-weight", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "15_dual_class_weight_semantic_proxy_topk"; Extra = @("--ablation-preset", "full", "--no-minor-balanced-labeled", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "16_dual_sampler_class_weight_semantic_proxy_topk"; Extra = @("--ablation-preset", "full", "--no-dual-agreement", "--no-dual-consistency") },
    @{ Name = "17_full_dual_with_agreement_consistency"; Extra = @("--ablation-preset", "full") }
)

foreach ($Seed in $Seeds) {
    foreach ($Experiment in $Experiments) {
        $Out = Join-Path $ResultRoot ("seed_$Seed\$($Experiment.Name)")
        New-Item -ItemType Directory -Force -Path $Out | Out-Null
        $Args = $CommonArgs + @(
            "--seed", "$Seed",
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
