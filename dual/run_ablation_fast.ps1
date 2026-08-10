param(
    [int[]]$Seeds = @(5),
    [string]$ResultRoot = "results\ablation_fast"
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
    "--lr", "0.03",
    "--expand-labels",
    "--imb-ratio", "100",
    "--fast-ablation",
    "--no-save-checkpoint"
)

$Experiments = @(
    @{ Name = "01_fixmatch"; Preset = "fixmatch"; Extra = @() },
    @{ Name = "02_dual_only"; Preset = "dual_only"; Extra = @() },
    @{ Name = "03_balanced_sampler"; Preset = "dual_sampler"; Extra = @() },
    @{ Name = "04_class_weight"; Preset = "dual_sampler_weight"; Extra = @() },
    @{ Name = "05_minority_pseudo_bias"; Preset = "dual_sampler_weight_bias"; Extra = @() },
    @{ Name = "06_classwise_topk"; Preset = "dual_sampler_weight_bias_topk"; Extra = @() },
    @{ Name = "07_agreement_consistency"; Preset = "dual_agreement_consistency"; Extra = @() },
    @{ Name = "08_full"; Preset = "full"; Extra = @("--dual-bias-cotrain") }
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
