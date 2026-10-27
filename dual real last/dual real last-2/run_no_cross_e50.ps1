$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$PythonExe = 'C:\Users\FORYOUCOM\Desktop\DeCon\.venv-cu128\Scripts\python.exe'
$WorkingDir = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\dual'
$ResultsRoot = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\results'
$QueueDir = Join-Path $ResultsRoot 'dual_real_last_2_no_cross_queue'
$TrainFile = Join-Path $PSScriptRoot 'train_head_tail_exchange_sampler_alpha_val_split.py'
if (Test-Path -LiteralPath $QueueDir) { throw 'Queue already exists.' }
foreach ($Seed in @(1)) {
    $OutputDir = Join-Path $ResultsRoot "dual_real_last_2_lla_hard_no_cross_validation_cifar10_e50_s512_seed$Seed"
    if (Test-Path -LiteralPath $OutputDir) { throw "Output already exists: $OutputDir" }
}
New-Item -ItemType Directory -Path $QueueDir | Out-Null
function Write-QueueLog([string]$Message) {
    $Line = (Get-Date -Format o) + ' ' + $Message
    Add-Content -LiteralPath (Join-Path $QueueDir 'queue.log') -Value $Line -Encoding UTF8
    Write-Output $Line
}
try {
    foreach ($Seed in @(1)) {
        $OutputDir = Join-Path $ResultsRoot "dual_real_last_2_lla_hard_no_cross_validation_cifar10_e50_s512_seed$Seed"
        New-Item -ItemType Directory -Path $OutputDir | Out-Null
        $TrainArgs = @(
            '-B', '-u', ('"' + $TrainFile + '"'),
            '--dataset', 'cifar10', '--seed', [string]$Seed,
            '--num-labeled', '4000', '--expand-labels', '--imb-ratio', '100',
            '--arch', 'wideresnet', '--batch-size', '32', '--mu', '7', '--lr', '0.03',
            '--total-steps', '25600', '--eval-step', '512',
            '--dual-train-mode', 'dual_sampler_weighted_ce',
            '--minor-balanced-sampler-alpha', '1.2',
            '--val-samples-per-class', '500', '--diagnose-disagreement',
            '--disable-cross', '--num-workers', '2', '--no-progress', '--out', ('"' + $OutputDir + '"')
        )
        $TrainArgs | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $QueueDir "seed$Seed.arguments.json") -Encoding UTF8
        Write-QueueLog "START seed$Seed"
        $Process = Start-Process -FilePath $PythonExe -ArgumentList $TrainArgs -WorkingDirectory $WorkingDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $QueueDir "seed$Seed.stdout.log") -RedirectStandardError (Join-Path $QueueDir "seed$Seed.stderr.log")
        $ProcessHandle = $Process.Handle
        Write-QueueLog "PID seed$Seed $($Process.Id)"
        $Process.WaitForExit()
        if ($Process.ExitCode -ne 0) { throw "seed$Seed failed with exit code $($Process.ExitCode)" }
        $Metrics = Import-Csv -LiteralPath (Join-Path $OutputDir 'train_metrics.csv')
        if ($Metrics[-1].epoch -ne '50') { throw "seed$Seed did not reach epoch 50" }
        $Gates = Import-Csv -LiteralPath (Join-Path $OutputDir 'correction_gate_stats.csv')
        if ($Gates | Where-Object { [int]$_.gate_selected -ne 0 -or [int]$_.conflicting_self_suppressed -ne 0 }) { throw 'Unexpected self suppression' }
        if (!(Test-Path -LiteralPath (Join-Path $OutputDir 'checkpoint.pth.tar'))) { throw 'Missing checkpoint' }
        Write-QueueLog "DONE seed$Seed epoch=50"
    }
    Write-QueueLog 'ALL_DONE'
} catch {
    Write-QueueLog "FAILED $($_.Exception.Message)"
    throw
}
