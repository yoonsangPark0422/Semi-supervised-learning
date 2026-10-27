param([int]$ExistingSoftSeed2Pid = 0)
$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$PythonExe = 'C:\Users\FORYOUCOM\Desktop\DeCon\.venv-cu128\Scripts\python.exe'
$WorkingDir = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\dual'
$ResultsRoot = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\results'
$QueueDir = Join-Path $ResultsRoot 'dual_real_last_comparison_2seeds_each_20261025'
$Runs = @(
    @{Variant='soft'; Seed=2},
    @{Variant='hard'; Seed=1},
    @{Variant='hard'; Seed=2}
)
foreach ($Run in $Runs) {
    $Prefix = if ($Run.Variant -eq 'soft') { 'dual_real_last_lla_soft' } else { 'dual_real_last_2_lla_hard' }
    $Run.Out = Join-Path $ResultsRoot ($Prefix + '_validation_cifar10_e50_s512_seed' + $Run.Seed)
    $SourceDir = if ($Run.Variant -eq 'soft') { $PSScriptRoot } else { Join-Path $PSScriptRoot 'dual real last-2' }
    $Run.Script = Join-Path $SourceDir 'train_head_tail_exchange_sampler_alpha_val_split.py'
    $Run.Attach = ($Run.Variant -eq 'soft' -and $Run.Seed -eq 2 -and $ExistingSoftSeed2Pid -gt 0)
    if ((Test-Path -LiteralPath $Run.Out) -and !$Run.Attach) { throw "Refusing to overwrite $($Run.Out)" }
    if (!(Test-Path -LiteralPath $Run.Script)) { throw "Missing $($Run.Script)" }
}
if (Test-Path -LiteralPath $QueueDir) { throw "Queue directory already exists: $QueueDir" }
New-Item -ItemType Directory -Path $QueueDir | Out-Null
$Runs | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath (Join-Path $QueueDir 'manifest.json') -Encoding UTF8
function Write-QueueLog([string]$Message) {
    $Line = (Get-Date -Format o) + ' ' + $Message
    Add-Content -LiteralPath (Join-Path $QueueDir 'queue.log') -Value $Line -Encoding UTF8
    Write-Output $Line
}
try {
    foreach ($Run in $Runs) {
        $Name = $Run.Variant + '_seed' + $Run.Seed
        if (!$Run.Attach) { New-Item -ItemType Directory -Path $Run.Out | Out-Null }
        $TrainArgs = @(
            '-B', '-u', ('"' + $Run.Script + '"'),
            '--dataset', 'cifar10', '--seed', [string]$Run.Seed,
            '--num-labeled', '4000', '--expand-labels', '--imb-ratio', '100',
            '--arch', 'wideresnet', '--batch-size', '32', '--mu', '7', '--lr', '0.03',
            '--total-steps', '25600', '--eval-step', '512',
            '--dual-train-mode', 'dual_sampler_weighted_ce',
            '--minor-balanced-sampler-alpha', '1.2',
            '--val-samples-per-class', '500', '--diagnose-disagreement',
            '--num-workers', '2', '--no-progress', '--out', ('"' + $Run.Out + '"')
        )
        Write-QueueLog "START $Name"
        if ($Run.Attach) {
            $Existing = Get-CimInstance Win32_Process -Filter "ProcessId=$ExistingSoftSeed2Pid"
            if ($Existing.CommandLine -notlike ('*' + $Run.Out + '*')) { throw 'Existing training process mismatch' }
            $Process = Get-Process -Id $ExistingSoftSeed2Pid
            Write-QueueLog 'Attached to running soft seed 2; logs remain in original queue directory'
        } else {
            $Process = Start-Process -FilePath $PythonExe -ArgumentList $TrainArgs -WorkingDirectory $WorkingDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $QueueDir "$Name.stdout.log") -RedirectStandardError (Join-Path $QueueDir "$Name.stderr.log")
        }
        Write-QueueLog "PID $Name $($Process.Id)"
        $ProcessHandle = $Process.Handle
        $Process.WaitForExit()
        if (!$Run.Attach -and $Process.ExitCode -ne 0) { throw "$Name exited with code $($Process.ExitCode)" }
        $Metrics = Import-Csv -LiteralPath (Join-Path $Run.Out 'train_metrics.csv')
        if ($Metrics[-1].epoch -ne '50') { throw "$Name did not finish epoch 50" }
        if (!(Test-Path -LiteralPath (Join-Path $Run.Out 'checkpoint.pth.tar'))) { throw "$Name checkpoint missing" }
        Write-QueueLog "DONE $Name epoch=50"
    }
    Write-QueueLog 'ALL_DONE'
} catch {
    Write-QueueLog "FAILED $($_.Exception.Message)"
    throw
}
