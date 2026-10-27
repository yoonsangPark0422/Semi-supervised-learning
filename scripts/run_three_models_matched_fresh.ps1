$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$PythonExe = 'C:\Users\FORYOUCOM\Desktop\DeCon\.venv-cu128\Scripts\python.exe'
$MainRoot = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main'
$WorkingDir = Join-Path $MainRoot 'dual'
$RepoRoot = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-upload-worktree'
$ResultRoot = Join-Path $MainRoot 'results\three_models_matched_fresh_cifar10_seed1'
$Audit = Get-Content -LiteralPath (Join-Path $ResultRoot 'split_audit.json') -Raw | ConvertFrom-Json
if (!$Audit.all_three_exact_indices_equal) { throw 'Data audit failed' }
$Common = @('--dataset','cifar10','--seed','1','--num-labeled','4000','--expand-labels',
    '--arch','wideresnet','--batch-size','32','--mu','7','--lr','0.03',
    '--wdecay','0.0005','--ema-decay','0.999',
    '--total-steps','25600','--eval-step','512','--val-samples-per-class','0',
    '--num-workers','2','--no-progress','--eval-final-only')
$DualArgs = @('--imb-ratio','100','--dual-train-mode','dual_sampler_weighted_ce',
    '--minor-balanced-sampler-alpha','1.2','--ensemble-minor-weight','0.8',
    '--major-cross-skip-minor-conf-threshold','-1')
$Runs = @(
    @{Name='new_dual'; Script=(Join-Path $WorkingDir 'train_head_tail_exchange_sampler_alpha_val_split.py'); Extra=$DualArgs},
    @{Name='real_last_dual_2'; Script=(Join-Path $RepoRoot 'dual real last\dual real last-2\train_head_tail_exchange_sampler_alpha_val_split.py'); Extra=$DualArgs},
    @{Name='decon'; Script='C:\Users\FORYOUCOM\Desktop\DeCon\train_dual_matched.py'; Extra=@(
        '--match-dual-data','--num-max','1614','--num-max-u','5000',
        '--imb-ratio-label','100','--imb-ratio-unlabel','1',
        '--tau','1.0','--ema-u','0.99','--tau3','2.0','--tau4','1.0')}
)
foreach ($Run in $Runs) {
    $Run.Out = Join-Path $ResultRoot $Run.Name
    if (Test-Path -LiteralPath $Run.Out) { throw "Refusing to overwrite $($Run.Out)" }
    if (!(Test-Path -LiteralPath $Run.Script)) { throw 'Training script missing' }
}
function Log([string]$Message) {
    $Line = (Get-Date -Format o) + ' ' + $Message
    Add-Content -LiteralPath (Join-Path $ResultRoot 'queue.log') -Value $Line -Encoding UTF8
    Write-Output $Line
}
try {
    foreach ($Run in $Runs) {
        New-Item -ItemType Directory -Path $Run.Out | Out-Null
        $TrainArgs = @('-B','-u',('"'+$Run.Script+'"')) + $Common + $Run.Extra + @('--out',('"'+$Run.Out+'"'))
        $TrainArgs | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run.Out 'arguments.json') -Encoding UTF8
        Get-FileHash -LiteralPath $Run.Script -Algorithm SHA256 | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Run.Out 'source_hash.json')
        Copy-Item -LiteralPath $Run.Script -Destination (Join-Path $Run.Out 'training_source.py')
        Log "START $($Run.Name) seed=1 epochs=50 fresh=True"
        $Process = Start-Process -FilePath $PythonExe -ArgumentList $TrainArgs -WorkingDirectory $WorkingDir -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $Run.Out 'stdout.log') -RedirectStandardError (Join-Path $Run.Out 'stderr.log')
        $ProcessHandle = $Process.Handle
        Log "PID $($Run.Name) $($Process.Id)"
        $Process.WaitForExit()
        if ($Process.ExitCode -ne 0) { throw "$($Run.Name) exit=$($Process.ExitCode)" }
        $Rows = @(Import-Csv -LiteralPath (Join-Path $Run.Out 'per_class_eval.csv'))
        if (!$Rows.Count -or ($Rows | Where-Object { [int]$_.epoch -ne 50 -or [double]$_.support -ne 1000 })) {
            throw 'Expected only final epoch test metrics with 1000 images per class'
        }
        if (!(Test-Path -LiteralPath (Join-Path $Run.Out 'checkpoint.pth.tar'))) { throw 'Checkpoint missing' }
        Log "DONE $($Run.Name) epoch=50 official_test=True"
    }
    Log 'ALL_DONE'
} catch {
    Log "FAILED $($_.Exception.Message)"
    throw
}
