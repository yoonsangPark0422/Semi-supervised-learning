$ErrorActionPreference = 'Stop'
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
$PythonExe = 'C:\Users\FORYOUCOM\Desktop\DeCon\.venv-cu128\Scripts\python.exe'
$TrainFile = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-upload-worktree\dual real last\dual real last-2\train_head_tail_exchange_sampler_alpha_val_split.py'
$WorkingDir = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\dual'
$OutputDir = 'C:\Users\FORYOUCOM\Desktop\Semi-supervised-learning-main\Semi-supervised-learning-main\results\dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed1'
if (Test-Path -LiteralPath $OutputDir) { throw 'Output already exists; refusing to overwrite.' }
New-Item -ItemType Directory -Path $OutputDir | Out-Null
Set-Location -LiteralPath $WorkingDir
$TrainArgs = @(
    '-B', '-u', $TrainFile,
    '--dataset', 'cifar10', '--seed', '1',
    '--num-labeled', '4000', '--expand-labels', '--imb-ratio', '100',
    '--arch', 'wideresnet', '--batch-size', '32', '--mu', '7', '--lr', '0.03',
    '--total-steps', '25600', '--eval-step', '512',
    '--dual-train-mode', 'dual_sampler_weighted_ce',
    '--minor-balanced-sampler-alpha', '1.2',
    '--val-samples-per-class', '500', '--diagnose-disagreement',
    '--num-workers', '2', '--no-progress', '--out', $OutputDir
)
& $PythonExe @TrainArgs
exit $LASTEXITCODE
